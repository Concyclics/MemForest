"""TreeBuilder: lifecycle-aware build pipeline for session, entity, and scene trees.

Three-phase architecture:
  Phase 1 — Structure (no LLM):  build_structure / ingest_structure
  Phase 2 — Summarize (LLM):     flush
  Phase 3 — Index (embedding):    embed_roots / build_node_index

Each phase can be called independently. Convenience wrappers
(build_from_fact_lists, ingest_session, flush_dirty_trees) run all
phases in sequence for backward compatibility.
"""

from __future__ import annotations

import datetime
import threading
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from src.build.entity_router import ENTITY_USER_TREE_ID, EntityRouter
from src.build.scene_router import SceneRouter
from src.build.summary_manager import SummaryManager
from src.build.tree import (
    build_tree_from_cells,
    build_tree_from_facts,
    collect_dirty_requests,
    fill_upper_level_inputs,
    insert_cell,
    insert_fact,
    rebuild_dirty_queues_from_flags,
)
from src.build.tree_types import MemTree, SummaryRequest, SummaryResult, TreeBuildResult
from src.config.tree_config import TreeConfig
from src.prompt.tree_prompts import build_tree_prompt_manager

if TYPE_CHECKING:
    from src.api.client import OpenAIChatClient, OpenAIEmbeddingClient
    from src.utils.types import ManagedFact
    from src.build.node_index import NodeIndex
    from src.config.config import MemForestConfig
    from src.utils.types import ManagedFact, MemCell


class TreeBuilder:
    """Batch and incremental builder with lifecycle-based entity routing.

    The build pipeline is split into three independent phases:

    **Phase 1 — Structure** (pure, no LLM calls):
      - ``build_structure``: batch build from fact lists
      - ``ingest_structure``: incremental insert from a new session

    **Phase 2 — Summarize** (LLM, parallelizable per level):
      - ``flush``: generate summaries for all dirty nodes

    **Phase 3 — Index** (embedding):
      - ``embed_roots``: update root summary embeddings
      - ``build_node_index``: full FAISS node index

    Convenience wrappers (``build_from_fact_lists``, ``ingest_session``,
    ``flush_dirty_trees``) run all three phases in sequence.
    """

    def __init__(
        self,
        *,
        config: "MemForestConfig",
        chat_client: "OpenAIChatClient",
        embedding_client: "OpenAIEmbeddingClient",
    ) -> None:
        self._config = config
        self._tree_cfg: TreeConfig = config.tree
        self._chat_client = chat_client
        self._embedding_client = embedding_client
        self._summary_manager = SummaryManager(
            chat_client=chat_client,
            config=self._tree_cfg.summary_manager,
            model_name=config.api.llm.model_name,
            prompt_manager=build_tree_prompt_manager(),
        )
        self._scene_router = SceneRouter(
            embedding_client=embedding_client,
            config=self._tree_cfg.scene,
        )
        self._entity_router = EntityRouter(config=self._tree_cfg.entity)
        self._trees: dict[str, MemTree] = {}
        self._fact_catalog: dict[str, "ManagedFact"] = {}
        self._session_alias_map: dict[str, str] = {}
        self._lock = threading.RLock()

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def trees(self) -> dict[str, MemTree]:
        with self._lock:
            return dict(self._trees)

    @property
    def scene_router(self) -> SceneRouter:
        return self._scene_router

    @property
    def entity_router(self) -> EntityRouter:
        return self._entity_router

    @property
    def summary_manager(self) -> SummaryManager:
        return self._summary_manager

    @property
    def session_alias_map(self) -> dict[str, str]:
        with self._lock:
            return dict(self._session_alias_map)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 1 — Structure (no LLM)
    # ═══════════════════════════════════════════════════════════════════════

    def build_structure(
        self,
        *,
        all_facts: "list[ManagedFact]",
        session_id_to_cells: dict[str, "list[MemCell]"] | None = None,
        cell_to_facts: dict[str, list[str]] | None = None,
        scene_post_route_hook: "Callable[[SceneRouter, Callable[[str], ManagedFact | None]], None] | None" = None,
    ) -> list[str]:
        """Phase 1: build all tree structures from facts and cells.

        Routes facts through entity/scene routers, constructs trees with
        single-payload L0 leaves, and marks all nodes dirty. No LLM calls
        unless ``scene_post_route_hook`` runs one (e.g. scene relabel / judge).

        The hook fires after ``scene_router.assign_many`` but before scene
        buckets are frozen into tree structures, so hook mutations (label
        rewrites, cluster merges via ``judge_merge_candidates``) flow into the
        tree labels and bucket grouping. After the hook runs, scene buckets
        are re-derived from the router's current state.

        Returns:
            List of tree_ids that were built.
        """
        session_id_to_cells = session_id_to_cells or {}
        cell_to_facts = cell_to_facts or {}
        sorted_facts = sorted(
            all_facts,
            key=lambda f: (f.time_start if f.time_start is not None else 0.0, f.fact_id),
        )
        self._fact_catalog = {fact.fact_id: fact for fact in sorted_facts}
        self._scene_router.bootstrap(sorted_facts)
        self._ensure_session_aliases(session_id_to_cells.keys())

        # ── routing ──────────────────────────────────────────────────────
        for fact in sorted_facts:
            self._entity_router.assign(fact)
        self._scene_router.assign_many(sorted_facts)

        # ── optional scene post-processing (relabel / judge) ────────────
        if scene_post_route_hook is not None:
            scene_post_route_hook(self._scene_router, self._fact_catalog.get)

        # Re-derive scene buckets from the router's current state. If the
        # hook merged or dropped clusters, this picks up the survivors
        # automatically; otherwise the result is identical to the assignment
        # map above.
        scene_buckets: dict[str, list["ManagedFact"]] = defaultdict(list)
        for cluster_id in self._scene_router.all_cluster_ids():
            cluster = self._scene_router.get_cluster(cluster_id)
            if cluster is None:
                continue
            for fid in cluster.fact_ids:
                fact = self._fact_catalog.get(fid)
                if fact is not None:
                    scene_buckets[cluster_id].append(fact)

        suppressed = self._compute_suppressed_entity_keys(scene_buckets)
        self._entity_router.apply_suppression(suppressed)
        entity_buckets = self._build_entity_buckets(sorted_facts)

        # ── build tree structures ────────────────────────────────────────
        all_trees: list[MemTree] = []

        for tree_id, facts in entity_buckets.items():
            label = "user" if tree_id == ENTITY_USER_TREE_ID else tree_id.replace("entity:", "").replace("_", " ")
            k = self._tree_cfg.entity.user_k if tree_id == ENTITY_USER_TREE_ID else self._tree_cfg.entity.entity_k
            tree, _ = build_tree_from_facts(
                tree_id=tree_id,
                tree_type="entity",
                tree_key=tree_id,
                label=label,
                k=k,
                facts=facts,
            )
            all_trees.append(tree)

        for cluster_id, facts in scene_buckets.items():
            cluster = self._scene_router.get_cluster(cluster_id)
            label = cluster.label if cluster else cluster_id.replace("scene:", "")
            tree_id = f"tree:{cluster_id}"
            tree, _ = build_tree_from_facts(
                tree_id=tree_id,
                tree_type="scene",
                tree_key=cluster_id,
                label=label,
                k=self._tree_cfg.scene.k,
                facts=facts,
            )
            all_trees.append(tree)

        for session_id, cells in session_id_to_cells.items():
            public_session_id = self._public_session_id(session_id)
            tree_id = f"session:{public_session_id}"
            public_cells = self._alias_cells(cells, public_session_id)
            tree, _ = build_tree_from_cells(
                tree_id=tree_id,
                session_id=public_session_id,
                k=self._tree_cfg.session.k,
                cells=public_cells,
                cell_to_facts=cell_to_facts,
            )
            all_trees.append(tree)

        with self._lock:
            self._trees = {tree.tree_id: tree for tree in all_trees}

        return [tree.tree_id for tree in all_trees]

    def ingest_structure(
        self,
        *,
        new_facts: "list[ManagedFact]",
        cells: "list[MemCell]",
        cell_to_facts: dict[str, list[str]] | None = None,
    ) -> set[str]:
        """Phase 1: incrementally insert facts/cells into existing trees.

        Routes new facts, inserts into existing trees (or creates new ones),
        marks affected nodes dirty. No LLM calls.

        Can be called multiple times before a single ``flush()``.

        Returns:
            Set of dirty tree_ids that were structurally modified.
        """
        cell_to_facts = cell_to_facts or {}
        dirty_trees: set[str] = set()
        sorted_facts = sorted(
            new_facts,
            key=lambda f: (f.time_start if f.time_start is not None else 0.0, f.fact_id),
        )
        for fact in sorted_facts:
            self._fact_catalog[fact.fact_id] = fact
        if cells:
            self._ensure_session_aliases([cells[0].session_id])

        if not self._scene_router.all_cluster_ids():
            self._scene_router.bootstrap(sorted_facts)

        # ── scene routing + insert ───────────────────────────────────────
        scene_assignments = self._scene_router.assign_many(sorted_facts)
        for fact in sorted_facts:
            self._entity_router.assign(fact)
            for cluster_id in scene_assignments.get(fact.fact_id, []):
                tree_id = f"tree:{cluster_id}"
                tree = self._trees.get(tree_id)
                if tree is None:
                    cluster = self._scene_router.get_cluster(cluster_id)
                    label = cluster.label if cluster else cluster_id.replace("scene:", "")
                    tree, _ = build_tree_from_facts(
                        tree_id=tree_id,
                        tree_type="scene",
                        tree_key=cluster_id,
                        label=label,
                        k=self._tree_cfg.scene.k,
                        facts=[fact],
                    )
                    self._trees[tree_id] = tree
                else:
                    insert_fact(tree, fact)
                dirty_trees.add(tree_id)

        # ── entity routing + insert ──────────────────────────────────────
        for tree_id, facts in self._build_entity_buckets(sorted_facts).items():
            tree = self._trees.get(tree_id)
            if tree is None:
                label = "user" if tree_id == ENTITY_USER_TREE_ID else tree_id.replace("entity:", "").replace("_", " ")
                k = self._tree_cfg.entity.user_k if tree_id == ENTITY_USER_TREE_ID else self._tree_cfg.entity.entity_k
                candidate = self._entity_router.candidate(tree_id.replace("entity:", ""))
                facts_for_tree = facts
                if candidate and candidate.distinct_fact_count > len(facts):
                    facts_for_tree = [
                        self._fact_catalog[fid]
                        for fid in candidate.fact_ids
                        if fid in self._fact_catalog
                    ]
                tree, _ = build_tree_from_facts(
                    tree_id=tree_id,
                    tree_type="entity",
                    tree_key=tree_id,
                    label=label,
                    k=k,
                    facts=sorted(
                        facts_for_tree,
                        key=lambda fact: (fact.time_start if fact.time_start is not None else 0.0, fact.fact_id),
                    ),
                )
                self._trees[tree_id] = tree
            else:
                for fact in facts:
                    insert_fact(tree, fact)
            dirty_trees.add(tree_id)

        # ── session tree insert ──────────────────────────────────────────
        if cells:
            session_id = cells[0].session_id
            public_session_id = self._public_session_id(session_id)
            public_cells = self._alias_cells(cells, public_session_id)
            tree_id = f"session:{public_session_id}"
            tree = self._trees.get(tree_id)
            if tree is None:
                tree, _ = build_tree_from_cells(
                    tree_id=tree_id,
                    session_id=public_session_id,
                    k=self._tree_cfg.session.k,
                    cells=public_cells,
                    cell_to_facts=cell_to_facts,
                )
                self._trees[tree_id] = tree
            else:
                for cell in public_cells:
                    insert_cell(tree, cell, cell_to_facts.get(cell.cell_id, []))
            dirty_trees.add(tree_id)

        return dirty_trees

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2 — Summarize (LLM, per-level parallel)
    # ═══════════════════════════════════════════════════════════════════════

    def flush(
        self,
        *,
        tree_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> list[str]:
        """Phase 2: generate summaries for all dirty nodes.

        Collects dirty nodes across specified trees (or all trees),
        processes them bottom-up by level:
          - L0 entity/scene: passthrough fact_text (no LLM)
          - L0 session: LLM summarize cell text
          - L1+: LLM summarize child summaries (after lower level completes)

        Returns:
            List of tree_ids that were flushed.
        """
        trees_to_flush = [
            self._trees[tree_id]
            for tree_id in (tree_ids or self._trees.keys())
            if tree_id in self._trees
        ]
        if not trees_to_flush:
            return []

        all_requests_by_level = [
            collect_dirty_requests(tree, self._fact_catalog)
            for tree in trees_to_flush
        ]
        self._run_summaries_bottom_up(
            trees_to_flush,
            all_requests_by_level,
            self._fact_catalog,
        )
        return [tree.tree_id for tree in trees_to_flush]

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 3 — Index (embedding)
    # ═══════════════════════════════════════════════════════════════════════

    def embed_roots(
        self,
        *,
        tree_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> int:
        """Phase 3: embed root summaries for FAISS recall.

        Args:
            tree_ids: Trees to embed. Defaults to all trees.

        Returns:
            Number of trees whose roots were embedded.
        """
        trees_to_embed = [
            self._trees[tree_id]
            for tree_id in (tree_ids or self._trees.keys())
            if tree_id in self._trees
        ]
        if not trees_to_embed:
            return 0
        self._update_root_embeddings(trees_to_embed)
        return len(trees_to_embed)

    def build_node_index(
        self,
        trees: "list[MemTree] | None" = None,
        *,
        index_dir: Path,
        vector_dim: int = 1024,
    ) -> "NodeIndex":
        """Phase 3: build and save a NodeIndex from the current (or provided) trees.

        Embeds all non-L0 nodes in batched API calls.
        - FAISS search index: root-only (one entry per tree for cleaner ranking).
        - emb_store: all non-L0 nodes (root + L1 + L2) for TreeBrowser navigation.

        Args:
            trees: Trees to index. Defaults to all trees built so far.
            index_dir: Directory to save the NodeIndex files.
            vector_dim: Embedding dimension (must match embedding_client output).

        Returns:
            Populated NodeIndex (already saved to index_dir).
        """
        from src.build.node_index import NodeIndex, NodeEntry

        if trees is None:
            with self._lock:
                trees = list(self._trees.values())

        node_index = NodeIndex(index_dir=Path(index_dir), vector_dim=vector_dim)

        # Collect all non-L0 nodes for embedding (needed by browse).
        # L0 non-root nodes are never embedded.
        # Depth-0 scene trees (root=L0, single-fact) are embedded for browse
        # but excluded from FAISS search — their raw-fact embeddings are too
        # specific and crowd out larger trees in ranking.
        to_embed: list[tuple[str, str, str, bool]] = []  # (tree_id, node_id, level_str, searchable)
        summaries: list[str] = []
        for tree in trees:
            root_id = tree.root_node_id
            root_node = tree.nodes.get(root_id)
            is_depth0_scene = (
                tree.tree_type == "scene"
                and root_node is not None
                and root_node.level == 0
            )
            for node_id, node in tree.nodes.items():
                if not node.summary:
                    continue
                is_l0_non_root = node.level == 0 and node_id != root_id
                if is_l0_non_root:
                    continue
                is_root = node_id == root_id
                if is_root:
                    level_str = "root"
                elif node.level == 1:
                    level_str = "L1"
                elif node.level == 2:
                    level_str = "L2"
                else:
                    level_str = f"L{node.level}"
                searchable = is_root and not is_depth0_scene
                to_embed.append((tree.tree_id, node_id, level_str, searchable))
                summaries.append(node.summary)

        if to_embed:
            embeddings = self._embedding_client.embed_texts(summaries)
            for (tree_id, node_id, level_str, searchable), emb in zip(to_embed, embeddings):
                entry = NodeEntry(node_id=node_id, tree_id=tree_id, level=level_str)
                node_index.add_node(entry, emb, searchable=searchable)

        node_index.save(Path(index_dir))
        return node_index

    # ═══════════════════════════════════════════════════════════════════════
    # Convenience wrappers (backward-compatible, run all 3 phases)
    # ═══════════════════════════════════════════════════════════════════════

    def build_from_fact_lists(
        self,
        *,
        all_facts: "list[ManagedFact]",
        session_id_to_cells: dict[str, "list[MemCell]"] | None = None,
        cell_to_facts: dict[str, list[str]] | None = None,
        show_progress: bool = True,
        scene_post_route_hook: "Callable[[SceneRouter, Callable[[str], ManagedFact | None]], None] | None" = None,
    ) -> list[TreeBuildResult]:
        """Convenience: Phase 1 + 2 + 3 in sequence. Backward-compatible."""
        start = time.time()

        # Phase 1: structure
        tree_ids = self.build_structure(
            all_facts=all_facts,
            session_id_to_cells=session_id_to_cells,
            cell_to_facts=cell_to_facts,
            scene_post_route_hook=scene_post_route_hook,
        )

        # Phase 2: summarize
        self.flush(tree_ids=tree_ids)

        # Phase 3: embed roots
        self.embed_roots(tree_ids=tree_ids)

        elapsed = time.time() - start
        return self._make_build_results(tree_ids, elapsed)

    def ingest_session(
        self,
        *,
        new_facts: "list[ManagedFact]",
        cells: "list[MemCell]",
        cell_to_facts: dict[str, list[str]] | None = None,
        show_progress: bool = False,
    ) -> list[TreeBuildResult]:
        """Convenience: Phase 1 + 2 + 3 in sequence. Backward-compatible."""
        start = time.time()

        # Phase 1: structure
        dirty_trees = self.ingest_structure(
            new_facts=new_facts,
            cells=cells,
            cell_to_facts=cell_to_facts,
        )

        # Phase 2: summarize
        self.flush(tree_ids=dirty_trees)

        # Phase 3: embed roots
        self.embed_roots(tree_ids=dirty_trees)

        elapsed = time.time() - start
        return self._make_build_results(dirty_trees, elapsed)

    def flush_dirty_trees(
        self,
        *,
        tree_ids: set[str] | list[str] | tuple[str, ...] | None = None,
        elapsed_start: float | None = None,
    ) -> list[TreeBuildResult]:
        """Convenience: Phase 2 + 3 in sequence. Backward-compatible."""
        # Phase 2: summarize
        flushed = self.flush(tree_ids=tree_ids)

        # Phase 3: embed roots
        self.embed_roots(tree_ids=flushed)

        elapsed = (
            max(0.0, time.time() - elapsed_start)
            if elapsed_start is not None
            else 0.0
        )
        return self._make_build_results(flushed, elapsed)

    # ═══════════════════════════════════════════════════════════════════════
    # Internal methods
    # ═══════════════════════════════════════════════════════════════════════

    def _make_build_results(
        self,
        tree_ids: set[str] | list[str],
        elapsed: float,
    ) -> list[TreeBuildResult]:
        results: list[TreeBuildResult] = []
        for tree_id in tree_ids:
            tree = self._trees.get(tree_id)
            if tree is None:
                continue
            results.append(TreeBuildResult(
                tree_id=tree.tree_id,
                tree_type=tree.tree_type,
                label=tree.label,
                node_count=len(tree.nodes),
                leaf_count=len(tree.session_leaves) or len(tree.fact_ids_ordered),
                summary_calls=self._summary_manager.total_calls,
                elapsed_sec=elapsed,
            ))
        return results

    def _build_entity_buckets(self, facts: list["ManagedFact"]) -> dict[str, list["ManagedFact"]]:
        buckets: dict[str, list["ManagedFact"]] = defaultdict(list)
        active_keys = set(self._entity_router.all_active_entity_keys())
        for fact in facts:
            if fact.origin == "user":
                buckets[ENTITY_USER_TREE_ID].append(fact)
            for entity in fact.entities:
                normalized = _normalize_entity_key(entity)
                if normalized and normalized in active_keys:
                    buckets[f"entity:{normalized}"].append(fact)
        return buckets

    def _compute_suppressed_entity_keys(
        self,
        scene_buckets: dict[str, list["ManagedFact"]],
    ) -> set[str]:
        scene_fact_sets = [set(fact.fact_id for fact in facts) for facts in scene_buckets.values() if facts]
        suppressed: set[str] = set()
        for candidate in self._entity_router.iter_candidates():
            if candidate.state != "active":
                continue
            if candidate.entity_key == "user":
                continue
            if candidate.distinct_fact_count > self._tree_cfg.entity.suppress_below_facts:
                continue
            fact_ids = set(candidate.fact_ids)
            if any(fact_ids and fact_ids.issubset(scene_fact_ids) for scene_fact_ids in scene_fact_sets):
                suppressed.add(candidate.entity_key)
        return suppressed

    def _run_summaries_bottom_up(
        self,
        trees: list[MemTree],
        all_requests_by_level: list[dict[int, list[SummaryRequest]]],
        fact_map: "dict[str, ManagedFact]",
    ) -> None:
        if not trees:
            return
        max_level = max((max(requests.keys()) for requests in all_requests_by_level if requests), default=0)

        # Pre-populate results_by_node with existing node summaries, stripped of
        # any previously-prepended headers.  This ensures fill_upper_level_inputs
        # never falls back to a headered tree.nodes[cid].summary when assembling
        # input text for higher-level (L1+) LLM calls during incremental updates.
        results_by_node: dict[str, str] = {}
        for tree in trees:
            for node in tree.nodes.values():
                if node.summary:
                    results_by_node[node.node_id] = node.summary

        # Build a fast tree_id → tree lookup used for L0 passthrough and result application.
        tree_by_id: dict[str, "MemTree"] = {t.tree_id: t for t in trees}

        for level in range(max_level + 1):
            llm_requests: list[SummaryRequest] = []  # requests that actually need an LLM call
            for tree_idx, request_by_level in enumerate(all_requests_by_level):
                if level not in request_by_level:
                    continue
                tree = trees[tree_idx]
                if level == 0:
                    for req in request_by_level[level]:
                        if not req.input_text.strip():
                            continue
                        # L0 passthrough: all tree types skip LLM at leaf level.
                        # Entity/scene L0 = single atomic fact, session L0 = 2-turn
                        # conversation text.  Data shows session L0 LLM summaries
                        # actually expand text (0.7x compression ratio), so raw
                        # passthrough is both cheaper and more faithful.  Real
                        # compression begins at L1 where k children are merged.
                        node = tree.nodes.get(req.node_id)
                        if node:
                            node.summary = req.input_text
                            node.summary_dirty = False
                        results_by_node[req.node_id] = req.input_text
                else:
                    updated = fill_upper_level_inputs(tree, {level: request_by_level[level]}, results_by_node)
                    llm_requests.extend(updated.get(level, []))

            llm_requests = [r for r in llm_requests if r.input_text.strip()]
            if not llm_requests:
                continue
            batch_results = self._summary_manager.generate_summaries(llm_requests, show_progress=False)
            for result in batch_results:
                if result.error:
                    continue
                results_by_node[result.node_id] = result.summary
                t = tree_by_id.get(result.tree_id)
                if t:
                    node = t.nodes.get(result.node_id)
                    if node:
                        node.summary = result.summary
                        node.summary_dirty = False
        for tree in trees:
            rebuild_dirty_queues_from_flags(tree)

        # After all summaries are settled, prepend compact date+topic headers
        # to every internal node (L1+) so that root-node embeddings capture the
        # time span.  results_by_node intentionally stays header-free to avoid
        # polluting higher-level LLM inputs — headers are applied only to the
        # final node.summary values here.
        for tree in trees:
            self._prepend_node_headers(tree)

    @staticmethod
    def _prepend_node_headers(tree: "MemTree") -> None:
        """Prepend '[date_range | label]' header to every L1+ node summary."""
        for node in tree.nodes.values():
            if node.level == 0 or not node.summary:
                continue
            date_range = _make_date_range_str(node.time_start, node.time_end)
            label = tree.label.replace("_", " ") if tree.label else ""
            if date_range and label:
                header = f"[{date_range} | {label}]"
            elif date_range:
                header = f"[{date_range}]"
            elif label:
                header = f"[{label}]"
            else:
                continue
            # Avoid double-headers on incremental rebuilds
            if not node.summary.startswith("["):
                node.summary = f"{header} {node.summary}"

    def _update_root_embeddings(self, trees: list[MemTree]) -> None:
        to_embed = [(tree, tree.nodes.get(tree.root_node_id)) for tree in trees]
        to_embed = [(tree, node) for tree, node in to_embed if node and node.summary and tree.lifecycle_state == "active"]
        if not to_embed:
            return
        embeddings = self._embedding_client.embed_texts([node.summary for _, node in to_embed])
        for (tree, _), embedding in zip(to_embed, embeddings):
            tree.centroid = embedding

    def _ensure_session_aliases(self, session_ids) -> None:
        with self._lock:
            for session_id in session_ids:
                if session_id not in self._session_alias_map:
                    alias = f"sess_{len(self._session_alias_map) + 1:04d}"
                    self._session_alias_map[str(session_id)] = alias

    def _public_session_id(self, session_id: str) -> str:
        with self._lock:
            alias = self._session_alias_map.get(str(session_id))
            if alias is None:
                alias = f"sess_{len(self._session_alias_map) + 1:04d}"
                self._session_alias_map[str(session_id)] = alias
            return alias

    @staticmethod
    def _alias_cells(cells: list["MemCell"], public_session_id: str) -> list["MemCell"]:
        return [replace(cell, session_id=public_session_id) for cell in cells]


def _normalize_entity_key(entity: str) -> str | None:
    text = str(entity).strip().lower().replace("&", " and ")
    if not text or text in {"user", "assistant", "unknown"}:
        return None
    out = []
    last_was_sep = False
    for ch in text:
        if ch.isalnum():
            out.append(ch)
            last_was_sep = False
        else:
            if not last_was_sep:
                out.append("_")
            last_was_sep = True
    normalized = "".join(out).strip("_")
    return normalized or None


def _make_date_range_str(time_start: float, time_end: float) -> str:
    """Return a compact date-range string like 'Jan 2024 – Mar 2024'.

    Returns an empty string if timestamps are missing or invalid.
    """
    if not time_start or not time_end or time_start <= 0 or time_end <= 0:
        return ""
    try:
        utc = datetime.timezone.utc
        start_dt = datetime.datetime.fromtimestamp(time_start, tz=utc)
        end_dt = datetime.datetime.fromtimestamp(time_end, tz=utc)
        start_str = start_dt.strftime("%b %Y")
        end_str = end_dt.strftime("%b %Y")
        if start_str == end_str:
            return start_str
        # Same year: "Jan – Mar 2024"
        if start_dt.year == end_dt.year:
            return f"{start_dt.strftime('%b')} – {end_str}"
        return f"{start_str} – {end_str}"
    except (OSError, OverflowError, ValueError):
        return ""


