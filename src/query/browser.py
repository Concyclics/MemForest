"""TreeBrowser: true beam search from root to L0 leaf facts.

Algorithm — global beam search:
  Maintain a set of *frontier* nodes (initially just root).
  At each step, expand ALL frontier nodes' children, score every child globally,
  keep only the top-*beam* children as the new frontier.
  Repeat until all frontier nodes are L0 leaves.

  This guarantees the final L0 set has at most *beam* nodes, regardless of tree
  depth.  (The old implementation kept top-beam per parent, giving beam^depth.)

Design principles (from ablation):
  - Always descend to L0 (43 % facts compressed away at internal layers)
  - Pure embedding cos_sim per node (root guidance ≈ child_only, < 0.8 pp diff)
  - Optional LLM-guided mode for browse_types: anchor_a, anchor_b, preference

Node embedding lookup: NodeIndex.get_embedding(node_id) — O(1), no API call.
Sub-query embedding: cached within one browse_all() call to avoid duplicate embeds.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from src.build.tree_types import MemTree, MemTreeNode
from src.query.planner import BROWSE_TYPE_ANCHOR_A, BROWSE_TYPE_ANCHOR_B, BROWSE_TYPE_PREFERENCE

if TYPE_CHECKING:
    from src.api.client import OpenAIChatClient, OpenAIEmbeddingClient
    from src.build.node_index import NodeIndex
    from src.build.tree_store import TreeStore
    from src.config.query_config import BrowseConfig
    from src.query.planner import BrowsePlan

_LLM_BROWSE_TYPES = {BROWSE_TYPE_ANCHOR_A, BROWSE_TYPE_ANCHOR_B, BROWSE_TYPE_PREFERENCE}


class TreeBrowser:
    """True beam search on MemTree structures. Always reaches L0 single-payload leaves."""

    def __init__(
        self,
        *,
        node_index: "NodeIndex",
        tree_store: "TreeStore",
        config: "BrowseConfig",
        embedding_client: "OpenAIEmbeddingClient | None" = None,
        chat_client: "OpenAIChatClient | None" = None,
    ) -> None:
        self._node_index = node_index
        self._tree_store = tree_store
        self._config = config
        self._emb_client = embedding_client
        self._chat_client = chat_client

    def browse(
        self,
        plan: "BrowsePlan",
        *,
        query_emb: list[float],
        beam_width: int | None = None,
        sub_query_emb_cache: dict[str, list[float]] | None = None,
    ) -> list[str]:
        """Descend tree guided by plan.sub_query and return leaf fact_ids (or cell_ids).

        Args:
            plan: BrowsePlan with tree_id, sub_query, browse_type.
            query_emb: Pre-computed query embedding (normalized).
            beam_width: Override config.beam_width. None = use config default.
            sub_query_emb_cache: Shared dict for sub_query embedding reuse within a batch.

        Returns:
            List of fact_ids (entity/scene trees) or cell_ids (session trees).
        """
        tree = self._tree_store.get(plan.tree_id)
        if tree is None:
            return []

        beam = beam_width if beam_width is not None else self._config.beam_width
        root = tree.nodes.get(tree.root_node_id)
        if root is None:
            return []

        # L0 root — nothing to search
        if root.level == 0:
            return list(root.child_ids)

        # Compute sub_query embedding (reuse if same as original query)
        sub_query_emb = self._get_sub_query_emb(
            plan, query_emb, sub_query_emb_cache
        )

        # Choose ranking function
        #
        # llm_guided=True enables LLM-driven child ranking. The browse_type
        # gate (_LLM_BROWSE_TYPES) historically restricted LLM to anchor/
        # preference trees only. Setting llm_guided_all_types=True bypasses
        # that gate so every tree uses the LLM — matches ablation F4 (llm+subq)
        # which reached 79.2% answer correctness on LongMemEval 60q.
        use_llm = (
            self._config.llm_guided
            and self._chat_client is not None
            and (
                self._config.llm_guided_all_types
                or plan.browse_type in _LLM_BROWSE_TYPES
            )
        )

        # ── True beam search: global top-beam across levels ────────────────
        #
        # Frontier = nodes still to expand.  settled_l0 = final L0 nodes.
        # Each iteration:
        #   1. Expand every frontier node → score all children globally.
        #   2. Keep the global top-(beam − |settled_l0|) children.
        #   3. L0 children move to settled; internal children become new frontier.
        # Stop when frontier is empty or beam budget exhausted.

        settled_l0: list[MemTreeNode] = []
        frontier: list[MemTreeNode] = [root]

        while frontier:
            remaining = beam - len(settled_l0)
            if remaining <= 0:
                break

            # Score all children of all frontier nodes
            candidates: list[tuple[float, MemTreeNode]] = []
            for node in frontier:
                children = self._get_node_children(tree, node)

                if use_llm:
                    ranked = self._rank_children_llm(tree, node, plan.sub_query)
                    for rank, child in enumerate(ranked):
                        candidates.append((len(ranked) - rank, child))
                else:
                    for child in children:
                        emb = self._node_index.get_embedding(child.node_id)
                        score = _dot(sub_query_emb, emb) if emb else 0.0
                        candidates.append((score, child))

            if not candidates:
                break

            # Global top-k
            candidates.sort(key=lambda x: x[0], reverse=True)
            top = candidates[:remaining]

            next_frontier: list[MemTreeNode] = []
            for _, node in top:
                if node.level == 0:
                    settled_l0.append(node)
                else:
                    next_frontier.append(node)

            frontier = next_frontier

        result: list[str] = []
        for node in settled_l0:
            result.extend(node.child_ids)
        return result

    def browse_all(
        self,
        plans: "list[BrowsePlan]",
        *,
        query_emb: list[float],
        beam_width: int | None = None,
        max_workers: int | None = None,
    ) -> list[str]:
        """Browse multiple plans in parallel. Returns deduplicated fact/cell ids."""
        if not plans:
            return []

        workers = max_workers if max_workers is not None else self._config.max_workers
        sub_query_emb_cache: dict[str, list[float]] = {}
        seen: set[str] = set()
        result: list[str] = []

        def _browse_one(plan: "BrowsePlan") -> list[str]:
            return self.browse(
                plan,
                query_emb=query_emb,
                beam_width=beam_width,
                sub_query_emb_cache=sub_query_emb_cache,
            )

        with ThreadPoolExecutor(max_workers=min(workers, len(plans))) as pool:
            futures = {pool.submit(_browse_one, p): p for p in plans}
            for fut in as_completed(futures):
                try:
                    ids = fut.result()
                    for fid in ids:
                        if fid not in seen:
                            seen.add(fid)
                            result.append(fid)
                except Exception:
                    pass

        return result

    # ── tree child accessor ──────────────────────────────────────────────────

    def _get_node_children(
        self, tree: MemTree, node: MemTreeNode
    ) -> list[MemTreeNode]:
        """Return all tree-node children (both internal and L0)."""
        result: list[MemTreeNode] = []
        for cid in node.child_ids:
            child = tree.nodes.get(cid)
            if child is not None:
                result.append(child)
        return result

    # ── ranking ───────────────────────────────────────────────────────────────

    def _rank_children_llm(
        self,
        tree: MemTree,
        node: MemTreeNode,
        question: str,
    ) -> list[MemTreeNode]:
        """LLM-guided child ranking. 1 LLM call per node.

        Only active when config.llm_guided=True and browse_type is anchor/preference.
        Falls back to embedding ranking if LLM call fails or returns malformed output.
        """
        children: list[MemTreeNode] = []
        for child_id in node.child_ids:
            child_node = tree.nodes.get(child_id)
            if child_node is not None:
                children.append(child_node)

        if not children or self._chat_client is None:
            return children

        child_lines = []
        for i, child in enumerate(children):
            preview = child.summary or ""
            child_lines.append(f"[{i}] node_id={child.node_id}: {preview}")
        child_block = "\n".join(child_lines)

        system_prompt = (
            "You are a memory tree navigator. Rank the child memory nodes by relevance "
            "to the question. Return JSON only: {\"ranked_indices\": [int, ...]}"
        )
        user_prompt = f"Question: {question}\n\nChild nodes:\n{child_block}"

        try:
            result = self._chat_client.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                step_label="browse_llm_rank",
            )
            ranked_indices = result.get("ranked_indices", [])
            valid = [i for i in ranked_indices if isinstance(i, int) and 0 <= i < len(children)]
            ranked_set = set(valid)
            valid += [i for i in range(len(children)) if i not in ranked_set]
            return [children[i] for i in valid]
        except Exception:
            return children

    # ── sub-query embedding ───────────────────────────────────────────────────

    def _get_sub_query_emb(
        self,
        plan: "BrowsePlan",
        query_emb: list[float],
        cache: dict[str, list[float]] | None,
    ) -> list[float]:
        """Return embedding for plan.sub_query, reusing query_emb if identical."""
        sub_query = plan.sub_query

        if self._emb_client is None:
            return query_emb

        if cache is not None and sub_query in cache:
            return cache[sub_query]

        try:
            raw = self._emb_client.embed_texts([sub_query])[0]
            emb = _normalize(raw)
        except Exception:
            emb = query_emb

        if cache is not None:
            cache[sub_query] = emb

        return emb


# ── helpers ───────────────────────────────────────────────────────────────────

def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-9:
        return list(vec)
    return [x / norm for x in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
