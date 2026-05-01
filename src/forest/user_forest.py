"""UserForest: per-user memory forest with full lifecycle management.

Owns all per-user components:
  - DeletableFactManager   (fact store + FAISS dedup)
  - CachingTreeBuilder     (incremental ingest + rebuild with summary cache)
  - TreeStore              (tree JSON persistence)
  - NodeIndex              (M4 augmented FAISS for recall + browse)
  - ForestQuery            (recall→plan→browse pipeline)
  - ChunkExtractionPipeline (turns → MemCells → MemoryItems)
  - SessionRegistry        (session/turn/fact bookkeeping + deletion)

Thread safety: all public methods acquire a single per-user RLock.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.build.node_index import NodeIndex
from src.build.tree import delete_cell, delete_fact
from src.build.tree_store import TreeStore
from src.extraction.pipeline import ChunkExtractionPipeline
from src.forest.memforest import (
    CachingTreeBuilder,
    DeletableFactManager,
    _collect_descendant_fact_ids,
    _content_key,
    _extract_summary_cache_from_trees,
)
from src.forest.session_registry import SessionRegistry
from src.query.pipeline import FactManagerLoader, ForestQuery
from src.utils.types import MemCell, NormalizedTurn

if TYPE_CHECKING:
    from src.api.client import OpenAIChatClient, OpenAIEmbeddingClient
    from src.build.tree_types import MemTree
    from src.config.config import MemForestConfig
    from src.query.pipeline import QueryResult
    from src.utils.types import ManagedFact


# ── result types ──────────────────────────────────────────────────────────────

@dataclass
class IngestResult:
    """Result of UserForest.ingest_session."""
    session_id: str
    cells_extracted: int
    facts_inserted: int
    facts_merged: int
    trees_updated: list[str]


# ── serialization helpers ─────────────────────────────────────────────────────

def _memcell_to_dict(cell: MemCell) -> dict:
    return asdict(cell)


def _memcell_from_dict(d: dict) -> MemCell:
    d = dict(d)
    d["turns"] = [NormalizedTurn(**t) for t in d.get("turns", [])]
    return MemCell(**d)


# ── UserForest ────────────────────────────────────────────────────────────────

class UserForest:
    """Encapsulates all memory components for a single user.

    Ingest, deletion, query, and persistence operations are all
    guarded by a single threading.RLock so concurrent callers
    (e.g. from MemForest.query_parallel) do not interleave.
    """

    def __init__(
        self,
        user_id: str,
        user_dir: Path,
        config: "MemForestConfig",
        *,
        chat_client: "OpenAIChatClient",
        embedding_client: "OpenAIEmbeddingClient",
    ) -> None:
        from src.extraction.dedup import LLMFactEquivalenceJudge

        self.user_id = user_id
        self._dir = Path(user_dir)
        self._config = config
        self._chat_client = chat_client
        self._embedding_client = embedding_client
        self._lock = threading.RLock()

        fm_cfg = config.extraction.fact_manager
        ex_cfg = config.extraction

        self._fact_manager = DeletableFactManager(
            storage_dir=self._dir / "facts",
            embedding_client=embedding_client,
            judge=LLMFactEquivalenceJudge(
                chat_client=chat_client,
                model_name=config.api.llm.model_name,
                temperature=0.0,
                max_tokens=200,
                top_p=0.2,
            ),
            embedding_model_name=config.api.embedding.model_name,
            vector_dim=config.api.embedding.dimension,
            persist_on_write=False,
            top_k=fm_cfg.top_k,
            similarity_threshold=fm_cfg.similarity_threshold,
            max_llm_pairs_per_item=fm_cfg.max_llm_pairs_per_item,
            normalize_embeddings=fm_cfg.normalize_embeddings,
        )
        self._tree_builder = CachingTreeBuilder(
            config=config,
            chat_client=chat_client,
            embedding_client=embedding_client,
        )
        self._tree_store = TreeStore(storage_dir=self._dir / "trees")
        self._node_index = NodeIndex(
            index_dir=self._dir / "node_index",
            vector_dim=config.api.embedding.dimension,
        )
        self._registry = SessionRegistry()
        self._query_pipeline = ForestQuery(
            embedding_client=embedding_client,
            chat_client=chat_client,
            tree_store=self._tree_store,
            node_index=self._node_index,
            fact_loader=FactManagerLoader(self._fact_manager, self._registry),
            fact_manager=self._fact_manager,
            config=config.query,
        )
        self._extraction_pipeline = ChunkExtractionPipeline(
            backend=chat_client,
            chunking=ex_cfg.chunking,
            max_items_per_cell=ex_cfg.max_items_per_cell,
            max_assistant_items_per_cell=ex_cfg.max_assistant_items_per_cell,
            max_topics_per_item=ex_cfg.max_topics_per_item,
            max_attribute_keys_per_item=ex_cfg.max_attribute_keys_per_item,
            max_domain_keys_per_item=ex_cfg.max_domain_keys_per_item,
            max_collection_keys_per_item=ex_cfg.max_collection_keys_per_item,
        )
        self._cell_store: dict[str, MemCell] = {}
        self._summary_cache: dict[str, str] = {}

    # ── ingest ────────────────────────────────────────────────────────────────

    def ingest_session(
        self,
        session_id: str,
        turns: list[dict],
    ) -> IngestResult:
        """Extract facts from turns and incrementally ingest into this user's forest.

        Args:
            session_id: Session identifier (used as tree key for session trees).
            turns: List of turn dicts (role/content/timestamp/content_id).

        Returns:
            IngestResult with extraction and tree-update counts.
        """
        with self._lock:
            # 1. Extract cells + memory items
            extraction_result = self._extraction_pipeline.extract_session(
                session_id, turns
            )

            # 2. Ingest into FactManager
            fact_write_result = self._fact_manager.add_session_result(
                extraction_result
            )

            # 3. Build cell_to_facts from FactOccurrence data
            cell_to_facts: dict[str, list[str]] = {}
            for fact_id in fact_write_result.canonical_fact_ids:
                fact = self._fact_manager.get_fact(fact_id)
                if fact is None:
                    continue
                for occ in fact.occurrences:
                    if occ.session_id == session_id:
                        lst = cell_to_facts.setdefault(occ.cell_id, [])
                        if fact_id not in lst:
                            lst.append(fact_id)

            # 4. Incremental tree ingest
            touched_facts = [
                f for fid in fact_write_result.canonical_fact_ids
                if (f := self._fact_manager.get_fact(fid)) is not None
            ]
            build_results = self._tree_builder.ingest_session(
                new_facts=touched_facts,
                cells=extraction_result.cells,
                cell_to_facts=cell_to_facts,
            )

            # 5. Persist updated trees
            for br in build_results:
                tree = self._tree_builder.trees.get(br.tree_id)
                if tree is not None:
                    self._tree_store.save_tree(tree)
                    self._tree_store.register(tree)

            # 6. Rebuild NodeIndex
            self._node_index = self._tree_builder.build_node_index(
                list(self._tree_builder.trees.values()),
                index_dir=self._dir / "node_index",
                vector_dim=self._config.api.embedding.dimension,
            )
            self._rewire_query_pipeline()

            # 7. Register with SessionRegistry
            self._registry.register_session(
                session_id=session_id,
                cells=extraction_result.cells,
                cell_to_facts=cell_to_facts,
            )

            # 8. Store cells for future rebuild
            for cell in extraction_result.cells:
                self._cell_store[cell.cell_id] = cell

            return IngestResult(
                session_id=session_id,
                cells_extracted=len(extraction_result.cells),
                facts_inserted=fact_write_result.inserted_count,
                facts_merged=fact_write_result.merged_count,
                trees_updated=[br.tree_id for br in build_results],
            )

    # ── deletion ──────────────────────────────────────────────────────────────

    def delete_session(self, session_id: str) -> None:
        """Mark session deleted and apply local tree deletions with lazy summary flush."""
        with self._lock:
            result = self._registry.delete_session(session_id)
            orphaned = self._registry.compute_orphaned_facts(
                {f.fact_id for f in self._fact_manager.iter_facts()}
            )
            self._apply_local_deletions(
                orphaned_fact_ids=orphaned,
                affected_cell_ids=set(result.affected_cell_ids),
            )

    def delete_turn(self, session_id: str, turn_id: str) -> None:
        """Mark turn deleted and apply local tree deletions with lazy summary flush."""
        with self._lock:
            result = self._registry.delete_turn(session_id, turn_id)
            orphaned = self._registry.compute_orphaned_facts(
                {f.fact_id for f in self._fact_manager.iter_facts()}
            )
            if orphaned or result.affected_cell_ids:
                self._apply_local_deletions(
                    orphaned_fact_ids=orphaned,
                    affected_cell_ids=set(result.affected_cell_ids),
                )

    # ── query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        *,
        query_time: float | None = None,
        top_k: int | None = None,
        max_facts: int | None = None,
    ) -> "QueryResult":
        """Run the full recall→plan→browse pipeline for this user.

        Args:
            question: Natural-language question.
            query_time: Unix timestamp for temporal context (optional).
            top_k: Override recall top_k.
            max_facts: Override browse max_facts.
        """
        with self._lock:
            self._rewire_query_pipeline()
            session_alias_map = self._tree_builder.session_alias_map
            return self._query_pipeline.query(
                question,
                query_time=query_time,
                session_alias_map=session_alias_map or None,
                top_k=top_k,
                max_facts=max_facts,
            )

    def build_context(self, result: "QueryResult") -> str:
        """Render a QueryResult into a text context string for LLM answering."""
        return self._query_pipeline.build_context(result)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist all user state to ``user_dir``."""
        with self._lock:
            self._dir.mkdir(parents=True, exist_ok=True)

            # 1. FactManager
            self._fact_manager.save()

            # 2. Trees (already saved incrementally; call save to ensure consistency)
            for tree in self._tree_builder.trees.values():
                self._tree_store.save_tree(tree)

            # 3. NodeIndex
            self._node_index.save(self._dir / "node_index")

            # 4. SessionRegistry
            self._registry.save(self._dir / "session_registry.json")

            # 5. Summary cache
            (self._dir / "summary_cache.json").write_text(
                json.dumps(self._summary_cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # 6. Cell store
            cell_data = {
                cell_id: _memcell_to_dict(cell)
                for cell_id, cell in self._cell_store.items()
            }
            (self._dir / "cell_store.json").write_text(
                json.dumps(cell_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # 7. Metadata (session_alias_map for TreeBuilder restore)
            metadata = {
                "user_id": self.user_id,
                "saved_at": time.time(),
                "session_alias_map": self._tree_builder.session_alias_map,
            }
            (self._dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def load(self) -> None:
        """Restore all user state from ``user_dir``. No-op if dir absent."""
        with self._lock:
            if not self._dir.exists():
                return

            # 1. FactManager
            self._fact_manager.load()

            # 2. Trees
            self._tree_store.load_all()
            for tree_id in self._tree_store.all_tree_ids():
                tree = self._tree_store.get(tree_id)
                if tree is not None:
                    self._tree_builder._trees[tree_id] = tree

            # 3. NodeIndex
            self._node_index.load(self._dir / "node_index")

            # 4. SessionRegistry
            self._registry.load(self._dir / "session_registry.json")

            # 5. Summary cache
            cache_path = self._dir / "summary_cache.json"
            if cache_path.exists():
                self._summary_cache = json.loads(
                    cache_path.read_text(encoding="utf-8")
                )

            # 6. Cell store
            cell_path = self._dir / "cell_store.json"
            if cell_path.exists():
                raw = json.loads(cell_path.read_text(encoding="utf-8"))
                self._cell_store = {
                    cell_id: _memcell_from_dict(d) for cell_id, d in raw.items()
                }

            # 7. Restore session_alias_map
            meta_path = self._dir / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                alias_map = meta.get("session_alias_map", {})
                with self._tree_builder._lock:
                    self._tree_builder._session_alias_map = alias_map

            self._rewire_query_pipeline()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _rewire_query_pipeline(self) -> None:
        """Ensure ForestQuery and its sub-components reference the current NodeIndex."""
        self._query_pipeline._node_index = self._node_index
        self._query_pipeline._retriever._node_index = self._node_index
        self._query_pipeline._browser._node_index = self._node_index
        # Also re-wire tree_store reference in case it was replaced during rebuild
        self._query_pipeline._tree_store = self._tree_store
        self._query_pipeline._retriever._tree_store = self._tree_store
        self._query_pipeline._browser._tree_store = self._tree_store
        # BU recall caches fact_id → tree_ids; invalidate after any mutation.
        self._query_pipeline._retriever.invalidate_fact_index()

    def _apply_local_deletions(
        self,
        *,
        orphaned_fact_ids: set[str],
        affected_cell_ids: set[str],
    ) -> None:
        touched_tree_ids: set[str] = set()
        current_cache = _extract_summary_cache_from_trees(self._tree_builder.trees)
        self._summary_cache = {**self._summary_cache, **current_cache}
        self._tree_builder.set_summary_cache(self._summary_cache)

        for cell_id in affected_cell_ids:
            cell_rec = self._registry._cells.get(cell_id)
            if cell_rec is None:
                continue
            public_session_id = self._tree_builder.session_alias_map.get(
                str(cell_rec.session_id),
                str(cell_rec.session_id),
            )
            tree_id = f"session:{public_session_id}"
            tree = self._tree_builder._trees.get(tree_id)
            if tree is not None and delete_cell(tree, cell_id):
                touched_tree_ids.add(tree_id)
            self._cell_store.pop(cell_id, None)

        if orphaned_fact_ids:
            self._fact_manager.delete_facts(orphaned_fact_ids)

        self._strip_deleted_occurrences(orphaned_fact_ids)

        if orphaned_fact_ids:
            for fact_id in orphaned_fact_ids:
                self._tree_builder._fact_catalog.pop(fact_id, None)

            self._tree_builder.entity_router.remove_fact_ids(
                orphaned_fact_ids,
                self._tree_builder._fact_catalog,
            )
            self._tree_builder.scene_router.remove_fact_ids(orphaned_fact_ids)

            for tree_id, tree in list(self._tree_builder._trees.items()):
                if tree.tree_type == "session":
                    continue
                deleted_any = False
                for fact_id in orphaned_fact_ids:
                    deleted_any = delete_fact(tree, fact_id) or deleted_any
                if deleted_any:
                    touched_tree_ids.add(tree_id)

        build_results = self._tree_builder.flush_dirty_trees(tree_ids=touched_tree_ids)
        flushed_tree_ids = {result.tree_id for result in build_results}
        for tree_id in touched_tree_ids:
            tree = self._tree_builder._trees.get(tree_id)
            if tree is None:
                continue
            self._tree_store.save_tree(tree)
            self._tree_store.register(tree)
        self._refresh_node_index_for_trees(touched_tree_ids)

        refreshed_cache = _extract_summary_cache_from_trees(
            {
                tree_id: self._tree_builder._trees[tree_id]
                for tree_id in flushed_tree_ids
                if tree_id in self._tree_builder._trees
            }
        )
        if refreshed_cache:
            self._summary_cache.update(refreshed_cache)
            self._tree_builder.set_summary_cache(self._summary_cache)
        self._rewire_query_pipeline()

    def _refresh_node_index_for_trees(self, tree_ids: set[str]) -> None:
        from src.build.node_index import NodeEntry

        if not tree_ids:
            return
        for tree_id in tree_ids:
            self._node_index.remove_tree(tree_id)

        to_embed: list[tuple[str, str, str, str, bool]] = []  # (tree_id, node_id, level_str, summary, searchable)
        for tree_id in tree_ids:
            tree = self._tree_builder._trees.get(tree_id)
            if tree is None:
                continue
            root_id = tree.root_node_id
            root_node = tree.nodes.get(root_id)
            is_depth0_scene = (
                tree.tree_type == "scene"
                and root_node is not None
                and root_node.level == 0
            )
            for node_id, node in tree.nodes.items():
                if not node.summary or node.level == 0:
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
                to_embed.append((tree.tree_id, node_id, level_str, node.summary, searchable))

        if to_embed:
            embeddings = self._embedding_client.embed_texts(
                [summary for _, _, _, summary, _ in to_embed]
            )
            for (tree_id, node_id, level_str, _summary, searchable), embedding in zip(to_embed, embeddings):
                self._node_index.add_node(
                    NodeEntry(node_id=node_id, tree_id=tree_id, level=level_str),
                    embedding,
                    searchable=searchable,
                )
        self._node_index.save(self._dir / "node_index")

    def _strip_deleted_occurrences(self, orphaned_fact_ids: set[str]) -> None:
        """Remove FactOccurrences from deleted sessions/turns in surviving facts."""
        from dataclasses import replace as dc_replace

        with self._fact_manager._lock:
            for fact_id, fact in list(self._fact_manager._facts.items()):
                if fact_id in orphaned_fact_ids:
                    continue
                new_occs = []
                for occ in fact.occurrences:
                    session_rec = self._registry._sessions.get(occ.session_id)
                    if session_rec and session_rec.deleted:
                        continue
                    cell_rec = self._registry._cells.get(occ.cell_id)
                    if cell_rec and cell_rec.turn_ids:
                        all_turns_deleted = all(
                            self._registry._turns.get(
                                tid, _DeadTurnSentinel
                            ).deleted
                            for tid in cell_rec.turn_ids
                        )
                        if all_turns_deleted:
                            continue
                    new_occs.append(occ)
                if len(new_occs) != len(fact.occurrences):
                    self._fact_manager._facts[fact_id] = dc_replace(
                        fact, occurrences=new_occs
                    )


# Sentinel for missing turns — treated as deleted when checking orphan status
class _DeadTurnSentinel:
    deleted = True
