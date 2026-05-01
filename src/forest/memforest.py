"""MemForest: multi-user memory forest coordinator.

Contains:
  - DeletableFactManager: FactManager subclass with FAISS-safe fact deletion
  - CachingTreeBuilder: TreeBuilder subclass that seeds node summaries from
    a content-hash cache before calling the LLM (avoids re-summarizing
    unchanged nodes after deletion rebuilds)
  - ImportResult: result dataclass for legacy snapshot import
  - MemForest: public multi-user API
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np

from src.build.tree_builder import TreeBuilder
from src.build.tree_types import MemTree, MemTreeNode
from src.extraction.fact_manager import FactManager
from src.utils.types import ManagedFact

if TYPE_CHECKING:
    from src.build.node_index import NodeIndex
    from src.config.config import MemForestConfig
    from src.forest.forest_merge import ForestMergeResult
    from src.query.pipeline import QueryResult


# ── DeletableFactManager ─────────────────────────────────────────────────────

class DeletableFactManager(FactManager):
    """FactManager extended with surgical fact deletion and bulk import.

    Two additions over the base class:
      delete_facts(fact_ids_to_remove)  — FAISS-safe in-place deletion
      load_managed_facts(facts)         — bulk import bypassing deduplication
    """

    def delete_facts(self, fact_ids_to_remove: set[str]) -> int:
        """Remove facts and rebuild the FAISS index from surviving vectors.

        Uses ``faiss.IndexFlatIP.reconstruct(i, buf)`` to retrieve each
        surviving vector without API calls. Only valid for IndexFlatIP (which
        FactManager always uses).

        Returns:
            Count of facts actually removed.
        """
        with self._lock:
            if not fact_ids_to_remove:
                return 0

            # Collect surviving (fact_id, float32 vector) pairs
            surviving_ids: list[str] = []
            surviving_vecs: list[np.ndarray] = []
            buf = np.zeros(self.vector_dim, dtype="float32")
            for i, fid in enumerate(self._faiss_ids):
                if fid in fact_ids_to_remove:
                    continue
                self._index.reconstruct(i, buf)
                surviving_ids.append(fid)
                surviving_vecs.append(buf.copy())

            removed = len(self._faiss_ids) - len(surviving_ids)
            if removed == 0:
                return 0

            # Rebuild FAISS index from surviving vectors
            new_index = faiss.IndexFlatIP(self.vector_dim)
            if surviving_vecs:
                new_index.add(np.vstack(surviving_vecs))
            self._index = new_index
            self._faiss_ids = surviving_ids

            # Update _facts and _exact_index
            for fid in fact_ids_to_remove:
                fact = self._facts.pop(fid, None)
                if fact is not None:
                    key = self._normalize_fact_text(fact.fact_text)
                    # Only remove from exact_index if it still maps to this fact
                    if self._exact_index.get(key) == fid:
                        del self._exact_index[key]

            # Purge duplicate records that referenced removed facts
            self._duplicate_records = [
                rec for rec in self._duplicate_records
                if rec.canonical_fact_id not in fact_ids_to_remove
            ]

            return removed

    def load_managed_facts(
        self,
        facts: list[ManagedFact],
        *,
        embed_batch_size: int = 128,
    ) -> None:
        """Bulk-import pre-existing ManagedFact objects into FAISS.

        Embeds fact texts in batches and inserts directly into internal
        structures, bypassing the deduplication pipeline. Used for legacy
        snapshot conversion.

        Args:
            facts: Pre-loaded ManagedFact objects (e.g. from canonical_facts.jsonl).
            embed_batch_size: Maximum texts to embed per API call.
        """
        with self._lock:
            for i in range(0, len(facts), embed_batch_size):
                batch = facts[i : i + embed_batch_size]
                vectors = self._embed_texts([f.fact_text for f in batch])
                for fact, vec in zip(batch, vectors):
                    if fact.fact_id in self._facts:
                        continue  # already present — skip
                    self._facts[fact.fact_id] = fact
                    self._faiss_ids.append(fact.fact_id)
                    key = self._normalize_fact_text(fact.fact_text)
                    self._exact_index[key] = fact.fact_id
                    self._index.add(vec.reshape(1, -1))


# ── CachingTreeBuilder ────────────────────────────────────────────────────────

def _collect_descendant_fact_ids(tree: MemTree, node: MemTreeNode) -> frozenset[str]:
    """BFS to collect all fact_ids / cell_ids under a node with single-payload L0 leaves."""
    result: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current.level == 0:
            result.update(current.child_ids)
        else:
            for cid in current.child_ids:
                child = tree.nodes.get(cid)
                if child is not None:
                    stack.append(child)
    return frozenset(result)


def _content_key(fact_ids: frozenset[str]) -> str:
    """Stable MD5 hex key from a frozenset of fact/cell IDs."""
    raw = "|".join(sorted(fact_ids))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


class CachingTreeBuilder(TreeBuilder):
    """TreeBuilder that seeds node summaries from a content-hash cache.

    Before calling the LLM, ``_run_summaries_bottom_up`` checks each dirty
    internal node's descendant fact set against the cache. Nodes whose
    content is unchanged get their old summary restored, avoiding redundant
    LLM calls during deletion rebuilds.
    """

    def __init__(self, *args, summary_cache: dict[str, str] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._summary_cache: dict[str, str] = dict(summary_cache or {})

    def set_summary_cache(self, cache: dict[str, str]) -> None:
        self._summary_cache = dict(cache)

    def _run_summaries_bottom_up(self, trees, all_requests_by_level, fact_map) -> None:
        # Seed summaries from cache before any LLM calls
        if self._summary_cache:
            for tree in trees:
                for node in tree.nodes.values():
                    if node.level == 0 or not node.summary_dirty:
                        continue
                    descendants = _collect_descendant_fact_ids(tree, node)
                    if not descendants:
                        continue
                    key = _content_key(descendants)
                    cached = self._summary_cache.get(key)
                    if cached:
                        node.summary = cached
                        node.summary_dirty = False
        # Delegate to parent for any remaining dirty nodes
        super()._run_summaries_bottom_up(trees, all_requests_by_level, fact_map)


# ── ImportResult ──────────────────────────────────────────────────────────────

@dataclass
class ImportResult:
    """Result of MemForest.import_user_from_legacy."""
    user_id: str
    n_facts: int
    n_trees: int
    n_nodes_indexed: int


# ── MemForest ─────────────────────────────────────────────────────────────────

class MemForest:
    """Multi-user memory forest coordinator.

    Manages one UserForest per user_id, providing per-user isolation and
    cross-user parallelism via ThreadPoolExecutor.

    Usage::

        forest = MemForest("data/memforest", config=config)
        forest.register_user("alice")
        forest.ingest_session("alice", "sess_001", turns)
        result = forest.query("alice", "What did Alice say about travel?")
        forest.save("alice")

        # Parallel queries across users
        results = forest.query_parallel([
            {"user_id": "alice", "question": "..."},
            {"user_id": "bob",   "question": "..."},
        ])
    """

    def __init__(
        self,
        snapshot_dir: str | Path,
        config: "MemForestConfig | None" = None,
        *,
        max_workers: int = 8,
    ) -> None:
        from src.api.client import OpenAIChatClient, OpenAIEmbeddingClient
        from src.config.config import load_default_config

        self._snapshot_dir = Path(snapshot_dir)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._config = config or load_default_config()
        self._max_workers = max_workers

        # Shared API clients — OpenAI SDK connection pools are thread-safe
        self._chat_client = OpenAIChatClient(self._config.api.llm)
        self._embedding_client = OpenAIEmbeddingClient(self._config.api.embedding)

        self._users: dict[str, "UserForest"] = {}
        self._users_lock = threading.RLock()

    # ── user lifecycle ────────────────────────────────────────────────────────

    def register_user(self, user_id: str, *, load_existing: bool = True) -> None:
        """Create a UserForest for user_id.

        If load_existing=True and snapshot data exists on disk, state is
        automatically loaded. Calling this on an already-registered user
        is a no-op.
        """
        with self._users_lock:
            if user_id in self._users:
                return
            from src.forest.user_forest import UserForest
            uf = UserForest(
                user_id=user_id,
                user_dir=self._snapshot_dir / user_id,
                config=self._config,
                chat_client=self._chat_client,
                embedding_client=self._embedding_client,
            )
            if load_existing:
                uf.load()
            self._users[user_id] = uf

    def list_users(self) -> list[str]:
        with self._users_lock:
            return list(self._users.keys())

    def has_user(self, user_id: str) -> bool:
        with self._users_lock:
            return user_id in self._users

    # ── single-user operations ────────────────────────────────────────────────

    def ingest_session(
        self,
        user_id: str,
        session_id: str,
        turns: list[dict],
    ) -> "IngestResult":
        """Extract facts from turns and ingest into user's forest.

        Args:
            user_id: Registered user identifier.
            session_id: Session identifier (arbitrary string; used as tree key).
            turns: List of turn dicts with keys: role/speaker_tag, content/text,
                   timestamp (optional), content_id (optional turn_id).

        Returns:
            IngestResult with counts of extracted cells, facts, and updated trees.
        """
        return self._get_user(user_id).ingest_session(session_id, turns)

    def delete_session(self, user_id: str, session_id: str) -> None:
        """Delete a session and auto-rebuild affected trees.

        Facts that have occurrences only in this session are removed.
        Facts shared with other sessions retain their cross-session occurrences.
        Tree summaries are preserved via content-hash cache where possible.

        Raises:
            KeyError: user_id not registered or session_id not found.
        """
        self._get_user(user_id).delete_session(session_id)

    def delete_turn(self, user_id: str, session_id: str, turn_id: str) -> None:
        """Delete a single turn and rebuild if it orphaned any facts.

        Raises:
            KeyError: user_id not registered or turn_id not found.
            ValueError: turn_id belongs to a different session.
        """
        self._get_user(user_id).delete_turn(session_id, turn_id)

    def query(
        self,
        user_id: str,
        question: str,
        *,
        query_time: float | None = None,
        top_k: int | None = None,
        max_facts: int | None = None,
    ) -> "QueryResult":
        """Run the recall→plan→browse pipeline for one user.

        Args:
            user_id: Registered user identifier.
            question: Natural-language question.
            query_time: Unix timestamp for temporal context injection (optional).
            top_k: Override config.recall.top_k (number of recalled trees).
            max_facts: Override config.browse.max_facts (facts returned).

        Returns:
            QueryResult with top_facts, recalled_trees, browse_plans, etc.
        """
        return self._get_user(user_id).query(
            question,
            query_time=query_time,
            top_k=top_k,
            max_facts=max_facts,
        )

    # ── parallel operations ───────────────────────────────────────────────────

    def query_parallel(
        self,
        requests: list[dict],
    ) -> "list[QueryResult | Exception]":
        """Execute queries for multiple users concurrently.

        Each request dict must have:
          - ``user_id`` (str)
          - ``question`` (str)
        Optional keys: ``query_time``, ``top_k``, ``max_facts``.

        Returns results in the same order as requests. Per-request exceptions
        are returned in-place rather than raised.

        Different users run truly concurrently (independent locks). Same-user
        requests are serialized by the per-user RLock.
        """
        if not requests:
            return []

        results: list["QueryResult | Exception"] = [None] * len(requests)  # type: ignore[list-item]

        def _run(idx: int, req: dict) -> None:
            try:
                uf = self._get_user(req["user_id"])
                results[idx] = uf.query(
                    req["question"],
                    query_time=req.get("query_time"),
                    top_k=req.get("top_k"),
                    max_facts=req.get("max_facts"),
                )
            except Exception as exc:
                results[idx] = exc

        with ThreadPoolExecutor(max_workers=min(self._max_workers, len(requests))) as pool:
            futures = [pool.submit(_run, i, req) for i, req in enumerate(requests)]
            for fut in futures:
                fut.result()  # propagate thread exceptions (already caught above)

        return results

    def ingest_parallel(
        self,
        requests: list[dict],
    ) -> "list[IngestResult | Exception]":
        """Ingest sessions for multiple users concurrently.

        Each request dict must have: ``user_id``, ``session_id``, ``turns``.

        Returns results in request order; exceptions returned in-place.
        """
        if not requests:
            return []

        from src.forest.user_forest import IngestResult
        results: list["IngestResult | Exception"] = [None] * len(requests)  # type: ignore[list-item]

        def _run(idx: int, req: dict) -> None:
            try:
                uf = self._get_user(req["user_id"])
                results[idx] = uf.ingest_session(req["session_id"], req["turns"])
            except Exception as exc:
                results[idx] = exc

        with ThreadPoolExecutor(max_workers=min(self._max_workers, len(requests))) as pool:
            futures = [pool.submit(_run, i, req) for i, req in enumerate(requests)]
            for fut in futures:
                fut.result()

        return results

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, user_id: str | None = None) -> None:
        """Save one user (or all users if user_id is None) to disk."""
        with self._users_lock:
            if user_id is not None:
                targets = [self._users[user_id]]
            else:
                targets = list(self._users.values())
        for uf in targets:
            uf.save()

    def load(self, user_id: str | None = None) -> None:
        """Load one user (or all registered users if user_id is None) from disk."""
        with self._users_lock:
            if user_id is not None:
                targets = [self._users[user_id]]
            else:
                targets = list(self._users.values())
        for uf in targets:
            uf.load()

    # ── legacy snapshot import ────────────────────────────────────────────────

    def import_user_from_legacy(
        self,
        user_id: str,
        *,
        tree_store_dir: str | Path,
        node_index_dir: str | Path,
        facts_jsonl: str | Path,
        overwrite: bool = False,
    ) -> ImportResult:
        """Import a per-question build snapshot as a new user forest.

        Copies tree_store + node_index, re-embeds facts into FactManager,
        reconstructs SessionRegistry from session tree leaves, and saves.

        Args:
            user_id: New user identifier for this snapshot.
            tree_store_dir: Path to legacy ``tree_store/`` directory.
            node_index_dir: Path to legacy ``node_index/`` directory.
            facts_jsonl: Path to ``canonical_facts.jsonl``.
            overwrite: If True, overwrite existing user snapshot.

        Returns:
            ImportResult with counts.

        Raises:
            ValueError: user_id already registered and overwrite=False.
        """
        tree_store_dir = Path(tree_store_dir)
        node_index_dir = Path(node_index_dir)
        facts_jsonl = Path(facts_jsonl)

        with self._users_lock:
            if user_id in self._users and not overwrite:
                raise ValueError(
                    f"User {user_id!r} already registered. "
                    "Pass overwrite=True to replace."
                )
            # Remove existing user forest if overwriting
            if user_id in self._users:
                del self._users[user_id]

        user_dir = self._snapshot_dir / user_id

        # --- Step 1: Copy tree_store files ---
        dest_trees = user_dir / "trees"
        if dest_trees.exists():
            shutil.rmtree(str(dest_trees))
        if tree_store_dir.exists():
            shutil.copytree(str(tree_store_dir), str(dest_trees))
        else:
            dest_trees.mkdir(parents=True, exist_ok=True)

        # --- Step 2: Copy node_index files ---
        dest_node_index = user_dir / "node_index"
        if dest_node_index.exists():
            shutil.rmtree(str(dest_node_index))
        if node_index_dir.exists():
            shutil.copytree(str(node_index_dir), str(dest_node_index))
        else:
            dest_node_index.mkdir(parents=True, exist_ok=True)

        # --- Step 3: Load facts from JSONL ---
        facts = _load_facts_jsonl(facts_jsonl)

        # --- Step 4: Create UserForest (no load — we populate manually) ---
        from src.forest.user_forest import UserForest
        uf = UserForest(
            user_id=user_id,
            user_dir=user_dir,
            config=self._config,
            chat_client=self._chat_client,
            embedding_client=self._embedding_client,
        )
        # Don't call uf.load() — we populate components below

        # --- Step 5: Load trees into TreeStore + TreeBuilder ---
        uf._tree_store.load_all()
        for tree_id in uf._tree_store.all_tree_ids():
            tree = uf._tree_store.get(tree_id)
            if tree is not None:
                uf._tree_builder._trees[tree_id] = tree

        # --- Step 6: Load NodeIndex ---
        uf._node_index.load(dest_node_index)

        # --- Step 7: Re-embed facts → populate DeletableFactManager ---
        if facts:
            uf._fact_manager.load_managed_facts(facts)
        n_facts = len(uf._fact_manager.iter_facts())

        # --- Step 8: Build summary_cache from loaded tree nodes ---
        uf._summary_cache = _extract_summary_cache_from_trees(
            uf._tree_builder.trees
        )

        # --- Step 9: Reconstruct SessionRegistry from session tree leaves ---
        _populate_registry_from_trees(uf._registry, uf._tree_store)

        # --- Step 10: Re-wire ForestQuery references ---
        uf._rewire_query_pipeline()

        # --- Step 11: Save complete snapshot ---
        uf.save()

        with self._users_lock:
            self._users[user_id] = uf

        n_trees = len(uf._tree_store.all_tree_ids())
        n_nodes = uf._node_index.size()
        return ImportResult(
            user_id=user_id,
            n_facts=n_facts,
            n_trees=n_trees,
            n_nodes_indexed=n_nodes,
        )

    # ── forest merge ──────────────────────────────────────────────────────────

    def merge_user_forests(
        self,
        source_user_ids: list[str],
        target_user_id: str,
        *,
        overwrite: bool = False,
    ) -> "ForestMergeResult":
        """Efficient migration: merge several registered user forests into one target.

        Runs :func:`src.forest.forest_merge.merge_user_forests`, which dedupes
        facts via ``FactManager.merge_from``, unions routers, then either copies
        trees directly (when their cluster/entity/session identity is unique in
        the source set) or picks the largest tree as a trunk and inserts leaves
        from the others, flushing only the trunks that absorbed new leaves.

        Args:
            source_user_ids: Already-registered source forests to merge.
            target_user_id: New user_id to receive the merged forest.
            overwrite: If True, wipe any existing snapshot for target_user_id.

        Returns:
            ``ForestMergeResult`` with fact/tree counts and elapsed wall time.

        Raises:
            KeyError: A source user_id is not registered.
            ValueError: ``target_user_id`` already registered and overwrite=False.
        """
        from src.forest.forest_merge import merge_user_forests as _merge_impl
        from src.forest.user_forest import UserForest

        with self._users_lock:
            if target_user_id in self._users and not overwrite:
                raise ValueError(
                    f"Target user {target_user_id!r} already registered. "
                    "Pass overwrite=True to replace."
                )
            sources: list[UserForest] = []
            for uid in source_user_ids:
                uf = self._users.get(uid)
                if uf is None:
                    raise KeyError(f"Source user {uid!r} not registered.")
                sources.append(uf)
            if target_user_id in self._users:
                del self._users[target_user_id]

        target_dir = self._snapshot_dir / target_user_id
        if overwrite and target_dir.exists():
            shutil.rmtree(str(target_dir))

        target_uf = UserForest(
            user_id=target_user_id,
            user_dir=target_dir,
            config=self._config,
            chat_client=self._chat_client,
            embedding_client=self._embedding_client,
        )

        result = _merge_impl(sources=sources, target=target_uf)

        with self._users_lock:
            self._users[target_user_id] = target_uf

        return result

    # ── internal ──────────────────────────────────────────────────────────────

    def _get_user(self, user_id: str) -> "UserForest":
        with self._users_lock:
            uf = self._users.get(user_id)
        if uf is None:
            raise KeyError(
                f"User {user_id!r} not registered. Call register_user() first."
            )
        return uf


# ── module-level helpers ──────────────────────────────────────────────────────

def _load_facts_jsonl(path: Path) -> list[ManagedFact]:
    """Load ManagedFact objects from a canonical_facts.jsonl file."""
    from src.utils.types import FactOccurrence
    if not path.exists():
        return []
    facts: list[ManagedFact] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                occurrences = [
                    FactOccurrence(**occ)
                    for occ in d.get("occurrences", [])
                ]
                d["occurrences"] = occurrences
                facts.append(ManagedFact(**d))
            except Exception:
                pass
    return facts


def _extract_summary_cache_from_trees(
    trees: dict[str, MemTree],
) -> dict[str, str]:
    """Snapshot node summaries from all trees into content_key→summary dict."""
    cache: dict[str, str] = {}
    for tree in trees.values():
        for node in tree.nodes.values():
            if node.level == 0 or not node.summary:
                continue
            descendants = _collect_descendant_fact_ids(tree, node)
            if not descendants:
                continue
            key = _content_key(descendants)
            cache[key] = node.summary
    return cache


def _populate_registry_from_trees(registry, tree_store) -> None:
    """Reconstruct SessionRegistry from session tree leaves (legacy import)."""
    for tree_id in tree_store.all_tree_ids():
        tree = tree_store.get(tree_id)
        if tree is None or tree.tree_type != "session":
            continue
        session_id = tree.tree_key
        cell_ids = list(tree.session_leaves.keys())
        cell_fact_ids: dict[str, list[str]] = {}
        for cell_id, leaf in tree.session_leaves.items():
            cell_fact_ids[cell_id] = list(getattr(leaf, "fact_ids", []))
        registry.register_synthetic_session(
            session_id=session_id,
            cell_ids=cell_ids,
            cell_fact_ids=cell_fact_ids,
        )
