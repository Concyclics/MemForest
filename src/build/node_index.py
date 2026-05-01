"""NodeIndex: FAISS index over root node embeddings + full emb_store for browse.

Dual-purpose:
  1. Recall: search(query_emb, top_n) → list[NodeEntry] for ForestRetriever
     Root-only FAISS index — one entry per tree for cleaner ranking.
  2. Browse: get_embedding(node_id) → list[float] for TreeBrowser
     All non-L0 node embeddings stored for O(1) lookup at navigate time.

Files on disk (under index_dir/):
    faiss.index         ← IndexFlatIP, normalized vectors (root nodes only)
    node_entries.json   ← list[{node_id, tree_id, level}], parallel to FAISS rows
    emb_store.json      ← {node_id: embedding_list} for all non-L0 nodes
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class NodeEntry:
    """Metadata for one FAISS row."""

    node_id: str
    tree_id: str
    level: str  # "root" | "L1" | "L2"


class NodeIndex:
    """Thread-safe FAISS index (root-only for search) + full emb_store for browse."""

    def __init__(
        self,
        *,
        index_dir: Path,
        vector_dim: int = 1024,
    ) -> None:
        self._dir = Path(index_dir)
        self._dim = vector_dim
        self._lock = threading.RLock()

        # FAISS search index: root nodes only (parallel lists)
        self._entries: list[NodeEntry] = []
        self._embeddings: list[list[float]] = []

        # All non-L0 embeddings for browse (root + L1 + L2 + ...)
        self._emb_store: dict[str, list[float]] = {}

        # Cached FAISS index (rebuilt lazily, invalidated on add/remove)
        self._faiss_index = None

    # ── write API ─────────────────────────────────────────────────────────────

    def add_node(
        self,
        entry: NodeEntry,
        embedding: list[float],
        *,
        searchable: bool = True,
    ) -> None:
        """Add a node's embedding.

        Args:
            entry: Node metadata.
            embedding: Raw embedding vector (will be normalized).
            searchable: If True, add to FAISS search index. If False, only
                store in emb_store for browse lookup.
        """
        norm_emb = _normalize(embedding)
        with self._lock:
            # Always store in emb_store for browse
            self._emb_store[entry.node_id] = norm_emb

            if not searchable:
                return

            # Replace if already present in search index
            for i, e in enumerate(self._entries):
                if e.node_id == entry.node_id:
                    self._entries[i] = entry
                    self._embeddings[i] = norm_emb
                    self._faiss_index = None
                    return
            self._entries.append(entry)
            self._embeddings.append(norm_emb)
            self._faiss_index = None

    def remove_tree(self, tree_id: str) -> None:
        """Remove all entries belonging to tree_id."""
        with self._lock:
            # Remove from search index
            keep = [
                (e, emb)
                for e, emb in zip(self._entries, self._embeddings)
                if e.tree_id != tree_id
            ]
            self._entries = [e for e, _ in keep]
            self._embeddings = [emb for _, emb in keep]
            # Remove from emb_store (browse)
            prefix = tree_id + ":"
            self._emb_store = {
                k: v for k, v in self._emb_store.items()
                if not k.startswith(prefix) and k != tree_id
            }
            self._faiss_index = None

    # ── read API ──────────────────────────────────────────────────────────────

    def search(self, query_emb: list[float], top_n: int = 200) -> list[NodeEntry]:
        """Return up to top_n NodeEntries ranked by cosine similarity (highest first)."""
        with self._lock:
            n = len(self._entries)
            if n == 0:
                return []
            actual_k = min(top_n, n)
            q = _normalize(query_emb)
            return self._search_locked(q, actual_k)

    def get_embedding(self, node_id: str) -> list[float] | None:
        """Return the stored (normalized) embedding for a node, or None."""
        return self._emb_store.get(node_id)

    def all_entries(self) -> list[NodeEntry]:
        with self._lock:
            return list(self._entries)

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, index_dir: Path | None = None) -> None:
        """Write faiss.index, node_entries.json, emb_store.json to index_dir."""
        d = Path(index_dir) if index_dir else self._dir
        d.mkdir(parents=True, exist_ok=True)
        with self._lock:
            entries_data = [asdict(e) for e in self._entries]
            (d / "node_entries.json").write_text(
                json.dumps(entries_data, ensure_ascii=False), encoding="utf-8"
            )
            (d / "emb_store.json").write_text(
                json.dumps(self._emb_store, ensure_ascii=False), encoding="utf-8"
            )
            # Save FAISS binary if available, else skip
            try:
                import faiss
                import numpy as np

                if len(self._embeddings) > 0:
                    arr = np.array(self._embeddings, dtype=np.float32)
                    idx = faiss.IndexFlatIP(self._dim)
                    idx.add(arr)
                    faiss.write_index(idx, str(d / "faiss.index"))
            except Exception:
                pass  # Will use pure-Python fallback on load

    def load(self, index_dir: Path | None = None) -> None:
        """Restore from disk."""
        d = Path(index_dir) if index_dir else self._dir
        entries_path = d / "node_entries.json"
        emb_path = d / "emb_store.json"
        if not entries_path.exists():
            return
        with self._lock:
            raw = json.loads(entries_path.read_text(encoding="utf-8"))
            self._entries = [NodeEntry(**e) for e in raw]
            store = json.loads(emb_path.read_text(encoding="utf-8")) if emb_path.exists() else {}
            self._emb_store = {k: list(v) for k, v in store.items()}
            # Rebuild embeddings list in parallel-array order
            self._embeddings = [
                self._emb_store.get(e.node_id, []) for e in self._entries
            ]
            self._faiss_index = None

    # ── internal ──────────────────────────────────────────────────────────────

    def _search_locked(self, q: list[float], top_k: int) -> list[NodeEntry]:
        """Brute-force inner-product search; uses FAISS if available."""
        try:
            import faiss
            import numpy as np

            if self._faiss_index is None:
                arr = np.array(self._embeddings, dtype=np.float32)
                idx = faiss.IndexFlatIP(self._dim)
                idx.add(arr)
                self._faiss_index = idx
            q_arr = np.array([q], dtype=np.float32)
            sims, idxs = self._faiss_index.search(q_arr, top_k)
            return [self._entries[i] for i in idxs[0] if 0 <= i < len(self._entries)]
        except ImportError:
            pass

        # Pure-Python fallback
        scored = [(_dot(q, emb), i) for i, emb in enumerate(self._embeddings) if emb]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._entries[i] for _, i in scored[:top_k]]


# ── helpers ───────────────────────────────────────────────────────────────────

def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-9:
        return list(vec)
    return [x / norm for x in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
