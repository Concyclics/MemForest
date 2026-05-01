"""RootIndex: FAISS index over root summary embeddings for coarse recall.

Each tree's root summary is embedded and stored here. At query time,
the user's question is embedded and compared against all root summaries
to select the most relevant trees to browse.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from src.build.tree_types import TreeCard

if TYPE_CHECKING:
    from src.api.client import OpenAIEmbeddingClient


class RootIndex:
    """FAISS flat inner-product index over tree root summary embeddings.

    Thread-safe.
    """

    def __init__(
        self,
        *,
        embedding_client: "OpenAIEmbeddingClient",
        index_dir: Path,
        vector_dim: int,
        normalize_embeddings: bool = True,
    ) -> None:
        self._client = embedding_client
        self._dir = Path(index_dir)
        self._dim = vector_dim
        self._normalize = normalize_embeddings
        self._lock = threading.RLock()

        # Parallel arrays: row i in FAISS corresponds to _tree_ids[i]
        self._tree_ids: list[str] = []
        self._embeddings: list[list[float]] = []
        self._cards: dict[str, TreeCard] = {}

        # Lazy FAISS import so module loads without faiss installed
        self._faiss = None
        self._index = None

    # ── public API ────────────────────────────────────────────────────────────

    def add_tree(self, tree_id: str, card: TreeCard, embedding: list[float]) -> None:
        """Insert or replace a tree's root summary embedding."""
        with self._lock:
            if tree_id in self._cards:
                self._remove_tree_locked(tree_id)
            vec = _maybe_normalize(embedding) if self._normalize else list(embedding)
            self._tree_ids.append(tree_id)
            self._embeddings.append(vec)
            self._cards[tree_id] = card
            self._index = None  # invalidate cached FAISS index

    def remove_tree(self, tree_id: str) -> None:
        """Remove a tree from the index (used during split/deactivation)."""
        with self._lock:
            self._remove_tree_locked(tree_id)
            self._index = None

    def recall(
        self,
        query: str,
        *,
        top_k: int = 10,
        tree_type_filter: str | None = None,
    ) -> list[TreeCard]:
        """Embed the query and return the top-k most relevant TreeCards."""
        if not self._cards:
            return []

        try:
            embedding = self._client.embed_texts([query])[0]
        except Exception:
            return []

        query_vec = _maybe_normalize(embedding) if self._normalize else embedding

        with self._lock:
            cards, sims = self._search_locked(query_vec, top_k=top_k * 2)

        if tree_type_filter:
            filtered = [
                (card, sim) for card, sim in zip(cards, sims)
                if card.tree_type == tree_type_filter
            ]
        else:
            filtered = list(zip(cards, sims))

        return [card for card, _ in filtered[:top_k]]

    def recall_with_scores(
        self,
        query: str,
        *,
        top_k: int = 10,
        tree_type_filter: str | None = None,
    ) -> list[tuple[TreeCard, float]]:
        """Like recall() but also returns similarity scores."""
        if not self._cards:
            return []
        try:
            embedding = self._client.embed_texts([query])[0]
        except Exception:
            return []
        query_vec = _maybe_normalize(embedding) if self._normalize else embedding
        with self._lock:
            cards, sims = self._search_locked(query_vec, top_k=top_k * 2)
        pairs = list(zip(cards, sims))
        if tree_type_filter:
            pairs = [(c, s) for c, s in pairs if c.tree_type == tree_type_filter]
        return pairs[:top_k]

    def update_tree(self, tree_id: str, card: TreeCard, embedding: list[float]) -> None:
        """Alias for add_tree (replaces if exists)."""
        self.add_tree(tree_id, card, embedding)

    def size(self) -> int:
        with self._lock:
            return len(self._tree_ids)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Write index data to index_dir."""
        self._dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            ids_path = self._dir / "faiss_ids.json"
            ids_path.write_text(
                json.dumps(self._tree_ids, ensure_ascii=False), encoding="utf-8"
            )
            emb_path = self._dir / "embeddings.json"
            emb_path.write_text(
                json.dumps(self._embeddings, ensure_ascii=False), encoding="utf-8"
            )
            cards_data = {tid: _card_to_dict(c) for tid, c in self._cards.items()}
            cards_path = self._dir / "tree_cards.json"
            cards_path.write_text(
                json.dumps(cards_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def load(self) -> None:
        """Restore from disk."""
        ids_path = self._dir / "faiss_ids.json"
        emb_path = self._dir / "embeddings.json"
        cards_path = self._dir / "tree_cards.json"
        if not ids_path.exists():
            return
        with self._lock:
            self._tree_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            self._embeddings = json.loads(emb_path.read_text(encoding="utf-8"))
            cards_raw = json.loads(cards_path.read_text(encoding="utf-8"))
            self._cards = {tid: _card_from_dict(c) for tid, c in cards_raw.items()}
            self._index = None

    # ── internal ──────────────────────────────────────────────────────────────

    def _remove_tree_locked(self, tree_id: str) -> None:
        if tree_id not in self._cards:
            return
        idx = self._tree_ids.index(tree_id)
        self._tree_ids.pop(idx)
        self._embeddings.pop(idx)
        self._cards.pop(tree_id)

    def _search_locked(
        self,
        query_vec: list[float],
        top_k: int,
    ) -> tuple[list[TreeCard], list[float]]:
        """Brute-force inner-product search. Falls back to pure-Python if faiss unavailable."""
        n = len(self._embeddings)
        if n == 0:
            return [], []

        actual_k = min(top_k, n)

        # Try FAISS first
        try:
            import numpy as np
            if self._faiss is None:
                import faiss
                self._faiss = faiss
            faiss = self._faiss
            if self._index is None or True:  # always rebuild (small index)
                emb_array = np.array(self._embeddings, dtype=np.float32)
                idx = faiss.IndexFlatIP(self._dim)
                idx.add(emb_array)
                self._index = idx
            q = np.array([query_vec], dtype=np.float32)
            sims, indices = self._index.search(q, actual_k)
            cards = [self._cards[self._tree_ids[i]] for i in indices[0] if i >= 0]
            scores = [float(s) for s in sims[0] if s > -1e9]
            return cards, scores
        except ImportError:
            pass

        # Pure-Python fallback
        scored = [
            (_dot(query_vec, emb), i)
            for i, emb in enumerate(self._embeddings)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:actual_k]
        cards = [self._cards[self._tree_ids[i]] for _, i in top]
        scores = [s for s, _ in top]
        return cards, scores


# ── helpers ───────────────────────────────────────────────────────────────────

def _maybe_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-9:
        return vec
    return [x / norm for x in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _card_to_dict(c: TreeCard) -> dict:
    return {
        "tree_id": c.tree_id,
        "tree_type": c.tree_type,
        "label": c.label,
        "time_start": c.time_start,
        "time_end": c.time_end,
        "root_summary": c.root_summary,
        "item_count": c.item_count,
    }


def _card_from_dict(d: dict) -> TreeCard:
    return TreeCard(
        tree_id=str(d["tree_id"]),
        tree_type=str(d.get("tree_type", "entity")),
        label=str(d.get("label", "")),
        time_start=float(d.get("time_start", 0.0)),
        time_end=float(d.get("time_end", 0.0)),
        root_summary=str(d.get("root_summary", "")),
        item_count=int(d.get("item_count", 0)),
    )
