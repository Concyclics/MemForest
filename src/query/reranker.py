"""FactReranker: embedding cosine rerank over browse-collected facts.

No API calls — fact embeddings are pre-stored in FactManager at extraction time.
Rerank is purely in-memory: sort by cos_sim(fact_emb, query_emb).

Usage:
    reranker = FactReranker()
    top_facts = reranker.rerank(facts, query_emb, top_k=20)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.types import ManagedFact


class FactReranker:
    """Embedding-based fact reranker. No API calls."""

    def rerank(
        self,
        facts: "list[ManagedFact]",
        query_emb: list[float],
        *,
        top_k: int | None = 20,
    ) -> "list[ManagedFact]":
        """Return top_k facts sorted by cos_sim(fact_emb, query_emb) descending.

        Uses fact.metadata["embedding"] if present (stored at extraction time),
        otherwise assigns score 0.0 (fact appears last).

        top_k semantics: None or 0 → no truncation, return the full sorted list.
        """
        scored = self._score_facts(facts, query_emb)
        scored.sort(key=lambda x: x[0], reverse=True)
        if top_k is None or top_k <= 0:
            return [f for _, f in scored]
        return [f for _, f in scored[:top_k]]

    def rerank_with_scores(
        self,
        facts: "list[ManagedFact]",
        query_emb: list[float],
        *,
        top_k: int | None = 20,
    ) -> "list[tuple[float, ManagedFact]]":
        """Return (score, fact) pairs sorted descending. Useful for diagnostics.

        top_k semantics: None or 0 → no truncation, return the full sorted list.
        """
        scored = self._score_facts(facts, query_emb)
        scored.sort(key=lambda x: x[0], reverse=True)
        if top_k is None or top_k <= 0:
            return scored
        return scored[:top_k]

    # ── internal ──────────────────────────────────────────────────────────────

    def _score_facts(
        self,
        facts: "list[ManagedFact]",
        query_emb: list[float],
    ) -> "list[tuple[float, ManagedFact]]":
        q = _normalize(query_emb)
        result = []
        for fact in facts:
            emb = _get_fact_embedding(fact)
            if emb:
                score = _dot(q, emb)
            else:
                score = 0.0
            result.append((score, fact))
        return result


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_fact_embedding(fact: "ManagedFact") -> list[float] | None:
    """Extract embedding from fact. Stored in fact.metadata['embedding']."""
    emb = fact.metadata.get("embedding")
    if isinstance(emb, list) and emb:
        return emb
    return None


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-9:
        return list(vec)
    return [x / norm for x in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
