"""Embedding-first, LLM-confirmed duplicate fact removal."""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.prompt import FACT_EQUIVALENCE_PROMPT_NAME, build_dedup_prompt_manager
from src.utils.types import MemoryItem


class FactEquivalenceJudge(Protocol):
    def are_equivalent(
        self,
        fact_a: str,
        fact_b: str,
        *,
        trace: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Return (equivalent, preferred_side)."""

    def are_equivalent_batch(
        self,
        pairs: list[tuple[str, str]],
        *,
        traces: list[dict[str, Any] | None] | None = None,
        max_workers: int = 64,
    ) -> list[tuple[bool, str]]:
        """Return (equivalent, preferred_side) for each pair, concurrently."""
        ...


class EmbeddingTextClient(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for the given texts."""


class JsonChatClient(Protocol):
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        step_label: str = "llm",
        trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a parsed JSON object for the prompt pair."""


@dataclass(frozen=True)
class DedupDecision:
    index_a: int
    index_b: int
    similarity: float
    equivalent: bool
    preferred: str
    reason: str = ""


@dataclass(frozen=True)
class TextDedupResult:
    kept_texts: list[str]
    kept_indices: list[int]
    removed_indices: list[int]
    groups: list[list[int]]
    decisions: list[DedupDecision] = field(default_factory=list)


@dataclass(frozen=True)
class MemoryItemDedupResult:
    kept_items: list[MemoryItem]
    kept_indices: list[int]
    removed_indices: list[int]
    groups: list[list[int]]
    decisions: list[DedupDecision] = field(default_factory=list)


class LLMFactEquivalenceJudge:
    """LLM-backed factual equivalence check for near-duplicate memory facts."""

    def __init__(
        self,
        *,
        chat_client: JsonChatClient,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        top_p: float = 0.2,
    ) -> None:
        self.chat_client = chat_client
        self.model_name = model_name
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.top_p = float(top_p)
        self.prompt_manager = build_dedup_prompt_manager()

    def are_equivalent(
        self,
        fact_a: str,
        fact_b: str,
        *,
        trace: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        system_prompt, user_prompt = self.prompt_manager.render(
            FACT_EQUIVALENCE_PROMPT_NAME,
            {"fact_a": fact_a, "fact_b": fact_b},
        )
        payload = self.chat_client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            step_label="dedup_judge",
            trace=trace,
        )
        equivalent = bool(payload.get("equivalent", False))
        preferred = str(payload.get("preferred", "either")).strip().lower()
        if preferred not in {"a", "b", "either"}:
            preferred = "either"
        return equivalent, preferred

    def are_equivalent_batch(
        self,
        pairs: list[tuple[str, str]],
        *,
        traces: list[dict[str, Any] | None] | None = None,
        max_workers: int = 64,
    ) -> list[tuple[bool, str]]:
        """Evaluate multiple fact pairs concurrently."""
        if not pairs:
            return []
        if traces is None:
            traces = [None] * len(pairs)
        results: list[tuple[bool, str] | None] = [None] * len(pairs)
        with ThreadPoolExecutor(max_workers=min(max_workers, len(pairs))) as pool:
            futures = {
                pool.submit(self.are_equivalent, a, b, trace=t): idx
                for idx, ((a, b), t) in enumerate(zip(pairs, traces))
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = (False, "either")
        return [r if r is not None else (False, "either") for r in results]


def deduplicate_fact_texts(
    texts: list[str],
    *,
    embedding_client: EmbeddingTextClient,
    judge: FactEquivalenceJudge | None = None,
    similarity_threshold: float = 0.965,
    max_llm_pairs: int = 32,
    trace_prefix: str = "dedup",
) -> TextDedupResult:
    cleaned = [str(text or "").strip() for text in texts]
    uf = _UnionFind(len(cleaned))
    decisions: list[DedupDecision] = []

    exact_groups: dict[str, list[int]] = {}
    for idx, text in enumerate(cleaned):
        key = _normalize_fact_text(text)
        if not key:
            continue
        exact_groups.setdefault(key, []).append(idx)
    for indices in exact_groups.values():
        if len(indices) <= 1:
            continue
        head = indices[0]
        for other in indices[1:]:
            uf.union(head, other)
            decisions.append(
                DedupDecision(
                    index_a=head,
                    index_b=other,
                    similarity=1.0,
                    equivalent=True,
                    preferred="a",
                    reason="exact_normalized_match",
                )
            )

    if judge is not None:
        embeddings = embedding_client.embed_texts(cleaned)
        candidates = _candidate_pairs(cleaned, embeddings, similarity_threshold=similarity_threshold)
        llm_pairs_used = 0
        for idx_a, idx_b, similarity in candidates:
            if uf.find(idx_a) == uf.find(idx_b):
                continue
            if llm_pairs_used >= int(max_llm_pairs):
                break
            equivalent, preferred = judge.are_equivalent(
                cleaned[idx_a],
                cleaned[idx_b],
                trace={"request_id": f"{trace_prefix}:{idx_a}:{idx_b}"},
            )
            llm_pairs_used += 1
            decisions.append(
                DedupDecision(
                    index_a=idx_a,
                    index_b=idx_b,
                    similarity=similarity,
                    equivalent=equivalent,
                    preferred=preferred,
                    reason="llm_equivalence" if equivalent else "llm_non_equivalence",
                )
            )
            if equivalent:
                uf.union(idx_a, idx_b)

    groups = _groups_from_union_find(uf, len(cleaned))
    kept_indices: list[int] = []
    removed_indices: list[int] = []
    for group in groups:
        best_idx = _choose_text_representative(cleaned, group, decisions)
        kept_indices.append(best_idx)
        removed_indices.extend(idx for idx in group if idx != best_idx)
    kept_indices = sorted(kept_indices)
    removed_indices = sorted(removed_indices)
    return TextDedupResult(
        kept_texts=[cleaned[idx] for idx in kept_indices],
        kept_indices=kept_indices,
        removed_indices=removed_indices,
        groups=groups,
        decisions=decisions,
    )


def deduplicate_memory_items(
    items: list[MemoryItem],
    *,
    embedding_client: EmbeddingTextClient,
    judge: FactEquivalenceJudge | None = None,
    similarity_threshold: float = 0.965,
    max_llm_pairs: int = 32,
    trace_prefix: str = "dedup",
) -> MemoryItemDedupResult:
    texts = [item.fact_text for item in items]
    text_result = deduplicate_fact_texts(
        texts,
        embedding_client=embedding_client,
        judge=judge,
        similarity_threshold=similarity_threshold,
        max_llm_pairs=max_llm_pairs,
        trace_prefix=trace_prefix,
    )
    groups = text_result.groups
    kept_indices: list[int] = []
    removed_indices: list[int] = []
    kept_items: list[MemoryItem] = []
    for group in groups:
        best_idx = _choose_item_representative(items, group, text_result.decisions)
        kept_indices.append(best_idx)
        kept_items.append(items[best_idx])
        removed_indices.extend(idx for idx in group if idx != best_idx)
    kept_indices = sorted(kept_indices)
    removed_indices = sorted(removed_indices)
    kept_items = [items[idx] for idx in kept_indices]
    return MemoryItemDedupResult(
        kept_items=kept_items,
        kept_indices=kept_indices,
        removed_indices=removed_indices,
        groups=groups,
        decisions=text_result.decisions,
    )


def _candidate_pairs(
    texts: list[str],
    embeddings: list[list[float]],
    *,
    similarity_threshold: float,
) -> list[tuple[int, int, float]]:
    pairs: list[tuple[int, int, float]] = []
    for i in range(len(texts)):
        if not texts[i]:
            continue
        for j in range(i + 1, len(texts)):
            if not texts[j]:
                continue
            similarity = _cosine_similarity(embeddings[i], embeddings[j])
            if similarity >= float(similarity_threshold):
                pairs.append((i, j, similarity))
    pairs.sort(key=lambda row: row[2], reverse=True)
    return pairs


def _choose_text_representative(
    texts: list[str],
    group: list[int],
    decisions: list[DedupDecision],
) -> int:
    preferred_score: dict[int, int] = {idx: 0 for idx in group}
    for decision in decisions:
        if not decision.equivalent:
            continue
        if decision.index_a in preferred_score and decision.index_b in preferred_score:
            if decision.preferred == "a":
                preferred_score[decision.index_a] += 1
            elif decision.preferred == "b":
                preferred_score[decision.index_b] += 1
    return max(
        group,
        key=lambda idx: (
            preferred_score.get(idx, 0),
            _text_specificity_score(texts[idx]),
            -idx,
        ),
    )


def _choose_item_representative(
    items: list[MemoryItem],
    group: list[int],
    decisions: list[DedupDecision],
) -> int:
    preferred_score: dict[int, int] = {idx: 0 for idx in group}
    for decision in decisions:
        if not decision.equivalent:
            continue
        if decision.index_a in preferred_score and decision.index_b in preferred_score:
            if decision.preferred == "a":
                preferred_score[decision.index_a] += 1
            elif decision.preferred == "b":
                preferred_score[decision.index_b] += 1
    return max(
        group,
        key=lambda idx: (
            preferred_score.get(idx, 0),
            float(items[idx].confidence),
            len(items[idx].source_spans),
            _text_specificity_score(items[idx].fact_text),
            -idx,
        ),
    )


def _text_specificity_score(text: str) -> tuple[int, int, int]:
    raw = str(text or "")
    digit_count = sum(1 for ch in raw if ch.isdigit())
    uppercase_tokens = sum(1 for token in raw.split() if len(token) > 1 and token[:1].isupper())
    return (digit_count, uppercase_tokens, len(raw))


def _normalize_fact_text(text: str) -> str:
    lowered = str(text or "").strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _groups_from_union_find(uf: "_UnionFind", size: int) -> list[list[int]]:
    groups: dict[int, list[int]] = {}
    for idx in range(size):
        root = uf.find(idx)
        groups.setdefault(root, []).append(idx)
    return [sorted(group) for _, group in sorted(groups.items(), key=lambda row: min(row[1]))]


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra
