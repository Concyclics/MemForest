"""ForestRetriever: multi-direction tree recall.

Directions (config.recall.direction):
  - td    : root-embedding FAISS → top_k trees (classic).
  - bu    : query-embedding → fact FAISS top-M → back-project to trees →
            rank by hit_count → top_k. Requires FactManager.
  - union : TD top_k ∪ BU top_k, deduped. Default; matches BU's asymptote at
            1/3 of the budget and reaches full recall at K=10. See
            docs/experiments/recall/README.md §14.

LLM rerank remains available via config.recall.llm_rerank for the TD pool
path, but ablation shows net-zero at 500–800ms cost — disabled by default.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import TYPE_CHECKING

from src.build.node_index import NodeIndex
from src.build.tree_types import MemTree, SessionLeaf, TreeCard

if TYPE_CHECKING:
    from src.api.client import OpenAIChatClient, OpenAIEmbeddingClient
    from src.build.tree_store import TreeStore
    from src.config.query_config import RecallConfig
    from src.extraction.fact_manager import FactManager

log = logging.getLogger(__name__)


class ForestRetriever:
    """Multi-direction tree recall: TD / BU / union."""

    def __init__(
        self,
        *,
        embedding_client: "OpenAIEmbeddingClient",
        node_index: NodeIndex,
        tree_store: "TreeStore",
        config: "RecallConfig",
        chat_client: "OpenAIChatClient | None" = None,
        fact_manager: "FactManager | None" = None,
    ) -> None:
        self._emb_client = embedding_client
        self._node_index = node_index
        self._tree_store = tree_store
        self._config = config
        self._chat_client = chat_client
        self._fact_manager = fact_manager

        # fact_id → set(tree_id) cache, built lazily on first BU query.
        self._fact_to_trees: dict[str, set[str]] | None = None
        self._fact_to_trees_lock = threading.Lock()

    # ── public API ────────────────────────────────────────────────────────────

    def recall(
        self,
        question: str,
        *,
        top_k: int | None = None,
    ) -> list[TreeCard]:
        cards, _ = self.recall_with_query_emb(question, top_k=top_k)
        return cards

    def recall_with_query_emb(
        self,
        question: str,
        *,
        top_k: int | None = None,
    ) -> tuple[list[TreeCard], list[float]]:
        """Return (tree_cards, query_emb). query_emb reused downstream."""
        k = top_k if top_k is not None else self._config.top_k
        query_emb = self._embed_question(question)

        direction = self._resolved_direction()

        if direction == "td":
            if self._config.llm_rerank and self._chat_client is not None:
                selected = self._recall_td_with_rerank(query_emb, question, k)
            else:
                selected = self._recall_td(query_emb, k)
        elif direction == "bu":
            selected = self._recall_bu(query_emb, k)
        elif direction == "union":
            td_ids = self._recall_td(query_emb, k)
            bu_ids = self._recall_bu(query_emb, k)
            selected = _merge_preserving_order(td_ids, bu_ids)
        else:
            raise ValueError(f"Unknown recall direction {direction!r}")

        self._force_include(selected)
        return self._load_cards(selected), query_emb

    def invalidate_fact_index(self) -> None:
        """Drop the cached fact_id→tree_ids map. Call after tree_store mutation."""
        with self._fact_to_trees_lock:
            self._fact_to_trees = None

    # ── TD ────────────────────────────────────────────────────────────────────

    def _recall_td(self, query_emb: list[float], top_k: int) -> list[str]:
        """Root-only FAISS: one entry per tree, rank by cos_sim, take top_k."""
        node_hits = self._node_index.search(query_emb, top_n=top_k)
        return [entry.tree_id for entry in node_hits]

    # ── BU ────────────────────────────────────────────────────────────────────

    def _recall_bu(self, query_emb: list[float], top_k: int) -> list[str]:
        """Fact-level FAISS → back-project to enclosing trees → rank by hits."""
        if self._fact_manager is None:
            log.warning(
                "recall.direction requests BU but fact_manager is None; falling back to TD"
            )
            return self._recall_td(query_emb, top_k)

        top_m = max(top_k, int(self._config.bu_top_m))
        try:
            hits = self._fact_manager.search_similar_fact_by_vector(
                query_emb, top_k=top_m
            )
        except Exception:  # pragma: no cover
            log.exception("BU fact search failed; falling back to TD")
            return self._recall_td(query_emb, top_k)

        if not hits:
            return []

        fact_to_trees = self._get_fact_to_trees()
        tree_score: dict[str, int] = {}
        for fact, _score in hits:
            for tid in fact_to_trees.get(fact.fact_id, ()):
                tree_score[tid] = tree_score.get(tid, 0) + 1

        ordered = sorted(tree_score.items(), key=lambda kv: (-kv[1], kv[0]))
        return [tid for tid, _ in ordered[:top_k]]

    def _get_fact_to_trees(self) -> dict[str, set[str]]:
        """Lazy-build fact_id → {tree_id, ...} from tree_store."""
        with self._fact_to_trees_lock:
            if self._fact_to_trees is not None:
                return self._fact_to_trees
            mapping: dict[str, set[str]] = {}
            for tree_id in self._tree_store.all_tree_ids():
                tree = self._tree_store.get(tree_id)
                if tree is None:
                    continue
                for fid in _tree_fact_ids(tree):
                    mapping.setdefault(fid, set()).add(tree.tree_id)
            self._fact_to_trees = mapping
            return mapping

    # ── TD + LLM rerank (optional, kept for K=1 niche) ────────────────────────

    def _recall_td_with_rerank(
        self,
        query_emb: list[float],
        question: str,
        top_k: int,
    ) -> list[str]:
        overfetch = self._config.llm_rerank_overfetch
        candidate_ids = self._recall_td(query_emb, max(overfetch, top_k))
        candidate_cards = self._load_cards(candidate_ids)
        if not candidate_cards or self._chat_client is None:
            return candidate_ids[:top_k]

        summaries = []
        for i, card in enumerate(candidate_cards):
            preview = (card.root_summary or "")[:500]
            summaries.append(f"[{i}] tree_id={card.tree_id} label={card.label}: {preview}")
        summary_block = "\n".join(summaries)

        system_prompt = (
            "You are a memory retrieval ranker. Given a question and a list of memory tree "
            "summaries (each identified by index), select the most relevant tree indices "
            f"to answer the question. Return exactly {top_k} indices.\n"
            f'Output JSON only: {{"selected_indices": [int, ...]}}'
        )
        user_prompt = f"Question: {question}\n\nMemory trees:\n{summary_block}"

        try:
            result = self._chat_client.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                step_label="recall_llm_rerank",
            )
            indices = result.get("selected_indices", [])
            valid = [i for i in indices if isinstance(i, int) and 0 <= i < len(candidate_cards)]
            selected = [candidate_cards[i].tree_id for i in valid[:top_k]]
        except Exception:
            selected = candidate_ids[:top_k]

        for tid in candidate_ids:
            if len(selected) >= top_k:
                break
            if tid not in selected:
                selected.append(tid)
        return selected

    # ── helpers ───────────────────────────────────────────────────────────────

    def _resolved_direction(self) -> str:
        d = str(getattr(self._config, "direction", "td") or "td").lower()
        if d in ("bu", "union") and self._fact_manager is None:
            log.warning(
                "recall.direction=%r but fact_manager is None; falling back to 'td'", d
            )
            return "td"
        if d not in ("td", "bu", "union"):
            raise ValueError(f"Unknown recall direction {d!r}")
        return d

    def _embed_question(self, question: str) -> list[float]:
        raw = self._emb_client.embed_texts([question])[0]
        return _normalize(raw)

    def _force_include(self, selected: list[str]) -> None:
        if not self._config.force_entity_user:
            return
        for forced_id in ("entity:user",):
            if forced_id not in selected and self._tree_store.get(forced_id):
                selected.append(forced_id)

    def _load_cards(self, tree_ids: list[str]) -> list[TreeCard]:
        cards = []
        for tid in tree_ids:
            card = self._tree_store.get_tree_card(tid)
            if card is not None:
                cards.append(card)
        return cards


# ── module-level helpers ─────────────────────────────────────────────────────

def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-9:
        return list(vec)
    return [x / norm for x in vec]


def _tree_fact_ids(tree: MemTree) -> set[str]:
    """Collect all fact_ids contained in `tree`.

    Session trees keep facts inside SessionLeaf.fact_ids; entity/scene trees
    keep them in fact_ids_ordered.
    """
    out: set[str] = set()
    for fid in tree.fact_ids_ordered or ():
        out.add(fid)
    for leaf in (tree.session_leaves or {}).values():
        if isinstance(leaf, SessionLeaf):
            for fid in leaf.fact_ids or ():
                out.add(fid)
        else:  # dict form after load
            for fid in (leaf.get("fact_ids") or ()):
                out.add(fid)
    return out


def _merge_preserving_order(a: list[str], b: list[str]) -> list[str]:
    """Merge two tree-id lists, preserving priority (a first), deduping."""
    seen: set[str] = set()
    out: list[str] = []
    for tid in a:
        if tid not in seen:
            out.append(tid)
            seen.add(tid)
    for tid in b:
        if tid not in seen:
            out.append(tid)
            seen.add(tid)
    return out
