"""ForestQuery: end-to-end recall → plan → browse → rerank pipeline.

Entry point for the query layer. Composes ForestRetriever, BrowsePlanner,
TreeBrowser, and FactReranker into a single query() call.

LLM call budget (per query):
  - lightweight mode: 0 (recall) + 0 (plan) + 0 (browse) + 1 (answer) = 1 total
  - agentic mode:     0 (recall) + 1 (plan) + 0 (browse) + 1 (answer) = 2 total
  - agentic+llm_guided: 0 + 1 + N_nodes (browse) + 1 (answer) = 2 + N_nodes
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.query.browser import TreeBrowser
from src.query.planner import BrowsePlan, BrowsePlanner
from src.query.reranker import FactReranker
from src.query.retriever import ForestRetriever

if TYPE_CHECKING:
    from src.api.client import OpenAIChatClient, OpenAIEmbeddingClient
    from src.build.node_index import NodeIndex
    from src.build.tree_store import TreeStore
    from src.build.tree_types import TreeCard
    from src.config.query_config import QueryConfig
    from src.extraction.fact_manager import FactManager
    from src.utils.types import ManagedFact


@dataclass
class QueryResult:
    """Full output of one ForestQuery.query() call."""

    question: str
    question_type: str                    # planner's analysis ("simple" | "temporal" | ...)
    recalled_trees: "list[TreeCard]"      # K trees after recall
    browse_plans: list[BrowsePlan]        # one per (tree, sub_query) pair
    raw_ids: list[str]                    # all fact/cell ids from browse (pre-rerank)
    top_facts: "list[ManagedFact]"        # top_k after FactReranker
    session_date: str | None              # injected for temporal questions
    query_emb: list[float]               # normalized query embedding (for diagnostics)
    planner_explanation: str = ""        # planner's reasoning (for debug)
    n_raw_facts: int = 0                 # count before reranking


class ForestQuery:
    """Orchestrates the full recall → plan → browse → rerank query pipeline."""

    def __init__(
        self,
        *,
        embedding_client: "OpenAIEmbeddingClient",
        chat_client: "OpenAIChatClient",
        tree_store: "TreeStore",
        node_index: "NodeIndex",
        fact_loader: "FactLoader | None" = None,
        fact_manager: "FactManager | None" = None,
        config: "QueryConfig",
    ) -> None:
        self._emb_client = embedding_client
        self._chat_client = chat_client
        self._tree_store = tree_store
        self._node_index = node_index
        self._fact_manager = fact_manager
        self._fact_loader = fact_loader or NullFactLoader()
        self._config = config

        self._retriever = ForestRetriever(
            embedding_client=embedding_client,
            node_index=node_index,
            tree_store=tree_store,
            config=config.recall,
            chat_client=chat_client if config.recall.llm_rerank else None,
            fact_manager=fact_manager,
        )
        self._planner = BrowsePlanner(
            config=config.planner,
            chat_client=chat_client if config.planner.enabled else None,
            default_model_name="",
        )
        self._browser = TreeBrowser(
            node_index=node_index,
            tree_store=tree_store,
            config=config.browse,
            embedding_client=embedding_client,
            chat_client=chat_client if config.browse.llm_guided else None,
        )
        self._reranker = FactReranker()

    def query(
        self,
        question: str,
        *,
        query_time: float | None = None,
        session_alias_map: dict | None = None,
        top_k: int | None = None,
        beam_width: int | None = None,
        max_facts: int | None = None,
    ) -> QueryResult:
        """Run the full pipeline for one question.

        Args:
            question: Natural language question.
            query_time: Unix timestamp for temporal reference-date injection.
            session_alias_map: {alias → {session_id, time_end, ...}} for date injection.
            top_k: Override config.recall.top_k.
            beam_width: Override config.browse.beam_width.
            max_facts: Override config.browse.max_facts.

        Returns:
            QueryResult with recalled trees, browse plans, and top_k reranked facts.
        """
        # max_facts semantics: None → use config default; 0 or negative → no cap
        n_facts = max_facts if max_facts is not None else self._config.browse.max_facts
        rerank_top_k: int | None = n_facts if n_facts and n_facts > 0 else None

        # 1. Recall: M4 augmented FAISS → top_k trees + query_emb
        tree_cards, query_emb = self._retriever.recall_with_query_emb(
            question,
            top_k=top_k,
        )

        # 2. Plan: sub-query decomposition (1 LLM call if planner enabled)
        browse_plans = self._planner.plan(question, tree_cards)
        question_type = "simple"
        planner_explanation = ""

        # Extract question_type from planner internals via last decomposition
        # (BrowsePlanner stores last result internally for diagnostics)
        if hasattr(self._planner, "_last_decomposition") and self._planner._last_decomposition:  # type: ignore[attr-defined]
            dec = self._planner._last_decomposition  # type: ignore[attr-defined]
            question_type = dec.question_type
            planner_explanation = dec.explanation

        # 3. Browse: parallel embedding beam search → fact/cell ids
        raw_ids = self._browser.browse_all(
            browse_plans,
            query_emb=query_emb,
            beam_width=beam_width,
        )

        # 4. Load facts from raw_ids
        facts = self._fact_loader.load(raw_ids, tree_store=self._tree_store)

        # 5. Rerank: sort by cos_sim(fact_emb, query_emb), optionally truncate
        top_facts = self._reranker.rerank(facts, query_emb, top_k=rerank_top_k)

        # 6. Session date injection for temporal questions
        session_date: str | None = None
        if query_time is not None:
            from src.utils.time import render_time_text

            session_date = render_time_text(query_time)
        elif session_alias_map:
            session_date = _extract_session_date(session_alias_map)

        return QueryResult(
            question=question,
            question_type=question_type,
            recalled_trees=tree_cards,
            browse_plans=browse_plans,
            raw_ids=raw_ids,
            top_facts=top_facts,
            session_date=session_date,
            query_emb=query_emb,
            planner_explanation=planner_explanation,
            n_raw_facts=len(facts),
        )

    def build_context(self, result: QueryResult) -> str:
        """Render QueryResult into a text context block for LLM reasoning.

        Includes:
          - Recalled tree labels
          - Top facts (with time text)
          - Session date (if available, for temporal questions)
        """
        lines: list[str] = []

        if result.session_date:
            lines.append(f"[Reference date: {result.session_date}]\n")

        lines.append(f"Retrieved {len(result.top_facts)} memory facts:\n")
        for i, fact in enumerate(result.top_facts, 1):
            time_tag = _render_fact_time_tag(fact.time_start, fact.time_end)
            lines.append(f"{i}. {fact.fact_text}{time_tag}")

        return "\n".join(lines)


# ── FactLoader interface ──────────────────────────────────────────────────────

class FactLoader:
    """Abstract fact loader. Resolves fact_ids / cell_ids → ManagedFact list."""

    def load(
        self,
        ids: list[str],
        *,
        tree_store: "TreeStore | None" = None,
    ) -> "list[ManagedFact]":
        raise NotImplementedError


class NullFactLoader(FactLoader):
    """Returns empty list. Use when no fact store is available."""

    def load(self, ids, *, tree_store=None):
        return []


class JsonlFactLoader(FactLoader):
    """Loads facts from a canonical_facts.jsonl file (ablation-style setup).

    Used when the tree index is built from full500_facts_build data
    (i.e., FactManager is not used directly).
    """

    def __init__(self, facts_jsonl_path: str | Path) -> None:
        self._path = Path(facts_jsonl_path)
        self._cache: dict[str, "ManagedFact"] | None = None

    def _ensure_loaded(self) -> None:
        if self._cache is not None:
            return
        from src.utils.types import ManagedFact, FactOccurrence

        cache: dict[str, "ManagedFact"] = {}
        if not self._path.exists():
            self._cache = cache
            return
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    # Reconstruct ManagedFact from JSONL dict
                    fact = ManagedFact(
                        fact_id=str(d["fact_id"]),
                        fact_text=str(d.get("fact_text", "")),
                        embedding_id=str(d.get("embedding_id", "")),
                        origin=str(d.get("origin", "")),
                        semantic_role=str(d.get("semantic_role", "")),
                        entities=list(d.get("entities", [])),
                        topics=list(d.get("topics", [])),
                        attribute_keys=list(d.get("attribute_keys", [])),
                        domain_keys=list(d.get("domain_keys", [])),
                        collection_keys=list(d.get("collection_keys", [])),
                        detail_level=str(d.get("detail_level", "")),
                        confidence=float(d.get("confidence", 1.0)),
                        first_session_id=str(d.get("first_session_id", "")),
                        first_cell_id=str(d.get("first_cell_id", "")),
                        occurrences=[],
                        metadata=dict(d.get("metadata", {})),
                        time_start=d.get("time_start"),
                        time_end=d.get("time_end"),
                    )
                    cache[fact.fact_id] = fact
                except Exception:
                    pass
        self._cache = cache

    def load(
        self,
        ids: list[str],
        *,
        tree_store: "TreeStore | None" = None,
    ) -> "list[ManagedFact]":
        self._ensure_loaded()
        assert self._cache is not None
        result = []
        for fid in ids:
            fact = self._cache.get(fid)
            if fact is not None:
                result.append(fact)
            elif tree_store is not None:
                # Try resolving as cell_id (session tree leaf)
                cell = tree_store.get_cell(fid)
                if cell is not None:
                    # Synthesize a pseudo-fact from session cell raw_text
                    stub = _cell_to_stub_fact(fid, cell)
                    result.append(stub)
        return result


class FactManagerLoader(FactLoader):
    """Loads facts from a FactManager instance."""

    def __init__(self, fact_manager: "object", session_registry: "object | None" = None) -> None:
        self._fm = fact_manager
        self._registry = session_registry

    def load(
        self,
        ids: list[str],
        *,
        tree_store: "TreeStore | None" = None,
    ) -> "list[ManagedFact]":
        result = []
        seen: set[str] = set()
        for fid in ids:
            fact = self._fm.get_fact(fid)  # type: ignore[attr-defined]
            if fact is not None:
                if fact.fact_id not in seen:
                    seen.add(fact.fact_id)
                    result.append(fact)
                continue

            if self._registry is not None:
                try:
                    cell_fact_ids = self._registry.get_cell_fact_ids(fid)  # type: ignore[attr-defined]
                except Exception:
                    cell_fact_ids = []
                for cell_fact_id in cell_fact_ids:
                    cell_fact = self._fm.get_fact(cell_fact_id)  # type: ignore[attr-defined]
                    if cell_fact is not None and cell_fact.fact_id not in seen:
                        seen.add(cell_fact.fact_id)
                        result.append(cell_fact)
                if cell_fact_ids:
                    continue

            if tree_store is not None:
                cell = tree_store.get_cell(fid)
                if cell is not None:
                    stub = _cell_to_stub_fact(fid, cell)
                    if stub.fact_id not in seen:
                        seen.add(stub.fact_id)
                        result.append(stub)
        return result


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_session_date(session_alias_map: dict) -> str | None:
    """Extract the latest session date from alias map for temporal question injection.

    session_alias_map format: {alias → {session_id, time_end, ...}} or
                              {session_id → {alias, time_end, ...}}
    """
    max_time: float = 0.0
    max_time_str: str | None = None
    for _alias, info in session_alias_map.items():
        if not isinstance(info, dict):
            continue
        t = info.get("time_end") or info.get("end_time") or 0.0
        try:
            t = float(t)
        except (TypeError, ValueError):
            continue
        if t > max_time:
            max_time = t
            # Try a human-readable date string
            ts_str = info.get("time_end_str") or info.get("date_str")
            if ts_str:
                max_time_str = str(ts_str)
            else:
                import datetime
                try:
                    dt = datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc)
                    max_time_str = dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
    return max_time_str


def _render_fact_time_tag(time_start: float | None, time_end: float | None) -> str:
    if time_start is None:
        return ""
    from src.utils.time import render_time_text

    try:
        start_text = render_time_text(time_start)
    except (ValueError, OverflowError, OSError):
        return ""
    if time_end is None or float(time_end) == float(time_start):
        return f" [{start_text}]"
    try:
        end_text = render_time_text(time_end)
    except (ValueError, OverflowError, OSError):
        return f" [{start_text}]"
    return f" [{start_text} -> {end_text}]"


def _cell_to_stub_fact(cell_id: str, cell: "object") -> "ManagedFact":
    """Create a ManagedFact stub from a SessionLeaf for use in the context window."""
    from src.utils.types import ManagedFact

    raw_text = getattr(cell, "raw_text", "") or ""
    time_start = getattr(cell, "time_start", None)
    time_end = getattr(cell, "time_end", None)
    session_id = getattr(cell, "session_id", "")

    return ManagedFact(
        fact_id=cell_id,
        fact_text=raw_text[:2000],   # truncate very long cells
        embedding_id="",
        origin="session",
        semantic_role="event",
        entities=[],
        topics=[],
        attribute_keys=[],
        domain_keys=[],
        collection_keys=[],
        detail_level="high",
        confidence=1.0,
        first_session_id=session_id,
        first_cell_id=cell_id,
        occurrences=[],
        metadata={},
        time_start=time_start,
        time_end=time_end,
    )
