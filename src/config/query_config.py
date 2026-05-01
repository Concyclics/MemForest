"""Query-layer configuration for the Forest retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RecallConfig:
    """Configuration for the ForestRetriever.

    Default = PU10 (Union TD∪BU at K=10): reaches 100% essential_fact_coverage
    on LongMemEval 60q. See docs/experiments/recall/README.md §14.
    """

    top_k: int = 10
    """Per-direction top-K. For direction='union', the final recalled set is
    TD top_k ∪ BU top_k (deduped). PU10 hits 100% essential_fact_coverage."""

    direction: str = "union"
    """Recall direction: 'td' | 'bu' | 'union'.
      - td:    root-embedding FAISS → top_k trees (classic behavior).
      - bu:    fact-embedding FAISS → top-M facts → back-project to enclosing
               trees → rank by hit_count → top_k.
      - union: TD top_k ∪ BU top_k, deduped. Matches BU asymptote at 1/3 budget
               and reaches full recall at K=10. Requires FactManager."""

    bu_top_m: int = 30
    """BU fact over-fetch. top_m=30 was optimal in prior ablation
    (tree_build §5.1); raising it does not help."""

    force_entity_user: bool = True
    """Always include entity:user tree (force-appended if not in selection)."""

    llm_rerank: bool = False
    """If True, over-fetch top_k*llm_rerank_overfetch trees and LLM-rerank to
    top_k. Disabled: §14.3 shows zero net effect at 500–800ms cost per query."""

    llm_rerank_overfetch: int = 20
    """Overfetch multiplier for LLM reranker (top-N candidates sent to LLM)."""


@dataclass(frozen=True)
class BrowseConfig:
    """Configuration for the TreeBrowser (embedding beam search)."""

    beam_width: int = 10
    """Global beam width at each descent step.
    beam=10 → 98.9% essential-fact coverage on LongMemEval 60q; net +7.5pp
    end-to-end answer correctness vs beam=3 at PU10_F4 (v5 strict judge)."""

    max_facts: int = 0
    """Maximum facts returned to LLM context after FactReranker.
    0 or negative = no truncation (pass full browse output to answerer).
    Ablation: uncapped PU10_F4 reached 79.2% answer correctness. A small cap
    (e.g. 20) drops coverage on multi-fact preference/counting questions."""

    llm_guided: bool = True
    """If True, the LLM ranks children per internal node during beam descent.
    Combined with llm_guided_all_types=True this matches ablation F4 (llm+subq),
    which reached 79.2% on LongMemEval 60q @ PU10 beam=10."""

    llm_guided_all_types: bool = True
    """If True, llm_guided is applied to every browse plan regardless of
    browse_type. If False, LLM guidance is limited to anchor_a/anchor_b/
    preference trees (matches ablation F5, 67.9%)."""

    max_workers: int = 4
    """Parallel browse threads (one per BrowsePlan)."""


@dataclass(frozen=True)
class PlannerConfig:
    """Configuration for the BrowsePlanner (query decomposition)."""

    enabled: bool = True
    """If False, all trees get sub_query = original question (lightweight pass-through).
    If True, 1 LLM call self-analyzes and decomposes temporal/multi-session/preference questions."""

    model_name: str | None = None
    """Override LLM model name for planner. None = inherit from config.api.llm.model_name."""


@dataclass(frozen=True)
class QueryConfig:
    """Top-level query configuration.

    Two presets:
      - mode="lightweight": embedding only everywhere, no LLM in recall/browse/plan.
        LLM budget: 0 (recall) + 0 (browse) + 0 (plan) + 1 (answer) = 1 call.
      - mode="agentic" (default): planner self-analyzes and decomposes queries.
        LLM budget: 0 (recall) + 0 (browse) + 1 (plan) + 1 (answer) = 2 calls.
      - Optionally: recall.llm_rerank=True or browse.llm_guided=True for targeted gains.
    """

    mode: str = "agentic"
    """Preset name: "lightweight" | "agentic". Controls defaults; individual sub-configs override."""

    recall: RecallConfig = field(default_factory=RecallConfig)
    browse: BrowseConfig = field(default_factory=BrowseConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)

    node_index_vector_dim: int = 1024
    """Must match the embedding model dimension (Qwen3-Embedding-0.6B = 1024)."""


# ── presets ───────────────────────────────────────────────────────────────────

LIGHTWEIGHT = QueryConfig(
    mode="lightweight",
    recall=RecallConfig(top_k=10, direction="union", llm_rerank=False),
    browse=BrowseConfig(
        beam_width=3,
        llm_guided=False,
        llm_guided_all_types=False,
        max_facts=20,
    ),
    planner=PlannerConfig(enabled=False),
)

AGENTIC = QueryConfig(
    mode="agentic",
    recall=RecallConfig(top_k=10, direction="union", llm_rerank=False),
    browse=BrowseConfig(
        beam_width=10,
        llm_guided=True,
        llm_guided_all_types=True,
        max_facts=0,
    ),
    planner=PlannerConfig(enabled=True),
)

_PRESETS: dict[str, QueryConfig] = {
    "lightweight": LIGHTWEIGHT,
    "agentic": AGENTIC,
}


def get_preset(mode: str) -> QueryConfig:
    """Return a preset QueryConfig by mode name."""
    if mode not in _PRESETS:
        raise ValueError(f"Unknown query mode {mode!r}. Choose from: {list(_PRESETS)}")
    return _PRESETS[mode]
