"""BrowsePlanner: query decomposition for targeted tree browsing.

Empirically grounded design:
  - System has no pre-classified question type (no external classifier)
  - Planner self-analyzes with 1 LLM call (when enabled=True)
  - Decides question type: temporal | multi-session | preference | simple
  - Generates targeted sub-queries for each type
  - When disabled (lightweight mode): all trees get sub_query = question

LLM call budget:
  - planner.enabled=False: 0 calls
  - planner.enabled=True:  1 call (regardless of question count)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.api.client import OpenAIChatClient
    from src.build.tree_types import TreeCard
    from src.config.query_config import PlannerConfig

# Valid browse types
BROWSE_TYPE_DIRECT = "direct"
BROWSE_TYPE_ANCHOR_A = "anchor_a"
BROWSE_TYPE_ANCHOR_B = "anchor_b"
BROWSE_TYPE_AGGREGATE = "aggregate"
BROWSE_TYPE_PREFERENCE = "preference"


@dataclass(frozen=True)
class BrowsePlan:
    """One (tree, sub_query) browse directive."""

    tree_id: str
    sub_query: str
    browse_type: str = BROWSE_TYPE_DIRECT
    anchor_label: str = ""


@dataclass
class DecompositionResult:
    """Planner's self-analysis of the question."""

    needs_decomposition: bool
    question_type: str  # "temporal" | "multi-session" | "preference" | "simple"
    sub_queries: list[str] = field(default_factory=list)
    explanation: str = ""


_PLANNER_SYSTEM = """\
You are a memory retrieval query planner. Given a user question and a list
of candidate memory trees (with tree_id, tree_type, label, and a short root
summary), produce a targeted retrieval plan that emits one or more
(tree_id, sub_query) pairs — each sub_query TAILORED to what that specific
tree likely contains.

Return JSON only (no markdown):
{
  "question_type": "temporal" | "multi-session" | "preference" | "simple",
  "plans": [
    {"tree_id": "...", "sub_query": "...", "browse_type": "direct"}
  ],
  "explanation": "..."
}

Tree-type hints (use the tree_type field to frame sub_query):
- session: tree groups facts from one conversation window. Sub_query should
  ask "what did <topic> look like during this session / around <date>".
- entity: tree groups facts about one person/place/thing. Sub_query should
  ask about that entity in relation to the question.
- scene: tree groups facts about a recurring activity/hobby. Sub_query should
  ask about the aspect of the scene the question cares about.

Per-tree rules:
- Read each tree's label + root_summary. If the tree is clearly irrelevant
  (topic mismatch), still emit a plan with sub_query = the original question
  and browse_type = "direct" (the downstream browser will prune).
- If relevant, rewrite sub_query to anchor on what that tree actually holds,
  using tree-specific entities/dates from its summary when possible.
- For most questions you emit ONE plan per tree (browse_type = "direct").

Multi-plan patterns (emit more than one plan for the same tree_id):
- temporal (duration / time-gap between two events): for each tree, emit TWO
  plans — one anchored on event A (browse_type = "anchor_a") and one on
  event B (browse_type = "anchor_b"). Use anchor_label "anchor_A" /
  "anchor_B" in the JSON if helpful.
- multi-session (total / how many times / combined sum across occurrences):
  for each tree, emit ONE plan with browse_type = "aggregate".
- preference (likes / dislikes / habits): for each tree, emit TWO plans —
  one direct (original question) and one preference-targeted
  ("user stated preferences about <topic>"), browse_type = "preference".
- simple: one direct plan per tree.

Cover every tree_id given in the input. Do NOT invent tree_ids.
"""


def _format_tree_cards(tree_cards: "list[TreeCard]") -> str:
    lines: list[str] = []
    for i, card in enumerate(tree_cards, 1):
        summary = (card.root_summary or "").strip().replace("\n", " ")
        if len(summary) > 220:
            summary = summary[:217] + "..."
        label = (card.label or "").strip()
        lines.append(
            f"{i}. tree_id={card.tree_id}  type={card.tree_type}  "
            f"label={label}  items={card.item_count}\n   summary: {summary}"
        )
    return "\n".join(lines)


class BrowsePlanner:
    """Query decomposition planner. One LLM call when enabled."""

    def __init__(
        self,
        *,
        config: "PlannerConfig",
        chat_client: "OpenAIChatClient | None" = None,
        default_model_name: str = "",
    ) -> None:
        self._config = config
        self._chat_client = chat_client
        self._model_name = config.model_name or default_model_name
        self._last_decomposition: DecompositionResult | None = None

    def plan(
        self,
        question: str,
        tree_cards: "list[TreeCard]",
    ) -> list[BrowsePlan]:
        """Generate browse plans for all recalled trees.

        Disabled: one direct plan per tree with sub_query = question.
        Enabled:  1 LLM call sees the question + all tree_cards at once and
                  emits per-tree targeted sub_queries (batch prompt).
        """
        if not self._config.enabled or self._chat_client is None:
            self._last_decomposition = DecompositionResult(
                needs_decomposition=False,
                question_type="simple",
                sub_queries=[question],
                explanation="planner disabled",
            )
            return self._direct_plans(question, tree_cards)
        if not tree_cards:
            self._last_decomposition = DecompositionResult(
                needs_decomposition=False,
                question_type="simple",
                sub_queries=[question],
                explanation="no tree cards",
            )
            return []
        return self._batch_plan(question, tree_cards)

    def update_exclusions(
        self,
        plans: list[BrowsePlan],
        already_found: list[str],
    ) -> list[BrowsePlan]:
        """For multi-session evolver: inject 'OTHER THAN X, Y, ...' into sub_query.

        Called between evolver rounds to avoid re-retrieving already-found instances.
        """
        if not already_found:
            return plans
        exclusion = ", ".join(already_found)
        suffix = f" (OTHER THAN: {exclusion})"
        updated = []
        for plan in plans:
            if plan.browse_type == BROWSE_TYPE_AGGREGATE:
                new_query = plan.sub_query.rstrip() + suffix
                updated.append(BrowsePlan(
                    tree_id=plan.tree_id,
                    sub_query=new_query,
                    browse_type=plan.browse_type,
                    anchor_label=plan.anchor_label,
                ))
            else:
                updated.append(plan)
        return updated

    # ── internal ──────────────────────────────────────────────────────────────

    def _direct_plans(
        self,
        question: str,
        tree_cards: "list[TreeCard]",
    ) -> list[BrowsePlan]:
        return [
            BrowsePlan(tree_id=card.tree_id, sub_query=question, browse_type=BROWSE_TYPE_DIRECT)
            for card in tree_cards
        ]

    _VALID_BROWSE_TYPES = {
        BROWSE_TYPE_DIRECT,
        BROWSE_TYPE_ANCHOR_A,
        BROWSE_TYPE_ANCHOR_B,
        BROWSE_TYPE_AGGREGATE,
        BROWSE_TYPE_PREFERENCE,
    }

    def _batch_plan(
        self,
        question: str,
        tree_cards: "list[TreeCard]",
    ) -> list[BrowsePlan]:
        """One LLM call: (question + all tree cards) → per-tree sub_queries."""
        assert self._chat_client is not None

        tree_card_block = _format_tree_cards(tree_cards)
        user_prompt = (
            f"Question: {question}\n\n"
            f"Candidate memory trees ({len(tree_cards)}):\n{tree_card_block}\n\n"
            "Emit one JSON object as specified. Cover every tree_id above."
        )

        try:
            result = self._chat_client.generate_json(
                system_prompt=_PLANNER_SYSTEM,
                user_prompt=user_prompt,
                model_name=self._model_name or None,
                step_label="browse_planner",
            )
        except Exception:
            self._last_decomposition = DecompositionResult(
                needs_decomposition=False,
                question_type="simple",
                sub_queries=[question],
                explanation="planner LLM call failed",
            )
            return self._direct_plans(question, tree_cards)

        qtype = str(result.get("question_type", "simple")).lower()
        if qtype not in ("temporal", "multi-session", "preference", "simple"):
            qtype = "simple"

        valid_tids = {c.tree_id for c in tree_cards}
        raw_plans = result.get("plans") or []
        if not isinstance(raw_plans, list):
            raw_plans = []

        plans: list[BrowsePlan] = []
        seen_tids: set[str] = set()
        distinct_sub_queries: list[str] = []
        for item in raw_plans:
            if not isinstance(item, dict):
                continue
            tid = str(item.get("tree_id") or "")
            if tid not in valid_tids:
                continue
            sq = str(item.get("sub_query") or "").strip() or question
            btype = str(item.get("browse_type") or BROWSE_TYPE_DIRECT).lower()
            if btype not in self._VALID_BROWSE_TYPES:
                btype = BROWSE_TYPE_DIRECT
            anchor = str(item.get("anchor_label") or "")
            plans.append(BrowsePlan(
                tree_id=tid,
                sub_query=sq,
                browse_type=btype,
                anchor_label=anchor,
            ))
            seen_tids.add(tid)
            if sq not in distinct_sub_queries:
                distinct_sub_queries.append(sq)

        # Fill missing trees with a direct pass-through so no tree gets dropped.
        for card in tree_cards:
            if card.tree_id not in seen_tids:
                plans.append(BrowsePlan(
                    tree_id=card.tree_id,
                    sub_query=question,
                    browse_type=BROWSE_TYPE_DIRECT,
                ))

        if not plans:
            self._last_decomposition = DecompositionResult(
                needs_decomposition=False,
                question_type=qtype,
                sub_queries=[question],
                explanation="planner returned no usable plans",
            )
            return self._direct_plans(question, tree_cards)

        self._last_decomposition = DecompositionResult(
            needs_decomposition=len(distinct_sub_queries) > 1
                                or any(p.browse_type != BROWSE_TYPE_DIRECT for p in plans),
            question_type=qtype,
            sub_queries=distinct_sub_queries or [question],
            explanation=str(result.get("explanation", "")),
        )
        return plans
