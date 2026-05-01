"""Prompt template for LLM-backed fact equivalence checks."""

from __future__ import annotations

from src.prompt.prompt import PromptManager, PromptTemplate


FACT_EQUIVALENCE_PROMPT_NAME = "fact_equivalence"


FACT_EQUIVALENCE_PROMPT = PromptTemplate(
    name=FACT_EQUIVALENCE_PROMPT_NAME,
    system="""You compare two extracted memory facts and decide whether they describe the exact same fact.

Return JSON only with this schema:
{{
  "equivalent": true,
  "preferred": "a|b|either",
  "reason": "short explanation"
}}

Rules:
- Mark equivalent=true only if the two facts can safely collapse into one memory item.
- Facts are NOT equivalent if they refer to different dates, different occurrences, different entities, different quantities, or different roles.
- Facts are NOT equivalent if one is a broader summary and the other adds a materially new detail that should remain separately retrievable.
- Prefer the fact that is more precise, more specific, and more directly grounded.
- If both facts are equally good, return preferred="either".
""",
    user="""Fact A:
{fact_a}

Fact B:
{fact_b}""",
)


def build_dedup_prompt_manager() -> PromptManager:
    return PromptManager(templates=[FACT_EQUIVALENCE_PROMPT])
