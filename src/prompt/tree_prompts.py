"""LLM prompt templates for tree node summary generation."""

from __future__ import annotations

from src.prompt.prompt import PromptManager, PromptTemplate


TREE_SUMMARY_SESSION = PromptTemplate(
    name="tree_summary_session",
    system="""You are writing a STRUCTURAL NAVIGATION SUMMARY for a session tree \
node in a memory index. This summary must serve as both an answer source AND a \
navigation guide for downstream tree search.

Write one concise prose summary ({max_words} words or fewer).

Rules:
1. Lead with the most answer-bearing facts: who, what, when, where, how many.
2. Preserve ALL exact numbers, dates, prices, quantities, and named outcomes.
3. Use "X → Y" notation for state changes (e.g. "salary $80k → $95k").
4. Chronological order. End with latest known state.
5. Major clusters only — no exhaustive enumeration of minor details.
6. No assistant advice, interpretation, or filler phrases.

- OUTPUT FORMAT: respond with exactly one JSON key:
  {{"summary": "your prose here"}}
  Do NOT add any other keys.""",
    user="""Conversation window: {session_id} ({time_range_text}):
{input_text}""",
)

TREE_SUMMARY_ENTITY_USER = PromptTemplate(
    name="tree_summary_entity_user",
    system="""You are writing a STRUCTURAL NAVIGATION SUMMARY for the user entity \
tree. This summary must serve as both an answer source AND a navigation guide.

Write one concise prose summary ({max_words} words or fewer).

Rules:
1. Lead with answer-bearing user facts: role, preferences, states, constraints.
2. Preserve ALL exact numbers, dates, prices, amounts, and named outcomes.
3. Use "X → Y" notation for state changes (e.g. "weight 180lb → 165lb").
4. Group by domain cluster, then chronological within each cluster.
5. End with latest known user state, role, or preference.
6. No personality analysis, assistant advice, or interpretation.
7. Do NOT generalize away specific entities or value changes.

- OUTPUT FORMAT: respond with exactly one JSON key:
  {{"summary": "your prose here"}}
  Do NOT add any other keys.""",
    user="""Facts ({time_range_text}):
{input_text}""",
)

TREE_SUMMARY_ENTITY_GENERIC = PromptTemplate(
    name="tree_summary_entity_generic",
    system="""You are writing a STRUCTURAL NAVIGATION SUMMARY for one named entity \
tree. This summary must serve as both an answer source AND a navigation guide.

Write one concise prose summary ({max_words} words or fewer).

Rules:
1. First sentence: what this entity is (person, place, organization, product).
2. Then major interactions, events, values, or decisions in chronological order.
3. Preserve ALL exact names, prices, quantities, dates, and time anchors.
4. Use "X → Y" notation for state changes.
5. Stay entity-focal — do not drift into unrelated narrative.
6. End with latest known entity state or outcome.
7. No recommendations or interpretation.

- OUTPUT FORMAT: respond with exactly one JSON key:
  {{"summary": "your prose here"}}
  Do NOT add any other keys.""",
    user="""Entity: {tree_label} ({time_range_text}):
{input_text}""",
)

TREE_SUMMARY_SCENE = PromptTemplate(
    name="tree_summary_scene",
    system="""You are writing a STRUCTURAL NAVIGATION SUMMARY for one scene tree. \
This summary must serve as both an answer source AND a navigation guide.

Write one concise prose summary ({max_words} words or fewer).

Rules:
1. Lead with the most answer-bearing scene facts.
2. Major event clusters in chronological order.
3. Preserve ALL exact dates, quantities, values, prices, and named outcomes.
4. Use "X → Y" notation for state changes.
5. Mention key recurring actors or entities.
6. End with latest known scene outcome or state.
7. No recommendations, exposition, or generalization beyond the provided facts.

- OUTPUT FORMAT: respond with exactly one JSON key:
  {{"summary": "your prose here"}}
  Do NOT add any other keys.""",
    user="""Domain: {tree_label} ({time_range_text}):
{input_text}""",
)


def build_tree_prompt_manager() -> PromptManager:
    """Return a PromptManager loaded with all tree summary templates."""
    return PromptManager(
        templates=[
            TREE_SUMMARY_SESSION,
            TREE_SUMMARY_ENTITY_USER,
            TREE_SUMMARY_ENTITY_GENERIC,
            TREE_SUMMARY_SCENE,
        ]
    )
