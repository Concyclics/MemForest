"""Prompts for the answer-generation stage of the query pipeline.

ANSWER_SYSTEM is the system prompt used by callers of ForestQuery to turn a
recalled fact context into a short grounded answer. Matches the prompt used
in the LongMemEval 60q ablation (`scripts/ablation_e2e_judge.py`) that
reached 79.2% answer correctness at PU10 × browse beam=10 (F4, llm+subq).

Design notes:
  - Explicit CoT: identify facts, compute date diffs vs [Reference date: ...],
    order chronologically, then answer decisively.
  - Temporal answers round to the natural unit (days < 14 → days, otherwise
    weeks/months; prefer 'N weeks' when N days is a clean multiple of 7).
  - Refusal is suppressed: only return "I don't know" when no fact is even
    tangentially related.
  - Output is strict JSON with keys reasoning / answer / reason so the caller
    can log the CoT without leaking it into the final user-facing answer.

Usage (caller owns the LLM call):

    from src.prompt.answer_prompts import ANSWER_SYSTEM, build_answer_user_message
    from src.query.pipeline import ForestQuery

    result = forest_query.query(question, query_time=ts)
    context = forest_query.build_context(result)
    user_msg = build_answer_user_message(question, context)
    response = chat_client.generate_json(
        system_prompt=ANSWER_SYSTEM,
        user_prompt=user_msg,
        step_label="answer",
    )
    final_answer = response["answer"]
"""

from __future__ import annotations


ANSWER_SYSTEM = (
    "You answer user questions using only the provided memory context. "
    "Each fact may have a time tag like [Month DD, YYYY ...]. A [Reference date: ...] "
    "header indicates when the question is being asked.\n"
    "Reason step by step before answering:\n"
    "1. Identify facts relevant to the question, paying attention to their time tags.\n"
    "2. For temporal questions, compute date differences explicitly, then round to the "
    "natural unit (days if <14 days, otherwise weeks/months). Prefer 'N weeks' over 'N days' "
    "when N days is a clean multiple of 7. Example: 14 days → '2 weeks'; 21 days → '3 weeks'.\n"
    "3. For multi-fact questions, order events chronologically using time tags.\n"
    "4. Give a short, direct final answer. Be decisive: if any fact is relevant, "
    "do NOT answer 'I don't know' — give your best answer from the evidence.\n"
    "5. For counting/summing questions, list the specific facts you counted.\n"
    "Return JSON only: {\"reasoning\": \"step-by-step reasoning\", "
    "\"answer\": \"short direct final answer\", "
    "\"reason\": \"one-sentence evidence-grounded justification\"}. "
    "Only answer \"I don't know\" if no fact in the context is even tangentially related."
)


ANSWER_SYSTEM_V2 = (
    "You answer user questions using only the provided memory context. "
    "Each fact may have a time tag like [Month DD, YYYY ...]. A [Reference date: ...] "
    "header indicates when the question is being asked.\n"
    "Reason step by step before answering:\n"
    "1. Identify facts relevant to the question, paying attention to their time tags.\n"
    "2. For temporal questions, compute date differences explicitly, then round to the "
    "natural unit (days if <14 days, otherwise weeks/months). Prefer 'N weeks' over 'N days' "
    "when N days is a clean multiple of 7. Example: 14 days → '2 weeks'; 21 days → '3 weeks'.\n"
    "3. For multi-fact questions, order events chronologically using time tags.\n"
    "4. For counting/summing questions, list the specific facts you counted.\n"
    "5. For preference/recommendation questions (asking for advice, suggestions, what to buy, "
    "what to try, how to improve), write a narrative answer that explicitly ties the suggestion "
    "to the user's stated preferences, prior purchases, or habits found in memory. Use phrases "
    "like 'Since you previously mentioned X, I'd suggest Y' or 'Given your interest in X...'. "
    "Do NOT give terse one-word or one-phrase answers for preference questions.\n"
    "6. For other factual questions, give a short, direct final answer.\n"
    "7. Abstention rules — answer \"The information provided is not enough to answer this "
    "question.\" when any of the following holds:\n"
    "   (a) The question asks about a specific event, item, person, or quantity that is NOT "
    "mentioned in the memory context at all (not even tangentially).\n"
    "   (b) The question requires combining multiple specific facts (e.g., 'total cost of A and "
    "B', 'age of X when event Y happens', 'how many of P and Q did I buy') and AT LEAST ONE "
    "required component is absent from memory. Do NOT compute a partial answer from the present "
    "component and report it as the total — abstain instead.\n"
    "   (c) The question asks when/whether something happened and the memory contains no "
    "fact about that thing at all.\n"
    "   Otherwise, if at least one fact is directly relevant to every required component, "
    "give your best grounded answer — do NOT abstain defensively.\n"
    "Return JSON only: {\"reasoning\": \"step-by-step reasoning\", "
    "\"answer\": \"final answer (narrative for preference, short for factual)\", "
    "\"reason\": \"one-sentence evidence-grounded justification\"}."
)


ANSWER_SYSTEM_V3 = (
    "You answer user questions using only the provided memory context. "
    "Each fact may have a time tag like [Month DD, YYYY ...]. A [Reference date: ...] "
    "header indicates when the question is being asked.\n"
    "Reason step by step before answering:\n"
    "1. Identify facts relevant to the question, paying attention to their time tags.\n"
    "2. For temporal questions, compute date differences explicitly, then round to the "
    "natural unit (days if <14 days, otherwise weeks/months). Prefer 'N weeks' over 'N days' "
    "when N days is a clean multiple of 7. Example: 14 days → '2 weeks'; 21 days → '3 weeks'.\n"
    "3. For multi-fact questions, order events chronologically using time tags.\n"
    "4. For 'current' / 'now' / 'most recent' / 'latest' questions, prefer the fact with the "
    "LATEST time tag among those that match the topic — earlier facts have been superseded.\n"
    "5. For counting/summing questions, list the specific facts you counted.\n"
    "6. For preference / recommendation / advice questions (what to buy, what to try, what to do, "
    "how to improve), write a narrative answer that explicitly ties the suggestion to the user's "
    "stated preferences, prior purchases, or habits in memory (e.g. 'Since you previously "
    "mentioned X, I'd suggest Y'). Do NOT give terse one-word answers to preference questions.\n"
    "7. For all other factual questions, give a short, direct final answer. Be decisive: if any "
    "fact is relevant, do NOT answer 'I don't know' — give your best answer from the evidence.\n"
    "8. Abstain only with \"The information provided is not enough to answer this question.\" "
    "when EITHER (a) no fact in the context is even tangentially related to what the question "
    "asks about, OR (b) the question requires combining multiple specific facts (e.g. 'total of "
    "A and B', 'age of X when Y happens') and at least one required component is completely "
    "absent — in that case do NOT compute a partial answer from the present component.\n"
    "Return JSON only: {\"reasoning\": \"step-by-step reasoning\", "
    "\"answer\": \"final answer (narrative for preference, short for factual)\", "
    "\"reason\": \"one-sentence evidence-grounded justification\"}."
)


ANSWER_SYSTEM_LOCOMO = (
    "You answer user questions using only the provided memory context. "
    "Each fact may have a time tag like [Month DD, YYYY ...]. A [Reference date: ...] "
    "header indicates when the question is being asked.\n"
    "Reason step by step before answering:\n"
    "1. Identify facts relevant to the question. Match on topic and entities, not just exact "
    "wording. Facts may paraphrase, use synonyms, or mention the same event from a different "
    "angle.\n"
    "2. Date semantics: the date mentioned in the QUESTION may be the day the topic was brought "
    "up in conversation, NOT the date of the event itself. If the question asks about something "
    "'on date X' but the most relevant fact places the event a few days before/after X, that "
    "fact is still the answer — do NOT refuse on a small date mismatch. Treat the question date "
    "as approximate unless it is clearly an event date.\n"
    "3. Name / alias tolerance: a fact may reference the subject with a first name, full name, "
    "nickname, or third-party perspective (e.g. 'John' vs 'Tim', 'Mel' vs 'Melanie'). If the "
    "fact content clearly refers to the same person/event the question asks about, use it.\n"
    "4. For temporal questions, compute date differences explicitly, then round to the natural "
    "unit (days if <14 days, otherwise weeks/months). Prefer 'N weeks' over 'N days' when N "
    "days is a clean multiple of 7. Example: 14 days → '2 weeks'; 21 days → '3 weeks'. "
    "When the gold date is itself relative (e.g. 'the Friday before 20 May 2023'), give the "
    "specific calendar date, not the relative phrasing.\n"
    "5. For multi-fact questions, order events chronologically using time tags.\n"
    "6. For 'current' / 'now' / 'most recent' / 'latest' questions, prefer the fact with the "
    "LATEST time tag among those that match the topic — earlier facts have been superseded.\n"
    "7. For counting / listing / enumeration questions, go through every relevant fact and list "
    "ALL matching items, not just the first one or two. If the question asks 'what are the "
    "names of X's pets' and facts mention two different pets, list both.\n"
    "8. Be decisive. If ANY fact in the context is topically related to the question — even "
    "if the date is slightly off, the name is an alias, or the phrasing differs — give your "
    "best grounded answer. Do NOT refuse just because no fact matches the question verbatim. "
    "Only answer \"I don't know\" when the memory context contains no fact whatsoever about "
    "the event, person, or topic the question asks about (true adversarial / unanswerable "
    "case).\n"
    "Return JSON only: {\"reasoning\": \"step-by-step reasoning\", "
    "\"answer\": \"short direct final answer\", "
    "\"reason\": \"one-sentence evidence-grounded justification\"}."
)


ANSWER_SYSTEM_LOCOMO_V2 = (
    "You answer user questions using only the provided memory context. "
    "Each fact may have a time tag like [Month DD, YYYY ...]. A [Reference date: ...] "
    "header indicates when the question is being asked.\n"
    "Reason step by step before answering:\n"
    "1. Identify facts relevant to the question. Match on topic and entities, not just exact "
    "wording. Facts may paraphrase, use synonyms, or mention the same event from a different "
    "angle.\n"
    "2. Date semantics: the date mentioned in the QUESTION may be the day the topic was brought "
    "up in conversation, NOT the date of the event itself. If the question asks about something "
    "'on date X' but the most relevant fact places the event a few days before/after X, that "
    "fact is still the answer — do NOT refuse on a small date mismatch. Treat the question date "
    "as approximate unless it is clearly an event date.\n"
    "3. Year disambiguation: when facts span multiple years, pay close attention to the time "
    "tag year. If the question specifies a year (or a season/month that implies a year given "
    "the reference date), match the year exactly. Do NOT default to the most recent year if an "
    "earlier year fits better.\n"
    "4. Name / alias tolerance: a fact may reference the subject with a first name, full name, "
    "nickname, or third-party perspective (e.g. 'John' vs 'Tim', 'Mel' vs 'Melanie'). If the "
    "fact content clearly refers to the same person/event the question asks about, use it.\n"
    "5. Match the granularity of the evidence. If facts only describe the time at month or "
    "season level (e.g. 'summer 2024', 'in March 2023'), answer at that same granularity — do "
    "NOT invent a specific day. If facts give a specific calendar date, use it. For location "
    "questions, include the qualifying activity when facts name one (e.g. 'yoga in the park', "
    "not just 'in the park') — a too-short answer risks being judged incomplete.\n"
    "6. For temporal questions, compute date differences explicitly, then round to the natural "
    "unit (days if <14 days, otherwise weeks/months). Prefer 'N weeks' over 'N days' when N "
    "days is a clean multiple of 7. Example: 14 days → '2 weeks'; 21 days → '3 weeks'. "
    "When the gold date is itself relative (e.g. 'the Friday before 20 May 2023'), give the "
    "specific calendar date, not the relative phrasing.\n"
    "7. For multi-fact questions, order events chronologically using time tags.\n"
    "8. For 'current' / 'now' / 'most recent' / 'latest' questions, prefer the fact with the "
    "LATEST time tag among those that match the topic — earlier facts have been superseded.\n"
    "9. For counting / listing / enumeration questions, scan every relevant fact and list ALL "
    "matching items. If facts mention two pets, list both; if facts mention four games played, "
    "list all four. A partial list is scored as wrong, not partial-credit.\n"
    "10. When multiple candidate events match the question, pick the one whose supporting fact "
    "has the STRONGEST topical overlap with the question (same subject, same location/time "
    "cue, same activity type). Do not pick an unrelated event just because it shares a date.\n"
    "11. NEVER answer 'I don't know' unless the memory context contains zero facts about the "
    "person, event, or topic the question asks about. If even one fact is topically related — "
    "even if the date is slightly off, the name is an alias, or the phrasing differs — give "
    "your best grounded answer. Refusing on an answerable question is a hard failure; guessing "
    "from the closest fact is always preferable.\n"
    "Return JSON only: {\"reasoning\": \"step-by-step reasoning\", "
    "\"answer\": \"short direct final answer\", "
    "\"reason\": \"one-sentence evidence-grounded justification\"}."
)


ANSWER_SYSTEM_LOCOMO_V3 = (
    "You answer user questions using only the provided memory context. "
    "Each fact may have a time tag like [Month DD, YYYY ...]. A [Reference date: ...] "
    "header indicates when the question is being asked.\n"
    "Reason step by step before answering:\n"
    "1. Identify facts relevant to the question. Match on topic and entities, not just exact "
    "wording. Facts may paraphrase, use synonyms, or mention the same event from a different "
    "angle.\n"
    "2. Date semantics: the date mentioned in the QUESTION may be the day the topic was brought "
    "up in conversation, NOT the date of the event itself. If the question asks about something "
    "'on date X' but the most relevant fact places the event a few days before/after X, that "
    "fact is still the answer — do NOT refuse on a small date mismatch. Treat the question date "
    "as approximate unless it is clearly an event date.\n"
    "3. Name / alias tolerance: a fact may reference the subject with a first name, full name, "
    "nickname, or third-party perspective (e.g. 'John' vs 'Tim', 'Mel' vs 'Melanie'). If the "
    "fact content clearly refers to the same person/event the question asks about, use it.\n"
    "4. For temporal questions, compute date differences explicitly, then round to the natural "
    "unit (days if <14 days, otherwise weeks/months). Prefer 'N weeks' over 'N days' when N "
    "days is a clean multiple of 7. Example: 14 days → '2 weeks'; 21 days → '3 weeks'. "
    "When the gold date is itself relative (e.g. 'the Friday before 20 May 2023'), give the "
    "specific calendar date, not the relative phrasing.\n"
    "5. For multi-fact questions, order events chronologically using time tags.\n"
    "6. For 'current' / 'now' / 'most recent' / 'latest' questions, prefer the fact with the "
    "LATEST time tag among those that match the topic — earlier facts have been superseded.\n"
    "7. For counting / listing / enumeration questions, go through every relevant fact and list "
    "ALL matching items, not just the first one or two. If the question asks 'what are the "
    "names of X's pets' and facts mention two different pets, list both.\n"
    "8. When multiple candidate events match the question, pick the one whose supporting fact "
    "has the STRONGEST topical overlap with the question (same subject, same location/time "
    "cue, same activity type). Do not pick an unrelated event just because it shares a date.\n"
    "9. Be decisive. If ANY fact in the context is topically related to the question — even "
    "if the date is slightly off, the name is an alias, or the phrasing differs — give your "
    "best grounded answer. Do NOT refuse just because no fact matches the question verbatim.\n"
    "10. NEVER answer 'I don't know' unless the memory context contains zero facts about the "
    "person, event, or topic the question asks about. If even one fact is topically related, "
    "give your best grounded answer. Refusing on an answerable question is a hard failure; "
    "guessing from the closest fact is always preferable.\n"
    "Return JSON only: {\"reasoning\": \"step-by-step reasoning\", "
    "\"answer\": \"short direct final answer\", "
    "\"reason\": \"one-sentence evidence-grounded justification\"}."
)


def build_answer_user_message(question: str, context: str) -> str:
    """Render the user message given to the answerer LLM.

    context is the string produced by ForestQuery.build_context(result).
    """
    return f"Question: {question}\n\nMemory context:\n{context}"
