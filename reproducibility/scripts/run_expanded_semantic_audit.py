#!/usr/bin/env python3
"""Run reproducible, model-assisted audits for revision annotation gates.

This script never labels its output as human gold. It creates independent
review records that authors can inspect and sign off before manuscript use.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx


MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"No JSON object in response: {text[:300]}")
    return json.loads(text[start : end + 1])


def compact_facts(raw: str) -> list[dict[str, str]]:
    facts = json.loads(raw)
    return [
        {
            "fact_id": str(f.get("fact_id", "")),
            "fact_text": str(f.get("fact_text", "")),
            "time_text": str(f.get("time_text", "")),
        }
        for f in facts
    ]


def temporal_prompt(row: dict[str, str]) -> str:
    payload = {
        "question": row["question"],
        "gold_answer": row["gold_answer"],
        "reference_date": row["reference_date"],
        "candidate_facts": compact_facts(row["candidate_facts_json"]),
        "proposed_temporal_type": row["proposed_temporal_type"],
        "proposed_gold_fact_ids": row["proposed_gold_fact_ids"].split(";"),
        "proposed_conflict_fact_ids": row["proposed_conflict_fact_ids"].split(";"),
    }
    return f"""You are independently auditing temporal evidence labels for a
paper diagnostic. The diagnostic intentionally uses only three coarse buckets;
choose the closest bucket rather than inventing a fourth category.

Definitions:
- gold facts directly support the benchmark gold answer or the temporal
  operation needed to derive it;
- conflict facts describe the same target event/state/entity but can lead to a
  different date, order, or current-versus-old answer;
- irrelevant topical facts are neither gold nor conflict;
- historical-state covers a direct lookup of what/where/when was true or
  planned at a specified time. It is also the fallback bucket for a single
  dated event/state that does not require comparison or arithmetic;
- current-vs-old explicitly contrasts a latest/current/last state with an older
  state or asks for the state immediately before/after a change;
- ordering requires relative order or sequence among two or more events;
- duration requires interval arithmetic between two dates/events or resolving
  an elapsed-time expression;
- temporal-other covers temporal questions that do not fit the four buckets.

Check every proposed ID against the fact text. Canonical fact IDs and source
item IDs may be duplicate aliases of the same logical fact; retaining such
aliases is valid and must not by itself invalidate a proposal. A conflict must
state the same target event/state with an incompatible answer; do not classify
all nearby or topically related facts as conflicts. Keep at most eight logical
gold facts and eight conflicts. Before returning, verify that notes agree with
temporal_type and the two ID lists. Return JSON only with keys:
temporal_type (historical-state/current-vs-old/ordering/duration/temporal-other), gold_fact_ids,
conflict_fact_ids, temporal_type_valid (true/false), gold_ids_valid
(true/false), conflict_ids_valid (true/false), proposed_label_valid
(true only when all three preceding fields are true), confidence
(high/medium/low), notes. Keep notes below 120 words and do not repeat the
candidate fact list.

INPUT:
{json.dumps(payload, ensure_ascii=False)}"""


def entity_prompt(row: dict[str, str]) -> str:
    payload = {k: row[k] for k in (
        "fact_text", "extracted_entities", "normalized_entity_keys",
        "active_entity_keys", "candidate_states", "audit_stratum"
    )}
    return f"""You are auditing semantic entity routing for one memory fact.

An expected entity is a specific, reusable person, place, organization,
named product, titled work, or named event that is explicitly supported by the
fact and would form a useful long-term memory scope. Return at most three.
Ordinary features, ingredients, attributes, prices, standalone numbers, and
dates are not expected entity scopes unless the date names a specific event.

Judge active_entity_keys only. Precision and recall are independent:
- precision PASS if every nonempty active key is supported and useful; PARTIAL
  if some are overly generic/spurious; FAIL if most are unsupported. Never
  lower precision merely because an expected entity is missing. If there are
  no active keys, precision is NOT_APPLICABLE.
- recall PASS if active keys cover all salient expected entities; PARTIAL if
  at least one salient entity is covered but another is missing; FAIL if no
  salient expected entity is active. If no entity tree is semantically needed,
  recall is NOT_APPLICABLE.

Return JSON only with keys: expected_entities (list), precision_verdict
(PASS/PARTIAL/FAIL), recall_verdict (PASS/PARTIAL/FAIL/NOT_APPLICABLE),
confidence (high/medium/low), notes.

INPUT:
{json.dumps(payload, ensure_ascii=False)}"""


def judge_prompt(row: dict[str, str]) -> str:
    payload = {k: row[k] for k in (
        "qid", "cutoff", "question_type", "question", "gold_answer",
        "generated_answer", "strict_label", "public_label"
    )}
    return f"""You are adjudicating a disagreement between two answer judges.
Decide correctness against the question and reference answer. Require the
generated answer to provide the requested information, not merely repeat the
entity or question topic. Accept semantically equivalent wording, benchmark-
specified alternatives, and harmless extra detail. Reject missing required
list items, wrong attributes/dates/numbers, unsupported contradictions, and
vague category labels that do not answer the request.

Return JSON only with keys: adjudicated_label (CORRECT/WRONG),
preferred_judge (strict/public/neither), error_type, confidence
(high/medium/low), notes.

INPUT:
{json.dumps(payload, ensure_ascii=False)}"""


async def request_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    base_url: str,
    prompt: str,
) -> dict[str, Any]:
    async with semaphore:
        response = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 2600,
            },
        )
        response.raise_for_status()
        body = response.json()
        text = body["choices"][0]["message"]["content"]
        return {"audit": parse_json(text), "usage": body.get("usage", {})}


async def run_rows(
    rows: list[dict[str, str]],
    prompt_fn: Any,
    key_fn: Any,
    base_url: str,
    concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(300.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            request_one(client, semaphore, base_url, prompt_fn(row))
            for row in rows
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    output: list[dict[str, Any]] = []
    for row, result in zip(rows, results):
        record: dict[str, Any] = {"key": key_fn(row)}
        if isinstance(result, Exception):
            record.update({"error": repr(result), "audit": None})
        else:
            record.update(result)
        output.append(record)
    return output


def load_judge_disagreements(path: Path) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in read_csv(path):
        grouped[(row["qid"], row["cutoff"])][row["judge_arm"]] = row
    output = []
    for (qid, cutoff), arms in sorted(grouped.items()):
        if set(arms) != {"strict", "public"}:
            continue
        strict, public = arms["strict"], arms["public"]
        if strict["majority_label"] == public["majority_label"]:
            continue
        output.append({
            "qid": qid,
            "cutoff": cutoff,
            "question_type": strict["question_type"],
            "question": strict["question"],
            "gold_answer": strict["gold_answer"],
            "generated_answer": strict["generated_answer"],
            "strict_label": strict["majority_label"],
            "public_label": public["majority_label"],
        })
    return output


async def main_async(args: argparse.Namespace) -> None:
    temporal = read_csv(args.temporal_csv)
    entity = read_csv(args.entity_csv)
    judge = (
        read_csv(args.judge_review_csv)
        if args.judge_review_csv is not None
        else load_judge_disagreements(args.judge_csv)
    )
    for name, rows in (("temporal", temporal), ("entity", entity), ("judge", judge)):
        if not rows:
            raise ValueError(f"{name} audit source is empty")

    jobs = [
        ("temporal", temporal, temporal_prompt, lambda r: f"{r['benchmark']}::{r['qid']}"),
        ("entity", entity, entity_prompt, lambda r: f"{r['model']}::{r['qid']}::{r['fact_id']}"),
        ("judge", judge, judge_prompt, lambda r: f"{r['qid']}::k{r['cutoff']}"),
    ]
    for name, rows, prompt_fn, key_fn in jobs:
        if name not in args.tasks:
            continue
        result = await run_rows(
            rows, prompt_fn, key_fn, args.base_url, args.concurrency
        )
        suffix = f"_{len(rows)}" if args.size_suffix else ""
        write_jsonl(
            args.output_dir / f"{name}_independent_audit{suffix}.jsonl",
            result,
        )
        errors = sum(bool(row.get("error")) for row in result)
        print(f"{name}: rows={len(result)} errors={errors}")


def main() -> None:
    repro_root = Path(__file__).resolve().parents[1]
    expanded_root = repro_root / "results/semantic_audit/expanded"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temporal-csv", type=Path,
        default=expanded_root / "source/temporal_conflict_author_review_249.csv"
    )
    parser.add_argument(
        "--entity-csv", type=Path,
        default=expanded_root / "source/entity_routing_author_review_300.csv"
    )
    parser.add_argument(
        "--judge-csv", type=Path,
        default=repro_root / "results/mem0_corrected"
        / "top50_top200_control/per_question_majority.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=expanded_root / "raw"
    )
    parser.add_argument(
        "--judge-review-csv", type=Path,
        default=expanded_root / "source/judge_calibration_author_review_120.csv"
    )
    parser.add_argument(
        "--size-suffix", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument(
        "--tasks", nargs="+", choices=("temporal", "entity", "judge"),
        default=("temporal", "entity", "judge")
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
