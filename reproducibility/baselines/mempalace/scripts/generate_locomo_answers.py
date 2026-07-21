#!/usr/bin/env python3
"""Generate Gemma answers for MemPalace LoCoMo retrieval outputs.

Input is the `results.json` emitted by `mempalace/benchmarks/locomo_bench.py`.
The file contains each question and its retrieved session ids. This script
reconstructs the retrieved LoCoMo session transcripts, asks a local
OpenAI-compatible chat model for an answer, and writes one JSON object per
question.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI


ANSWER_SYSTEM = (
    "You answer questions using only the provided memory context. "
    "The context contains dated conversation sessions retrieved by a memory system.\n"
    "Rules:\n"
    "1. Match on the actual people, events, and dates in the context, not only exact wording.\n"
    "2. If the question asks for a date, person, place, relationship, count, or list, give a short direct answer.\n"
    "3. For temporal or comparison questions, reason from the dated sessions and answer decisively.\n"
    "4. For adversarial/unanswerable questions, answer \"I don't know\" only when the context contains no relevant evidence.\n"
    "Return JSON only with keys: reasoning, answer, reason."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locomo-data", required=True, type=Path)
    parser.add_argument("--retrieval-results", required=True, type=Path)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-context-chars", type=int, default=60000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_locomo(path: Path) -> dict[str, dict[str, Any]]:
    samples = json.loads(path.read_text(encoding="utf-8"))
    return {str(sample.get("sample_id")): sample for sample in samples}


def attach_qids(rows: list[dict[str, Any]], samples: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    counters: dict[str, int] = {}
    enriched = []
    for row in rows:
        sample_id = str(row["sample_id"])
        idx = counters.get(sample_id, 0)
        qas = samples[sample_id].get("qa", [])
        while idx < len(qas) and str(qas[idx].get("question")) != str(row.get("question")):
            idx += 1
        if idx >= len(qas):
            idx = counters.get(sample_id, 0)
        counters[sample_id] = idx + 1
        item = dict(row)
        item["question_id"] = f"{sample_id}_qa{idx:03d}"
        item["locomo_qa_idx"] = idx
        enriched.append(item)
    return enriched


def session_sort_key(session_id: str) -> int:
    match = re.search(r"(\d+)$", session_id)
    return int(match.group(1)) if match else 0


def render_session(sample: dict[str, Any], session_id: str) -> str:
    conv = sample["conversation"]
    sess_num = session_sort_key(session_id)
    key = f"session_{sess_num}"
    date = conv.get(f"{key}_date_time", "")
    turns = conv.get(key, [])
    lines = [f"[{session_id} | {date}]"]
    for turn in turns:
        speaker = turn.get("speaker", "?")
        dia_id = turn.get("dia_id", "")
        text = str(turn.get("text", "")).replace("\n", " ").strip()
        if dia_id:
            lines.append(f"{dia_id} {speaker}: {text}")
        else:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def build_context(sample: dict[str, Any], retrieved_ids: list[str], max_chars: int) -> str:
    parts = []
    for sid in sorted(retrieved_ids, key=session_sort_key):
        parts.append(render_session(sample, sid))
    context = "\n\n".join(parts)
    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n[Context truncated due to length.]"
    return context


def build_user_prompt(row: dict[str, Any], sample: dict[str, Any], context: str) -> str:
    conv = sample["conversation"]
    reference_date = ""
    for i in range(1, 200):
        val = conv.get(f"session_{i}_date_time")
        if val:
            reference_date = val
    return (
        f"Reference date: {reference_date}\n"
        f"Question: {row['question']}\n\n"
        f"Retrieved memory context:\n{context}\n\n"
        "Answer the question using only the retrieved memory context."
    )


def extract_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {"answer": str(obj)}
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else {"answer": str(obj)}
        except Exception:
            pass
    return {"reasoning": "", "answer": raw, "reason": "Model did not return parseable JSON."}


def call_answer(
    client: OpenAI,
    *,
    model: str,
    row: dict[str, Any],
    sample: dict[str, Any],
    max_context_chars: int,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    context = build_context(sample, row.get("retrieved_ids", []), max_context_chars)
    user_prompt = build_user_prompt(row, sample, context)
    started = time.time()
    last_error = None
    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": ANSWER_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=120,
            )
            content = response.choices[0].message.content or ""
            parsed = extract_json(content)
            return {
                "question_id": row["question_id"],
                "sample_id": row["sample_id"],
                "locomo_qa_idx": row["locomo_qa_idx"],
                "question": row["question"],
                "gold_answer": row.get("answer", ""),
                "category": row.get("category"),
                "evidence": row.get("evidence", []),
                "retrieved_ids": row.get("retrieved_ids", []),
                "retrieval_recall": row.get("recall"),
                "model_answer": str(parsed.get("answer", "")).strip(),
                "model_reason": str(parsed.get("reason", "")).strip(),
                "model_reasoning": str(parsed.get("reasoning", "")).strip(),
                "raw_response": content,
                "context_chars": len(context),
                "answer_seconds": round(time.time() - started, 3),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            time.sleep(min(2 * attempt, 8))
    return {
        "question_id": row["question_id"],
        "sample_id": row["sample_id"],
        "locomo_qa_idx": row["locomo_qa_idx"],
        "question": row["question"],
        "gold_answer": row.get("answer", ""),
        "category": row.get("category"),
        "evidence": row.get("evidence", []),
        "retrieved_ids": row.get("retrieved_ids", []),
        "retrieval_recall": row.get("recall"),
        "model_answer": "",
        "model_reason": "",
        "model_reasoning": "",
        "raw_response": "",
        "context_chars": 0,
        "answer_seconds": round(time.time() - started, 3),
        "error": last_error,
    }


def main() -> None:
    args = parse_args()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    samples = load_locomo(args.locomo_data)
    rows = attach_qids(json.loads(args.retrieval_results.read_text(encoding="utf-8")), samples)
    if args.limit > 0:
        rows = rows[: args.limit]

    done: dict[str, dict[str, Any]] = {}
    if args.resume and args.out_jsonl.exists():
        with args.out_jsonl.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("question_id") and obj.get("error") is None:
                    done[str(obj["question_id"])] = obj

    todo = [row for row in rows if row["question_id"] not in done]
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    print(f"rows={len(rows)} done={len(done)} todo={len(todo)} workers={args.workers}", flush=True)
    started = time.time()
    completed = 0

    with args.out_jsonl.open("a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    call_answer,
                    client,
                    model=args.model,
                    row=row,
                    sample=samples[str(row["sample_id"])],
                    max_context_chars=args.max_context_chars,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                ): row
                for row in todo
            }
            for fut in as_completed(futures):
                result = fut.result()
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                completed += 1
                if completed % 20 == 0 or completed == len(todo):
                    elapsed = time.time() - started
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    remaining = (len(todo) - completed) / rate if rate > 0 else 0.0
                    print(
                        f"completed {completed}/{len(todo)} "
                        f"rate={rate:.2f}/s eta={remaining/60:.1f}m",
                        flush=True,
                    )

    all_rows = []
    with args.out_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                all_rows.append(json.loads(line))
    errors = sum(1 for row in all_rows if row.get("error"))
    empty = sum(1 for row in all_rows if not str(row.get("model_answer", "")).strip())
    by_cat: dict[str, int] = {}
    for row in all_rows:
        cat = str(row.get("category"))
        by_cat[cat] = by_cat.get(cat, 0) + 1
    summary = {
        "total_rows": len(all_rows),
        "errors": errors,
        "empty_answers": empty,
        "by_category": by_cat,
        "out_jsonl": str(args.out_jsonl),
        "retrieval_results": str(args.retrieval_results),
        "model": args.model,
        "base_url": args.base_url,
    }
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
