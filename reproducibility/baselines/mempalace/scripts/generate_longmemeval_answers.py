#!/usr/bin/env python3
"""Generate Gemma answers for MemPalace LongMemEval retrieval outputs."""

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
    "You answer user questions using only the provided retrieved memory context. "
    "Each memory item may include a timestamp. A reference date tells you when the "
    "question is being asked.\n"
    "Rules:\n"
    "1. Match the question to relevant memory by topic, entities, and time.\n"
    "2. For temporal questions, compute date differences from the memory dates and the reference date.\n"
    "3. For latest/current questions, prefer the most recent relevant memory.\n"
    "4. For counting or list questions, use all relevant retrieved items.\n"
    "5. If the retrieved context has no relevant evidence, answer \"I don't know\".\n"
    "Return JSON only with keys: reasoning, answer, reason."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-jsonl", required=True, type=Path)
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=None,
        help="Original LongMemEval JSON file. When provided, full retrieved sessions and question_date are loaded from it.",
    )
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-context-chars", type=int, default=60000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_dataset_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(row.get("question_id")): row for row in data if row.get("question_id")}


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


def _session_id(corpus_id: str) -> str:
    if "_turn_" in corpus_id:
        return corpus_id.rsplit("_turn_", 1)[0]
    return corpus_id


def _turn_text(turn: dict[str, Any]) -> str:
    role = str(turn.get("role", "unknown")).strip() or "unknown"
    content = str(turn.get("content", "")).replace("\n", " ").strip()
    return f"{role}: {content}"


def build_context_from_dataset(
    dataset_row: dict[str, Any],
    retrieved_ids: list[str],
    *,
    max_chars: int,
) -> str:
    session_ids = [str(sid) for sid in dataset_row.get("haystack_session_ids", [])]
    dates = [str(date) for date in dataset_row.get("haystack_dates", [])]
    sessions = dataset_row.get("haystack_sessions", [])
    by_id = {
        sid: (dates[idx] if idx < len(dates) else "", sessions[idx])
        for idx, sid in enumerate(session_ids)
        if idx < len(sessions)
    }

    blocks = []
    seen = set()
    for rank, corpus_id in enumerate(retrieved_ids, 1):
        sid = _session_id(str(corpus_id))
        if sid in seen or sid not in by_id:
            continue
        seen.add(sid)
        date, turns = by_id[sid]
        text = "\n".join(_turn_text(turn) for turn in turns)
        blocks.append(f"{rank}. [{sid} | {date}]\n{text}")

    context = "\n\n".join(blocks)
    if len(context) > max_chars:
        context = context[:max_chars] + "\n[Context truncated due to length.]"
    return context


def build_context(
    row: dict[str, Any],
    top_k: int,
    max_chars: int,
    dataset_row: dict[str, Any] | None = None,
) -> str:
    ranked = ((row.get("retrieval_results") or {}).get("ranked_items") or [])[:top_k]
    retrieved_ids = [str(item.get("corpus_id", "")) for item in ranked if item.get("corpus_id")]
    if dataset_row and retrieved_ids:
        context = build_context_from_dataset(dataset_row, retrieved_ids, max_chars=max_chars)
        if context.strip():
            return context

    lines = []
    for i, item in enumerate(ranked, 1):
        cid = item.get("corpus_id", "")
        ts = item.get("timestamp", "")
        text = str(item.get("text", "")).replace("\n", " ").strip()
        lines.append(f"{i}. [{cid} | {ts}] {text}")
    context = "\n".join(lines)
    if len(context) > max_chars:
        context = context[:max_chars] + "\n[Context truncated due to length.]"
    return context


def call_answer(
    client: OpenAI,
    *,
    row: dict[str, Any],
    dataset_row: dict[str, Any] | None,
    model: str,
    top_k: int,
    max_context_chars: int,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    context = build_context(row, top_k, max_context_chars, dataset_row)
    question_date = (
        (dataset_row or {}).get("question_date")
        or row.get("question_date")
        or "unknown"
    )
    gold_answer = (dataset_row or {}).get("answer") or row.get("answer", "")
    user_prompt = (
        f"Reference date: {question_date}\n"
        f"Question: {row.get('question', '')}\n\n"
        f"Retrieved memory context:\n{context}\n\n"
        "Answer using only the retrieved memory context."
    )
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
                "question_id": row.get("question_id"),
                "question_type": row.get("question_type"),
                "question": row.get("question"),
                "question_date": question_date,
                "gold_answer": gold_answer,
                "retrieved_ids": [
                    item.get("corpus_id")
                    for item in ((row.get("retrieval_results") or {}).get("ranked_items") or [])[:top_k]
                ],
                "retrieval_metrics": (row.get("retrieval_results") or {}).get("metrics", {}),
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
        "question_id": row.get("question_id"),
        "question_type": row.get("question_type"),
        "question": row.get("question"),
        "question_date": question_date,
        "gold_answer": gold_answer,
        "retrieved_ids": [],
        "retrieval_metrics": (row.get("retrieval_results") or {}).get("metrics", {}),
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

    rows = load_rows(args.retrieval_jsonl)
    dataset_map = load_dataset_map(args.dataset_json)
    if args.limit > 0:
        rows = rows[: args.limit]

    done = {}
    if args.resume and args.out_jsonl.exists():
        with args.out_jsonl.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("question_id") and obj.get("error") is None:
                    done[str(obj["question_id"])] = obj
    todo = [row for row in rows if str(row.get("question_id")) not in done]
    print(f"rows={len(rows)} done={len(done)} todo={len(todo)} workers={args.workers}", flush=True)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    started = time.time()
    completed = 0
    with args.out_jsonl.open("a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    call_answer,
                    client,
                    row=row,
                    dataset_row=dataset_map.get(str(row.get("question_id"))),
                    model=args.model,
                    top_k=args.top_k,
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
    by_type: dict[str, int] = {}
    for row in all_rows:
        qtype = str(row.get("question_type"))
        by_type[qtype] = by_type.get(qtype, 0) + 1
    summary = {
        "total_rows": len(all_rows),
        "errors": errors,
        "empty_answers": empty,
        "by_type": by_type,
        "top_k": args.top_k,
        "out_jsonl": str(args.out_jsonl),
        "retrieval_jsonl": str(args.retrieval_jsonl),
        "dataset_json": str(args.dataset_json) if args.dataset_json else None,
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
