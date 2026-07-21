#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
MEMORYOS_ROOT = Path(os.environ.get("MEMORYOS_ROOT", Path.cwd())).resolve()
MEMORYOS_PYPI = MEMORYOS_ROOT / "memoryos-pypi"
sys.path.insert(0, str(MEMORYOS_PYPI))
sys.path.insert(0, str(SCRIPT_DIR))

from memoryos import Memoryos  # noqa: E402
from common import get_anscheck_prompt, true_or_false  # noqa: E402


def parse_lme_date(value: str | None) -> datetime:
    if not value:
        return datetime(2024, 1, 1)
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return datetime(2024, 1, 1)


def fmt_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def iter_memory_pairs(item: dict[str, Any]) -> list[dict[str, Any]]:
    memories: list[dict[str, Any]] = []
    sessions = item.get("haystack_sessions") or []
    session_ids = item.get("haystack_session_ids") or []
    dates = item.get("haystack_dates") or []

    for sidx, session in enumerate(sessions):
        session_id = str(session_ids[sidx]) if sidx < len(session_ids) else f"session_{sidx}"
        base_dt = parse_lme_date(str(dates[sidx]) if sidx < len(dates) else None)
        turns = session if isinstance(session, list) else []
        i = 0
        while i < len(turns):
            turn = turns[i] if isinstance(turns[i], dict) else {}
            role = str(turn.get("role", "")).lower()
            content = str(turn.get("content", "") or "")
            if role != "user" or not content.strip():
                i += 1
                continue

            response = ""
            end_idx = i
            if i + 1 < len(turns):
                next_turn = turns[i + 1] if isinstance(turns[i + 1], dict) else {}
                if str(next_turn.get("role", "")).lower() == "assistant":
                    response = str(next_turn.get("content", "") or "")
                    end_idx = i + 1

            memories.append(
                {
                    "user_input": content,
                    "agent_response": response,
                    "timestamp": fmt_ts(base_dt + timedelta(seconds=30 * i)),
                    "meta_data": {
                        "source_session_id": session_id,
                        "source_turn_indices": [i, end_idx],
                        "question_id": item.get("question_id"),
                    },
                }
            )
            i = end_idx + 1

    return memories


def chat_judge(client: OpenAI, model: str, prompt: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": "You are a strict grader. Answer with yes or no only."},
        {"role": "user", "content": prompt},
    ]
    started = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=64,
        stream=False,
    )
    elapsed = time.perf_counter() - started
    usage = getattr(resp, "usage", None)
    text = resp.choices[0].message.content or ""
    return {
        "text": text,
        "duration_seconds": elapsed,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
        "messages": messages,
    }


def write_summary(out_dir: Path, total: int) -> None:
    rows = []
    for path in sorted(out_dir.glob("item_*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if row.get("status") == "ok":
            rows.append(row)

    correct = sum(int(r.get("correct", 0)) for r in rows)
    by_type: dict[str, dict[str, int]] = {}
    for row in rows:
        qtype = str(row.get("question_type", "unknown"))
        bucket = by_type.setdefault(qtype, {"correct": 0, "total": 0})
        bucket["total"] += 1
        bucket["correct"] += int(row.get("correct", 0))

    summary = {
        "total_dataset_items": total,
        "completed_items": len(rows),
        "correct_items": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "by_question_type": {
            k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0.0}
            for k, v in sorted(by_type.items())
        },
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MemoryOS pypi backend on LongMemEval-S.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--from-index", type=int, default=0)
    parser.add_argument("--to-index", type=int, default=None)
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--llm-model", default="google/gemma-4-12B-it")
    parser.add_argument("--llm-api-key", default="EMPTY")
    parser.add_argument("--embed-base-url", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embed-api-key", default="EMPTY")
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--assistant-id", default="longmemeval_assistant")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    end = args.to_index if args.to_index is not None else len(data)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.store_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": str(args.dataset),
        "from_index": args.from_index,
        "to_index": end,
        "llm_base_url": args.llm_base_url,
        "llm_model": args.llm_model,
        "embedding_base_url": args.embed_base_url,
        "embedding_model": args.embed_model,
        "embedding_dim": args.embed_dim,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    judge_client = OpenAI(api_key=args.llm_api_key, base_url=args.llm_base_url)

    for idx in range(args.from_index, min(end, len(data))):
        item = data[idx]
        qid = str(item.get("question_id") or f"{idx:05d}")
        out_path = args.out_dir / f"item_{idx:05d}_{qid}.json"
        if args.resume and out_path.exists():
            continue

        row: dict[str, Any] = {
            "index": idx,
            "question_id": qid,
            "question_type": item.get("question_type"),
            "question": item.get("question"),
            "gold_answer": item.get("answer"),
            "status": "started",
        }
        t0 = time.perf_counter()
        try:
            memories = iter_memory_pairs(item)
            mem = Memoryos(
                user_id=f"longmemeval_{idx:05d}_{qid}",
                openai_api_key=args.llm_api_key,
                openai_base_url=args.llm_base_url,
                data_storage_path=str(args.store_dir),
                assistant_id=args.assistant_id,
                llm_model=args.llm_model,
                embedding_model_name=args.embed_model,
                embedding_model_kwargs={
                    "api_base_url": args.embed_base_url,
                    "api_key": args.embed_api_key,
                },
                performance_log_dir=str(args.out_dir / "performance" / f"item_{idx:05d}_{qid}"),
            )
            add_t0 = time.perf_counter()
            mem.add_memories(memories, preserve_recent_short_term=True)
            add_elapsed = time.perf_counter() - add_t0

            ref_date = str(item.get("question_date") or "")
            query = (
                f"Reference date: {ref_date}\n"
                f"Question: {item.get('question')}\n"
                "Answer the question using the user's historical memory. If the memory is insufficient, say so."
            )
            ans_t0 = time.perf_counter()
            answer = mem.get_response(query)
            answer_elapsed = time.perf_counter() - ans_t0

            judge_prompt = get_anscheck_prompt(
                str(item.get("question_type")),
                str(item.get("question")),
                str(item.get("answer")),
                answer or "",
                abstention=("abs" in qid.lower()),
            )
            judge = chat_judge(judge_client, args.llm_model, judge_prompt)
            correct = 1 if true_or_false(judge["text"]) else 0
            row.update(
                {
                    "status": "ok",
                    "memory_pairs": len(memories),
                    "answer": answer,
                    "correct": correct,
                    "judge_response": judge["text"],
                    "judge_usage": {
                        "prompt_tokens": judge["prompt_tokens"],
                        "completion_tokens": judge["completion_tokens"],
                        "total_tokens": judge["total_tokens"],
                    },
                    "timing_seconds": {
                        "add": add_elapsed,
                        "answer": answer_elapsed,
                        "judge": judge["duration_seconds"],
                        "total": time.perf_counter() - t0,
                    },
                }
            )
            print(
                f"[{idx:05d}] qid={qid} status=ok correct={correct} "
                f"pairs={len(memories)} total={row['timing_seconds']['total']:.2f}s",
                flush=True,
            )
        except Exception as exc:
            row.update(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "timing_seconds": {"total": time.perf_counter() - t0},
                }
            )
            print(f"[{idx:05d}] qid={qid} status=error error={row['error']}", flush=True)
        out_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        write_summary(args.out_dir, len(data))

    write_summary(args.out_dir, len(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
