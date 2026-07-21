#!/usr/bin/env python3
"""Reconcile a Zep Local 2x3 run and export paper-ready audit tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


MODELS = ("qwen4b", "qwen30b", "gemma")
BENCHMARKS = ("longmemeval", "locomo")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - rank) + ordered[upper] * (rank - lower)


def rounded(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def iso_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    return datetime.fromisoformat(value).timestamp()


def active_attempt(run_root: Path) -> tuple[int | None, float | None]:
    attempts = read_json(run_root / "attempts.json", {})
    active = attempts.get("active_attempt")
    for attempt in attempts.get("attempts") or []:
        if attempt.get("attempt") == active:
            return int(active), iso_timestamp(attempt.get("started_at"))
    return None, None


def aggregate_calls(
    run_dir: Path,
    model: str,
    benchmark: str,
    cutoff: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "calls": 0,
            "errors": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latencies": [],
            "started": [],
            "finished": [],
        }
    )
    for filename in ("llm_calls.jsonl", "embedding_calls.jsonl"):
        for row in iter_jsonl(run_dir / "calls" / filename):
            started = iso_timestamp(row.get("started_at"))
            if cutoff is not None and (started is None or started < cutoff):
                continue
            key = (str(row.get("kind") or "unknown"), str(row.get("stage") or "unknown"))
            item = grouped[key]
            item["calls"] += 1
            item["errors"] += int(row.get("status") != "ok")
            for token_key in ("input_tokens", "output_tokens", "total_tokens"):
                item[token_key] += int(row.get(token_key) or 0)
            item["latencies"].append(float(row.get("latency_ms") or 0.0))
            finished = iso_timestamp(row.get("finished_at"))
            if started is not None:
                item["started"].append(started)
            if finished is not None:
                item["finished"].append(finished)

    rows: list[dict[str, Any]] = []
    totals = Counter()
    for (kind, stage), item in sorted(grouped.items()):
        stage_wall = None
        if item["started"] and item["finished"]:
            stage_wall = max(item["finished"]) - min(item["started"])
        row = {
            "model_key": model,
            "benchmark": benchmark,
            "kind": kind,
            "stage": stage,
            "calls": item["calls"],
            "errors": item["errors"],
            "input_tokens": item["input_tokens"],
            "output_tokens": item["output_tokens"],
            "total_tokens": item["total_tokens"],
            "latency_ms_p50": rounded(percentile(item["latencies"], 0.50), 3),
            "latency_ms_p95": rounded(percentile(item["latencies"], 0.95), 3),
            "latency_ms_sum": rounded(sum(item["latencies"]), 3),
            "stage_wall_seconds": rounded(stage_wall, 3),
        }
        rows.append(row)
        for key in ("calls", "errors", "input_tokens", "output_tokens", "total_tokens"):
            totals[key] += item[key]
    return rows, dict(totals)


def aggregate_graph(run_dir: Path) -> dict[str, Any]:
    markers = list((run_dir / "build" / "groups").glob("*.complete.json"))
    totals = Counter()
    wall_seconds: list[float] = []
    for path in markers:
        marker = read_json(path, {})
        stats = marker.get("stats") or {}
        for key in ("nodes", "episodes", "entities", "communities", "relationships"):
            totals[key] += int(stats.get(key) or 0)
        wall_seconds.append(float(marker.get("wall_seconds") or 0.0))
    return {
        "built_groups": len(markers),
        **dict(totals),
        "group_build_seconds_sum": rounded(sum(wall_seconds), 3),
        "group_build_seconds_p50": rounded(percentile(wall_seconds, 0.50), 3),
        "group_build_seconds_p95": rounded(percentile(wall_seconds, 0.95), 3),
    }


def aggregate_queries(run_dir: Path) -> dict[str, Any]:
    paths = list((run_dir / "query" / "items").glob("*.json"))
    context_chars: list[float] = []
    latencies: list[float] = []
    empty = 0
    retrieved = Counter()
    for path in paths:
        row = read_json(path, {})
        chars = int(row.get("context_chars") or 0)
        context_chars.append(float(chars))
        latencies.append(float(row.get("latency_seconds") or 0.0))
        empty += int(chars == 0)
        for key, count in (row.get("retrieved_counts") or {}).items():
            retrieved[key] += int(count or 0)
    count = len(paths)
    result: dict[str, Any] = {
        "query_items": count,
        "empty_contexts": empty,
        "empty_context_rate": rounded(empty / count if count else None),
        "context_chars_mean": rounded(sum(context_chars) / count if count else None, 2),
        "context_chars_p50": rounded(percentile(context_chars, 0.50), 2),
        "context_chars_p95": rounded(percentile(context_chars, 0.95), 2),
        "query_seconds_p50": rounded(percentile(latencies, 0.50), 4),
        "query_seconds_p95": rounded(percentile(latencies, 0.95), 4),
    }
    for key in ("edges", "nodes", "episodes", "communities"):
        result[f"retrieved_{key}_mean"] = rounded(retrieved[key] / count if count else None, 3)
    return result


def aggregate_answers(run_dir: Path) -> dict[str, Any]:
    paths = list((run_dir / "answers" / "items").glob("*.json"))
    latencies: list[float] = []
    empty = 0
    abstained = 0
    for path in paths:
        row = read_json(path, {})
        responses = row.get("responses") or []
        answer = str(responses[0].get("answer") if responses else "").strip()
        empty += int(not answer)
        abstained += int("insufficient context" in answer.lower() or "cannot be answered" in answer.lower())
        latencies.append(float(row.get("latency_seconds") or 0.0))
    count = len(paths)
    return {
        "answer_items": count,
        "empty_answers": empty,
        "abstained_answers": abstained,
        "abstention_rate": rounded(abstained / count if count else None),
        "answer_seconds_p50": rounded(percentile(latencies, 0.50), 4),
        "answer_seconds_p95": rounded(percentile(latencies, 0.95), 4),
    }


def repair_diagnostics(run_dir: Path, cutoff: float | None) -> dict[str, Any]:
    stages = Counter()
    for row in iter_jsonl(run_dir / "errors" / "graphiti_json_failures.jsonl"):
        timestamp = iso_timestamp(row.get("timestamp"))
        if cutoff is not None and (timestamp is None or timestamp < cutoff):
            continue
        stages[str(row.get("stage") or "unknown")] += 1
    return {
        "json_failure_records": sum(stages.values()),
        "json_failure_stages": json.dumps(dict(sorted(stages.items())), sort_keys=True),
    }


def judge_index(run_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_json(run_root / "judge" / "summary_passk.json", []):
        if row.get("slice") == "overall":
            result[(str(row.get("method")), str(row.get("benchmark")))] = row
    return result


def summarize(run_root: Path, out_dir: Path) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []
    judge = judge_index(run_root)
    attempt_id, cutoff = active_attempt(run_root)
    for model in MODELS:
        for benchmark in BENCHMARKS:
            run_dir = run_root / model / benchmark
            manifest = read_json(run_dir / "manifest.json", {})
            build_summary = read_json(run_dir / "build" / "summary.json", {})
            query_summary = read_json(run_dir / "query" / "summary.json", {})
            answer_summary = read_json(run_dir / "answers" / "summary.json", {})
            calls, call_totals = aggregate_calls(run_dir, model, benchmark, cutoff)
            call_rows.extend(calls)
            judged = judge.get((f"zep_local_{model}", benchmark), {})
            cell = {
                "model_key": model,
                "model": manifest.get("model"),
                "benchmark": benchmark,
                "attempt": attempt_id,
                "status": (
                    "complete"
                    if build_summary and query_summary and answer_summary
                    and not build_summary.get("failures")
                    and not query_summary.get("failures")
                    and not answer_summary.get("failures")
                    else "running_or_incomplete"
                ),
                "expected_groups": manifest.get("group_count"),
                "build_completed": build_summary.get("completed"),
                "build_failures": len(build_summary.get("failures") or []),
                "query_expected": query_summary.get("rows"),
                "query_completed": query_summary.get("completed"),
                "query_failures": len(query_summary.get("failures") or []),
                "answer_expected": answer_summary.get("rows"),
                "answer_completed": answer_summary.get("completed"),
                "answer_failures": len(answer_summary.get("failures") or []),
                **aggregate_graph(run_dir),
                **aggregate_queries(run_dir),
                **aggregate_answers(run_dir),
                **repair_diagnostics(run_dir, cutoff),
                "calls": call_totals.get("calls", 0),
                "call_errors": call_totals.get("errors", 0),
                "input_tokens": call_totals.get("input_tokens", 0),
                "output_tokens": call_totals.get("output_tokens", 0),
                "total_tokens": call_totals.get("total_tokens", 0),
                "judge_n": judged.get("n"),
                "judge_errors": judged.get("judge_error_questions"),
                "pass@1": judged.get("pass@1"),
                "pass@1_correct": judged.get("pass@1_correct"),
            }
            cells.append(cell)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "cell_summary.csv", cells)
    write_csv(out_dir / "call_breakdown.csv", call_rows)
    result = {
        "run_root": str(run_root),
        "cells": cells,
        "call_breakdown": call_rows,
    }
    (out_dir / "run_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    out_dir = (args.out_dir or run_root / "summary").resolve()
    result = summarize(run_root, out_dir)
    complete = sum(cell["status"] == "complete" for cell in result["cells"])
    print(f"wrote {out_dir}: complete_cells={complete}/{len(result['cells'])}")


if __name__ == "__main__":
    main()
