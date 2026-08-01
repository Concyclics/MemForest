#!/usr/bin/env python3
"""Summarize native Zep query objects and serialized context tokens."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


MODELS = ("qwen4b", "qwen30b", "gemma")
BENCHMARKS = ("longmemeval", "locomo")
COUNT_KEYS = ("edges", "nodes", "episodes", "communities")


def query_digest(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def percentile_nearest_lower(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def summarize(run_root: Path, tokenizer_name: str) -> tuple[list[dict], dict]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    rows = []
    sources = {}
    for model in MODELS:
        for benchmark in BENCHMARKS:
            query_root = run_root / model / benchmark / "query" / "items"
            paths = sorted(query_root.glob("*.json"))
            expected = 500 if benchmark == "longmemeval" else 1986
            if len(paths) != expected:
                raise ValueError(f"{model}/{benchmark}: {len(paths)} != {expected}")
            records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
            token_lists = tokenizer(
                [record["context"] for record in records], add_special_tokens=False
            )["input_ids"]
            token_counts = [len(tokens) for tokens in token_lists]
            row = {
                "model_key": model,
                "benchmark": benchmark,
                "questions": len(records),
            }
            for key in COUNT_KEYS:
                values = [int(record["retrieved_counts"][key]) for record in records]
                row[f"{key}_mean"] = f"{sum(values) / len(values):.5f}"
            row["context_tokens_mean"] = f"{sum(token_counts) / len(token_counts):.3f}"
            row["context_tokens_p95"] = percentile_nearest_lower(token_counts, 0.95)
            rows.append(row)
            sources[f"{model}/{benchmark}"] = {
                "query_files": len(paths),
                "query_digest": query_digest(paths, run_root),
            }
    manifest = {
        "protocol_id": "zep_native_budget_v1_20260801",
        "tokenizer": tokenizer_name,
        "token_policy": "Exact serialized context only; excludes answer instruction and question.",
        "percentile": "nearest-lower order statistic at floor(0.95*(n-1))",
        "sources": sources,
    }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    rows, manifest = summarize(args.run_root, args.tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
