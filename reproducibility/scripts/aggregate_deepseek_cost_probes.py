#!/usr/bin/env python3
"""Aggregate independent DeepSeek 20-message cost probes."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METHODS = ("MemForest", "EverMemOS", "Mem0", "MemoryOS", "Zep Local")
METRICS = (
    "requests",
    "cache_hit_input_tokens",
    "cache_miss_input_tokens",
    "output_tokens",
    "total_tokens",
    "cost_usd_20_messages",
    "cost_usd_per_1k_messages",
)


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
    dataset = json.loads(
        (run_dir / "mini_locomo_flat.json").read_text(encoding="utf-8")
    )
    if not validation.get("valid") or validation.get("failed_chat_requests") != 0:
        raise ValueError(f"invalid or failed requests in {run_dir}")
    if len(dataset) != 1:
        raise ValueError(f"expected one frozen row in {run_dir}")
    row = dataset[0]
    message_count = sum(len(session) for session in row["haystack_sessions"])
    if message_count != 20:
        raise ValueError(f"expected 20 messages in {run_dir}, found {message_count}")
    source_id = str(row["locomo_sample_id"])
    if manifest.get("source_id") not in (None, source_id):
        raise ValueError(f"manifest/data source mismatch in {run_dir}")
    by_method = {str(item["method"]): item for item in validation["methods"]}
    if set(by_method) != set(METHODS):
        raise ValueError(f"method mismatch in {run_dir}: {sorted(by_method)}")
    meta = {
        "run_id": run_dir.name,
        "source_id": source_id,
        "question_id": str(row["question_id"]),
        "dataset_sha256": manifest["dataset_sha256"],
        "completed_at_utc": manifest["completed_at_utc"],
    }
    return meta, [by_method[method] for method in METHODS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--detail-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) != 3:
        raise SystemExit(f"expected exactly three runs, found {len(args.run)}")

    loaded = [load_run(path) for path in args.run]
    metadata = [item[0] for item in loaded]
    if len({item["source_id"] for item in metadata}) != 3:
        raise ValueError("the three probes must use distinct conversations")

    detail = []
    for meta, rows in loaded:
        for row in rows:
            detail.append({**meta, **row})

    summary = []
    for method in METHODS:
        method_rows = [row for row in detail if row["method"] == method]
        result = {"method": method, "probes": len(method_rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in method_rows]
            result[f"{metric}_mean"] = statistics.fmean(values)
            result[f"{metric}_min"] = min(values)
            result[f"{metric}_max"] = max(values)
        summary.append(result)

    ever_cost = next(
        row["cost_usd_20_messages_mean"]
        for row in summary
        if row["method"] == "EverMemOS"
    )
    for row in summary:
        row["cost_efficiency_vs_evermemos"] = (
            ever_cost / row["cost_usd_20_messages_mean"]
        )

    args.detail_output.parent.mkdir(parents=True, exist_ok=True)
    with args.detail_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(detail[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(detail)
    with args.summary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summary[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)

    validation = {
        "protocol_id": "deepseek_v4_flash_cost_probe_locomo_3x20messages_v1",
        "valid": True,
        "model": "deepseek-v4-flash",
        "thinking": "disabled",
        "probe_count": 3,
        "distinct_conversations": [item["source_id"] for item in metadata],
        "runs": metadata,
        "aggregation": "unweighted arithmetic mean across three equal-size probes; min/max retained",
        "cost_efficiency": "ratio of EverMemOS mean cost to each method mean cost",
        "prices_usd_per_million_tokens": {
            "cache_hit_input": 0.0028,
            "cache_miss_input": 0.14,
            "output": 0.28,
        },
        "methods": summary,
        "failed_chat_requests": 0,
    }
    args.validation_output.write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "protocol_id": validation["protocol_id"],
        "scope": "three distinct LoCoMo conversations, each truncated to 20 messages plus one frozen retrieval and answer",
        "model": validation["model"],
        "thinking": validation["thinking"],
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "cache_isolation": "fresh DeepSeek user_id namespace per method and conversation",
        "aggregation": validation["aggregation"],
        "conversations": metadata,
        "methods": list(METHODS),
        "secrets_persisted": False,
    }
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
