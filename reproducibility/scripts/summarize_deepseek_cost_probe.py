#!/usr/bin/env python3
"""Aggregate official DeepSeek billable token classes and direct cost."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


METHODS = ("memforest", "evermemos", "mem0", "memoryos", "zep_local")
DISPLAY = {
    "memforest": "MemForest",
    "evermemos": "EverMemOS",
    "mem0": "Mem0",
    "memoryos": "MemoryOS",
    "zep_local": "Zep Local",
}
HIT_USD_PER_M = 0.0028
MISS_USD_PER_M = 0.14
OUTPUT_USD_PER_M = 0.28


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args()

    grouped: dict[str, list[dict]] = defaultdict(list)
    failures: list[dict] = []
    with args.trace.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("record_type") != "request" or row.get("path") != "/v1/chat/completions":
                continue
            if row.get("status_code") == 200 and row.get("usage_available") is True:
                grouped[str(row.get("method"))].append(row)
            elif row.get("status_code") != 200:
                failures.append(row)

    summary = []
    for method in METHODS:
        rows = grouped[method]
        if not rows:
            raise RuntimeError(f"No successful billable requests captured for {method}")
        prompt = sum(int(row.get("prompt_tokens") or 0) for row in rows)
        hit = sum(int(row.get("prompt_cache_hit_tokens") or 0) for row in rows)
        miss = sum(int(row.get("prompt_cache_miss_tokens") or 0) for row in rows)
        output = sum(int(row.get("completion_tokens") or 0) for row in rows)
        if prompt != hit + miss:
            raise RuntimeError(f"Prompt decomposition failed for {method}: {prompt} != {hit}+{miss}")
        cost = (
            hit * HIT_USD_PER_M + miss * MISS_USD_PER_M + output * OUTPUT_USD_PER_M
        ) / 1_000_000
        summary.append(
            {
                "method": DISPLAY[method],
                "requests": len(rows),
                "cache_hit_input_tokens": hit,
                "cache_miss_input_tokens": miss,
                "output_tokens": output,
                "total_tokens": prompt + output,
                "cost_usd_20_messages": round(cost, 9),
                "cost_usd_per_1k_messages": round(cost * 50, 6),
            }
        )

    ever_cost = next(row["cost_usd_20_messages"] for row in summary if row["method"] == "EverMemOS")
    for row in summary:
        row["cost_efficiency_vs_evermemos"] = round(
            ever_cost / row["cost_usd_20_messages"], 3
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    validation = {
        "protocol_id": "deepseek_v4_flash_cost_probe_locomo_20messages_v2",
        "valid": not failures and all(grouped[method] for method in METHODS),
        "model": "deepseek-v4-flash",
        "thinking": "disabled",
        "prices_usd_per_million_tokens": {
            "cache_hit_input": HIT_USD_PER_M,
            "cache_miss_input": MISS_USD_PER_M,
            "output": OUTPUT_USD_PER_M,
        },
        "methods": summary,
        "failed_chat_requests": len(failures),
        "metric_semantics": "Direct cost from official API-returned billable token classes",
    }
    args.validation.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    if not validation["valid"]:
        raise SystemExit("DeepSeek cost probe validation failed")


if __name__ == "__main__":
    main()
