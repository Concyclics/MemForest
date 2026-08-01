#!/usr/bin/env python3
"""Reconcile MemForest's first extraction wave across vLLM and DeepSeek."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def first_wave(path: Path, backend: str) -> dict[str, Any]:
    rows = load_rows(path)
    if backend == "vllm":
        rows = [
            row
            for row in rows
            if row.get("method") == "memforest" and row.get("cached_tokens") is not None
        ]
        hit_key = "cached_tokens"
    else:
        rows = [
            row
            for row in rows
            if row.get("record_type") == "request" and row.get("method") == "memforest"
        ]
        hit_key = "prompt_cache_hit_tokens"
    rows = rows[:10]
    if len(rows) != 10:
        raise ValueError(f"{backend}: expected 10 first-wave requests, found {len(rows)}")
    times = [datetime.fromisoformat(str(row["time"])) for row in rows]
    hashes = sorted(str(row["prompt_hash"]) for row in rows)
    return {
        "backend": backend,
        "requests": len(rows),
        "start_span_ms": round((max(times) - min(times)).total_seconds() * 1000, 3),
        "hit_requests": sum(int(row.get(hit_key) or 0) > 0 for row in rows),
        "hit_tokens": sum(int(row.get(hit_key) or 0) for row in rows),
        "prompt_hashes": hashes,
        "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-trace", type=Path, required=True)
    parser.add_argument("--deepseek-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    vllm = first_wave(args.vllm_trace, "vllm")
    deepseek = first_wave(args.deepseek_trace, "deepseek")
    same_hashes = vllm["prompt_hashes"] == deepseek["prompt_hashes"]
    if not same_hashes:
        raise ValueError("first-wave prompt hashes differ across backends")
    payload = {
        "protocol_id": "memforest_cache_backend_reconciliation_v1",
        "valid": True,
        "same_prompt_hash_multiset": same_hashes,
        "interpretation": (
            "vLLM cached_tokens measure scheduler-level avoided prefill; DeepSeek "
            "prompt_cache_hit_tokens measure persisted provider cache units used for billing."
        ),
        "backends": [
            {key: value for key, value in vllm.items() if key != "prompt_hashes"},
            {key: value for key, value in deepseek.items() if key != "prompt_hashes"},
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
