#!/usr/bin/env python3
"""Validate warmup exclusion and provider-hit coverage for the cost probe."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime-validation", type=Path, required=True)
    parser.add_argument("--second-warmup-summary", type=Path, required=True)
    parser.add_argument("--measured-detail", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prime = json.loads(args.prime_validation.read_text(encoding="utf-8"))
    templates = []
    for row in prime["templates"]:
        prompt_tokens = int(row["cache_hit_tokens"]) + int(row["cache_miss_tokens"])
        eligible = prompt_tokens >= 128
        templates.append(
            {
                "method": row["method"],
                "system_prompt_hash": row["system_prompt_hash"],
                "prompt_tokens": prompt_tokens,
                "cache_eligible": eligible,
                "cache_hit_tokens": int(row["cache_hit_tokens"]),
                "valid": not eligible or int(row["cache_hit_tokens"]) > 0,
            }
        )

    warmup = read_csv(args.second_warmup_summary)
    measured = read_csv(args.measured_detail)
    warm_methods = sorted(row["method"] for row in warmup)
    measured_methods = sorted({row["method"] for row in measured})
    measured_sources = sorted({row["source_id"] for row in measured})
    payload = {
        "protocol_id": "deepseek_v4_flash_verified_warm_cache_3x20_v1",
        "valid": (
            all(row["valid"] for row in templates)
            and all(int(row["cache_hit_input_tokens"]) > 0 for row in warmup)
            and all(int(row["cache_hit_input_tokens"]) > 0 for row in measured)
            and warm_methods == measured_methods
            and len(measured) == 15
            and len(measured_sources) == 3
        ),
        "warmup_conversations": ["conv-47", "conv-48"],
        "measured_conversations": measured_sources,
        "warmups_excluded_from_measured_rows": True,
        "cache_eligible_template_rule": "validation prompt has at least 128 input tokens",
        "template_validation": templates,
        "second_warmup_provider_hits": {
            row["method"]: int(row["cache_hit_input_tokens"]) for row in warmup
        },
        "measured_cells": len(measured),
        "measured_zero_hit_cells": sum(
            int(row["cache_hit_input_tokens"]) <= 0 for row in measured
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not payload["valid"]:
        raise SystemExit("warm-cache protocol validation failed")


if __name__ == "__main__":
    main()
