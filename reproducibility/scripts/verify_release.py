#!/usr/bin/env python3
"""Verify compact VLDB revision result artifacts without external services."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def check_mem0() -> None:
    expected = {
        "qwen3_4b": (500, 178, 0.356, 0.366),
        "qwen3_30b": (500, 238, 0.4766666666666666, 0.476),
    }
    for lane, (count, correct, primary_pass1, sampled_pass1) in expected.items():
        root = ROOT / "results" / "mem0_corrected" / lane
        rows = list(read_jsonl(root / "per_question.jsonl"))
        assert len(rows) == count, (lane, len(rows))
        primary = read_json(root / "primary_pass1_three_judge.json")
        assert primary["total_questions"] == count
        assert primary["correct"] == correct
        assert abs(float(primary["accuracy"]) - primary_pass1) < 1e-12
        summary = read_json(root / "passk_summary.json")
        observed = float(summary["overall"]["pass@1"])
        assert abs(observed - sampled_pass1) < 1e-12, (lane, observed)


def check_gemma() -> None:
    root = ROOT / "results" / "gemma"
    coverage = read_json(root / "coverage.json")
    assert len(coverage) == 14
    assert all(row["missing"] == 0 and row["empty_answer"] == 0 for row in coverage)
    rows = list(read_jsonl(root / "per_question_judged.jsonl"))
    assert len(rows) == 14280
    assert not any(row.get("judge_error") for row in rows)
    assert not any("source_path" in row for row in rows)


def check_zep() -> None:
    roots = [
        ROOT / "results" / "zep_local" / "qwen3_4b",
        ROOT / "results" / "zep_local" / "qwen3_30b_gemma",
    ]
    counts = Counter()
    for root in roots:
        for row in read_jsonl(root / "per_question_judged.jsonl"):
            counts[row["method"]] += 1
            assert not row.get("judge_error")
            assert "context_source" not in row and "source_path" not in row
    assert counts == {
        "zep_local_qwen4b": 2486,
        "zep_local_qwen30b": 2486,
        "zep_local_gemma": 2486,
    }, counts


def check_judge_sensitivity() -> None:
    path = ROOT / "results" / "judge_prompt_sensitivity" / "summary.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15
    manifest = read_json(ROOT / "manifests" / "judge_prompt_sensitivity.json")
    assert manifest["validation"]["observed_calls"] == 8496
    assert manifest["validation"]["judge_errors"] == 0


def check_baseline_implementations() -> None:
    manifest = read_json(ROOT / "manifests" / "baseline_versions.json")
    expected = {
        "evermemos": ["upstream.patch", "scripts/prepare.sh", "scripts/run.sh"],
        "mem0": ["mem0_adapter.py", "mem0_local_qwen3.yaml"],
        "lightmem": ["upstream.patch", "scripts/run_longmemeval.sh", "scripts/run_locomo.sh"],
        "memoryos": ["upstream.patch", "scripts/run_longmemeval.py", "scripts/run_locomo_shards.sh"],
        "mempalace": ["upstream.patch", "scripts/run_longmemeval.sh", "scripts/run_locomo.sh"],
        "memorydata_zep_local": ["run_benchmark.py", "summarize_run.py"],
    }
    for method, relative_files in expected.items():
        artifact_path = manifest[method]["artifact_path"]
        baseline_root = ROOT / Path(artifact_path).relative_to("reproducibility")
        assert baseline_root.is_dir(), (method, baseline_root)
        for relative_file in relative_files:
            path = baseline_root / relative_file
            assert path.is_file() and path.stat().st_size > 0, (method, path)


def main() -> None:
    check_baseline_implementations()
    check_mem0()
    check_gemma()
    check_zep()
    check_judge_sensitivity()
    print("revision release verification: PASS")


if __name__ == "__main__":
    main()
