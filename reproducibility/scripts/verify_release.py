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

    profile = read_json(
        ROOT / "results" / "mem0_corrected" / "retrieval_budget_store_profile.json"
    )
    assert profile["validation"] == {
        "conversations": 500,
        "duplicate_conversation_point_keys": 0,
        "memory_points": 121594,
        "multi_user_conversations": 0,
    }
    assert abs(float(profile["overall"]["mean"]) - 243.188) < 1e-12
    top200 = profile["overall"]["budgets"]["200"]
    assert top200["questions_retrieving_all_memories"] == 40
    assert abs(float(top200["mean_store_fraction"]) - 0.8311433156049872) < 1e-12

    control_root = ROOT / "results" / "mem0_corrected" / "top50_top200_control"
    control = read_json(control_root / "manifest.json")
    assert control["validation"]["answers"] == 1000
    assert control["validation"]["judge_calls"] == 6000
    assert control["validation"]["answer_errors"] == 0
    assert control["validation"]["judge_errors"] == 0
    assert control["validation"]["strict_public_majority_disagreements"] == 40
    with (control_root / "accuracy_summary.csv").open(encoding="utf-8", newline="") as handle:
        accuracy_rows = list(csv.DictReader(handle))
    assert len(accuracy_rows) == 8
    accuracy = {
        (int(row["cutoff"]), row["judge_arm"], row["scope"]): float(row["accuracy"])
        for row in accuracy_rows
    }
    assert accuracy[(50, "strict", "overall")] == 0.466
    assert accuracy[(50, "public", "overall")] == 0.494
    assert accuracy[(200, "strict", "overall")] == 0.458
    assert accuracy[(200, "public", "overall")] == 0.486
    with (control_root / "per_question_majority.csv").open(encoding="utf-8", newline="") as handle:
        majority_rows = list(csv.DictReader(handle))
    assert len(majority_rows) == 2000
    assert Counter((int(row["cutoff"]), row["judge_arm"]) for row in majority_rows) == {
        (50, "strict"): 500,
        (50, "public"): 500,
        (200, "strict"): 500,
        (200, "public"): 500,
    }
    with (control_root / "paired_comparisons.csv").open(encoding="utf-8", newline="") as handle:
        paired_rows = list(csv.DictReader(handle))
    assert len(paired_rows) == 8
    budget_rows = [
        row for row in paired_rows
        if row["comparison"] == "top200_minus_top50" and row["scope"] == "overall"
    ]
    assert len(budget_rows) == 2
    assert all(float(row["delta"]) == -0.008 for row in budget_rows)
    judge_rows = [
        row for row in paired_rows
        if row["comparison"] == "public_minus_strict" and row["scope"] == "overall"
    ]
    assert len(judge_rows) == 2
    assert all(float(row["delta"]) == 0.028 for row in judge_rows)
    with (control_root / "context_summary.csv").open(encoding="utf-8", newline="") as handle:
        contexts = {int(row["cutoff"]): row for row in csv.DictReader(handle)}
    assert abs(float(contexts[50]["mean_prompt_tokens"]) - 2284.322) < 1e-12
    assert abs(float(contexts[200]["mean_prompt_tokens"]) - 7919.126) < 1e-12
    assert int(contexts[200]["stores_exhausted"]) == 40


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


def check_public_judge_three_backbone() -> None:
    root = ROOT / "results" / "public_judge_three_backbone"
    validation = read_json(root / "validation.json")
    assert validation["expected_inputs"] == 59664
    assert validation["complete_question_rows"] == 59664
    assert validation["unresolved_question_rows"] == 0
    assert validation["all_question_rows_complete"] is True

    manifest = read_json(root / "input_manifest.json")
    assert manifest["judge_model"] == "deepseek-v4-flash"
    assert manifest["frozen_inputs"] == 59664
    assert manifest["api_key_recorded"] is False

    with (root / "per_question_labels.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        labels = list(csv.DictReader(handle))
    assert len(labels) == 59664

    with (root / "summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = {
        (row["model"], row["method"], row["benchmark"], row["slice"]):
        float(row["public_accuracy"])
        for row in rows
    }
    expected = {
        ("qwen3_4b", "memforest", "longmemeval", "overall"): 0.726,
        ("qwen3_30b", "memforest", "longmemeval", "overall"): 0.818,
        ("gemma4_12b", "memforest_embed", "longmemeval", "overall"): 0.784,
        ("qwen3_4b", "memforest", "locomo", "cat1-4"): 0.7811688311688312,
        ("qwen3_30b", "memforest", "locomo", "cat1-4"): 0.8409090909090909,
        ("gemma4_12b", "evermemos", "locomo", "cat1-4"): 0.8857142857142857,
    }
    for key, expected_value in expected.items():
        assert abs(values[key] - expected_value) < 1e-12, (key, values[key])


def check_author_adjudicated_audit() -> None:
    root = ROOT / "results" / "semantic_audit" / "author_adjudicated"
    summary = read_json(root / "summary.json")
    assert summary["temporal"]["source_rows"] == 249
    assert summary["temporal"]["retained_rows"] == 231
    assert summary["temporal"]["excluded_rows"] == 18
    assert summary["entity"]["rows"] == 300
    assert summary["entity"]["active_precision"] == {
        "FAIL": 2,
        "PARTIAL": 1,
        "PASS": 124,
    }
    assert summary["judge"]["rows"] == 120

    expected_rows = {
        "temporal_labels_231.csv": 231,
        "temporal_excluded_18.csv": 18,
        "entity_routing_300.csv": 300,
        "judge_calibration_120.csv": 120,
    }
    for name, expected in expected_rows.items():
        with (root / name).open(encoding="utf-8", newline="") as handle:
            assert sum(1 for _ in csv.DictReader(handle)) == expected


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


def check_semantic_audit() -> None:
    root = ROOT / "results" / "semantic_audit"
    summary = read_json(root / "semantic_audit_summary.json")
    assert summary["provenance"]["human_gold"] is False
    assert summary["temporal"] == {
        "rows": 79,
        "high_confidence": 77,
        "provisional_fully_upheld": 33,
        "author_signoff_required": 46,
        "type_exact_match": 46,
        "gold_ids_exact_match": 65,
        "conflict_ids_exact_match": 65,
    }
    entity = summary["entity_routing"]
    assert entity["rows"] == 200
    assert entity["active_key_rows"] == 86
    assert entity["active_precision_pass"] == 84
    assert entity["active_full_recall_pass"] == 5
    judge = summary["judge_disagreements"]
    assert judge["rows"] == 40
    assert judge["codex_prefers_strict"] == 33
    assert judge["codex_prefers_public"] == 7
    assert judge["author_signoff_required"] == 8
    expected_raw = {"temporal": 79, "entity": 200, "judge": 40}
    for name, expected in expected_raw.items():
        rows = list(read_jsonl(root / "raw" / f"{name}_independent_audit.jsonl"))
        assert len(rows) == expected
        assert not any(row.get("error") for row in rows)

    expanded = root / "expanded"
    expanded_summary = read_json(expanded / "expanded_semantic_audit_summary.json")
    assert expanded_summary["provenance"]["human_gold"] is False
    temporal = expanded_summary["temporal"]
    assert temporal["rows"] == 249
    assert temporal["model_assisted_fully_upheld"] == 162
    assert temporal["author_signoff_required"] == 87
    entity = expanded_summary["entity_routing"]
    assert entity["rows"] == 300
    assert entity["active_key_rows"] == 127
    assert entity["active_precision_pass"] == 124
    assert entity["active_precision_author_signoff_required"] == 3
    judge = expanded_summary["judge_calibration"]
    assert judge["rows"] == 120
    assert judge["agreement_rows"] == 80
    assert judge["disagreement_rows"] == 40
    assert judge["stable_without_author_signoff"] == 108
    assert judge["author_signoff_required"] == 12
    expanded_raw = {"temporal": 249, "entity": 300, "judge": 120}
    for name, expected in expanded_raw.items():
        rows = list(read_jsonl(
            expanded / "raw" / f"{name}_independent_audit_{expected}.jsonl"
        ))
        assert len(rows) == expected
        assert not any(row.get("error") for row in rows)


def main() -> None:
    check_baseline_implementations()
    check_mem0()
    check_gemma()
    check_zep()
    check_judge_sensitivity()
    check_public_judge_three_backbone()
    check_semantic_audit()
    check_author_adjudicated_audit()
    print("revision release verification: PASS")


if __name__ == "__main__":
    main()
