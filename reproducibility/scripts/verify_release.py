#!/usr/bin/env python3
"""Verify compact VLDB revision result artifacts without external services."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def check_zep_native_budget() -> None:
    root = ROOT / "results" / "zep_local"
    with (root / "native_budget_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    observed = {(row["model_key"], row["benchmark"]): row for row in rows}
    expected = {
        ("qwen4b", "longmemeval"): (500, 3268.628, 4542),
        ("qwen4b", "locomo"): (1986, 1359.575, 1626),
        ("qwen30b", "longmemeval"): (500, 3125.450, 4412),
        ("qwen30b", "locomo"): (1986, 1210.353, 1413),
        ("gemma", "longmemeval"): (500, 3104.718, 4364),
        ("gemma", "locomo"): (1986, 1167.739, 1348),
    }
    for key, (questions, mean_tokens, p95_tokens) in expected.items():
        row = observed[key]
        assert int(row["questions"]) == questions
        assert abs(float(row["edges_mean"]) - 5.0) < 1e-12
        assert abs(float(row["communities_mean"])) < 1e-12
        assert abs(float(row["context_tokens_mean"]) - mean_tokens) < 1e-12
        assert int(row["context_tokens_p95"]) == p95_tokens
    manifest = read_json(root / "native_budget_manifest.json")
    assert manifest["protocol_id"] == "zep_native_budget_v1_20260801"
    assert len(manifest["sources"]) == 6
    assert sum(item["query_files"] for item in manifest["sources"].values()) == 7458


def check_write_path_traces() -> None:
    root = ROOT / "results" / "write_path_traces"
    with (root / "summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    observed = {
        (row["benchmark"], row["method"]): float(
            row["build_rate_turns_per_second"]
        )
        for row in rows
    }
    expected = {
        ("longmemeval", "memforest"): 2.841184640,
        ("longmemeval", "evermemos"): 0.470557491,
        ("longmemeval", "mem0"): 1.397339121,
        ("longmemeval", "memoryos"): 0.202259514,
        ("longmemeval", "zep_local"): 0.114172343,
        ("locomo", "memforest"): 7.019564973,
        ("locomo", "evermemos"): 0.738216007,
        ("locomo", "mem0"): 0.585997832,
        ("locomo", "memoryos"): 0.244689896,
        ("locomo", "zep_local"): 0.215605958,
    }
    for key, expected_value in expected.items():
        assert abs(observed[key] - expected_value) < 1e-12, key
    locomo_rows = [row for row in rows if row["benchmark"] == "locomo"]
    assert {row["source_id"] for row in locomo_rows} == {"conv-43"}
    assert {row["cross_instance_concurrency"] for row in locomo_rows} == {"1"}
    with (root / "memforest_longmemeval_instances.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        instances = list(csv.DictReader(handle))
    assert len(instances) == 3
    assert {row["qid"] for row in instances} == {
        "28dc39ac", "bc8a6e93", "7e00a6cb"
    }
    manifest = read_json(root / "manifest.json")
    assert manifest["protocol_id"] == (
        "figure1_native_write_trace_provenance_v1_20260801"
    )
    assert len(manifest["source_sha256"]) == 5


def check_runtime_configs() -> None:
    config = read_json(ROOT / "manifests" / "runtime_configs.json")
    assert config["memforest"]["canonicalization_top_k"] == 8
    assert config["memforest"]["canonicalization_similarity_threshold"] == 0.93
    assert config["memforest"]["tree_browse_beam_width"] == 10
    assert config["figure1_locomo"]["cross_instance_concurrency"] == 1
    assert config["zep_local_full_benchmark"]["benchmark_concurrency"] == 128


def check_judge_sensitivity() -> None:
    path = ROOT / "results" / "judge_prompt_sensitivity" / "summary.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15

    stratified_path = (
        ROOT / "results" / "judge_prompt_sensitivity" /
        "stratified_summary.csv"
    )
    with stratified_path.open(encoding="utf-8", newline="") as handle:
        stratified = list(csv.DictReader(handle))
    assert len(stratified) == 30
    temporal = {
        (row["benchmark"], row["method"], row["arm"]): row
        for row in stratified
        if row["scope"] == "temporal"
    }
    expected_temporal = {
        ("locomo", "evermemos", "appendix"): (150, 103),
        ("locomo", "evermemos", "mem0_tuned"): (150, 128),
        ("locomo", "memforest", "appendix"): (150, 107),
        ("locomo", "memforest", "mem0_tuned"): (150, 132),
        ("locomo", "mem0", "appendix"): (150, 9),
        ("locomo", "mem0", "mem0_tuned"): (150, 19),
        ("longmemeval", "evermemos", "appendix"): (122, 64),
        ("longmemeval", "evermemos", "mem0_initial"): (122, 70),
        ("longmemeval", "memforest", "appendix"): (122, 96),
        ("longmemeval", "memforest", "mem0_initial"): (122, 99),
        ("longmemeval", "mem0", "appendix"): (122, 37),
        ("longmemeval", "mem0", "mem0_initial"): (122, 43),
    }
    for key, (expected_n, expected_correct) in expected_temporal.items():
        assert int(temporal[key]["n"]) == expected_n
        assert int(temporal[key]["correct"]) == expected_correct

    manifest = read_json(ROOT / "manifests" / "judge_prompt_sensitivity.json")
    assert manifest["validation"]["observed_calls"] == 8496
    assert manifest["validation"]["judge_errors"] == 0
    assert manifest["scope"]["longmemeval_temporal_questions_per_method"] == 122
    assert manifest["scope"]["locomo_temporal_questions_per_method"] == 150


def check_public_judge_three_backbone() -> None:
    root = ROOT / "results" / "public_judge_three_backbone"
    validation = read_json(root / "validation.json")
    assert validation["expected_inputs"] == 59664
    assert validation["complete_question_rows"] == 59664
    assert validation["unresolved_question_rows"] == 0
    assert validation["all_question_rows_complete"] is True

    manifest = read_json(root / "input_manifest.json")
    assert manifest["judge_model"] == "deepseek-v4-flash"
    assert manifest["protocol_id"] == "three_backbone_full_public_native_unit_v3_20260731"
    assert manifest["repetitions"] == 1
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
        ("qwen3_4b", "memforest_embed", "longmemeval", "overall"): 0.694,
        ("qwen3_30b", "memforest_embed", "longmemeval", "overall"): 0.784,
        ("qwen3_4b", "memforest", "locomo", "cat1-4"): 0.7811688311688312,
        ("qwen3_30b", "memforest", "locomo", "cat1-4"): 0.8409090909090909,
        ("qwen3_4b", "memforest_embed", "locomo", "cat1-4"): 0.7662337662337663,
        ("qwen3_30b", "memforest_embed", "locomo", "cat1-4"): 0.8058441558441558,
        ("gemma4_12b", "evermemos", "locomo", "cat1-4"): 0.8857142857142857,
    }
    for key, expected_value in expected.items():
        assert abs(values[key] - expected_value) < 1e-12, (key, values[key])


def check_qwen_embed_main_protocol() -> None:
    root = ROOT / "results" / "qwen_embed_main_protocol"
    expected = {
        "qwen4b": {
            "model": "qwen3_4b",
            "sha256": "4e8690ea507e7a4b24ca510552f2ef37f094b4dbe9a8a403352fe49bae0e2fda",
            "mean_facts": {"longmemeval": 153.076, "locomo": 222.2754279959718},
        },
        "qwen30b": {
            "model": "qwen3_30b",
            "sha256": "9cc06648007f324402d11cc5372518d3ef75f30e549ae6b44637418a4401cca2",
            "mean_facts": {"longmemeval": 152.898, "locomo": 195.81772406847935},
        },
    }
    for stem, lane in expected.items():
        records_path = root / f"{stem}_records.jsonl"
        manifest = read_json(root / f"{stem}_records.manifest.json")
        assert manifest["protocol_id"] == "qwen_memforest_embed_native_top10_full_expand_v2_20260731"
        assert manifest["model"] == lane["model"]
        assert manifest["native_top_k"] == 10
        assert manifest["context_expansion"] == "full"
        assert manifest["answer_prompt"] == "memforest_default_v1"
        assert manifest["rows"] == 2486
        assert manifest["output_sha256"] == lane["sha256"]
        assert sha256(records_path) == lane["sha256"]
        rows = list(read_jsonl(records_path))
        assert Counter(row["benchmark"] for row in rows) == {
            "longmemeval": 500,
            "locomo": 1986,
        }
        assert len({(row["benchmark"], row["qid"]) for row in rows}) == 2486
        assert all(row["method"] == "memforest_embed_browse" for row in rows)
        assert all(row["native_top_k"] == 10 for row in rows)
        assert all(row["context_expansion"] == "full" for row in rows)
        assert all(row["answer_prompt"] == "memforest_default_v1" for row in rows)
        assert all(row["expanded_fact_count"] > 10 for row in rows)
        assert all(0 < row["context_chars"] <= 60000 for row in rows)
        for benchmark, expected_mean in lane["mean_facts"].items():
            selected = [row for row in rows if row["benchmark"] == benchmark]
            observed = sum(row["expanded_fact_count"] for row in selected) / len(selected)
            assert abs(observed - expected_mean) < 1e-12, (stem, benchmark, observed)


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


def check_deepseek_cost_probe() -> None:
    root = ROOT / "results" / "deepseek_cost_probe"
    validation = read_json(root / "validation.json")
    assert validation["valid"] is True
    assert validation["model"] == "deepseek-v4-flash"
    assert validation["thinking"] == "disabled"
    assert validation["failed_chat_requests"] == 0
    assert validation["probe_count"] == 3
    assert validation["distinct_conversations"] == ["conv-26", "conv-42", "conv-43"]
    expected = {
        "MemForest": (26601.666666666668, 24615, 28202, 0.0021152470000000002),
        "EverMemOS": (31398.666666666668, 30354, 32739, 0.004416824333333333),
        "Mem0": (19365.333333333332, 18010, 21594, 0.0010163163333333333),
        "MemoryOS": (13151.666666666666, 9088, 18932, 0.0019250296666666665),
        "Zep Local": (120484.33333333333, 117396, 125938, 0.011177157666666666),
    }
    rows = {row["method"]: row for row in validation["methods"]}
    assert set(rows) == set(expected)
    for method, (mean_total, min_total, max_total, mean_cost) in expected.items():
        assert abs(rows[method]["total_tokens_mean"] - mean_total) < 1e-12
        assert rows[method]["total_tokens_min"] == min_total
        assert rows[method]["total_tokens_max"] == max_total
        assert abs(rows[method]["cost_usd_20_messages_mean"] - mean_cost) < 1e-12
    trace = list(read_jsonl(root / "llm_usage.jsonl"))
    successful = [
        row for row in trace
        if row.get("record_type") == "request"
        and row.get("path") == "/v1/chat/completions"
        and row.get("status_code") == 200
    ]
    assert len(successful) == 740
    assert {row["probe_source_id"] for row in successful} == {
        "conv-26", "conv-42", "conv-43"
    }
    assert all("messages" not in row and "api_key" not in row for row in trace)


def check_write_conflicts() -> None:
    root = ROOT / "results" / "write_conflicts"
    manifest = read_json(root / "manifest.json")
    assert manifest["source_qids"] == ["28dc39ac", "bc8a6e93", "7e00a6cb"]
    assert manifest["tree_types"] == ["entity", "scene"]

    with (root / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert len(summary_rows) == 8
    aggregate = {
        row["scope"]: row for row in summary_rows if row["qid"] == "aggregate"
    }
    assert set(aggregate) == {"all", "without_global_fallback"}
    assert int(aggregate["all"]["sessions"]) == 147
    assert int(aggregate["all"]["max_facts_per_touched_tree"]) == 53
    assert abs(
        float(aggregate["all"]["session_pair_overlap_rate"])
        - 0.8931985796230538
    ) < 1e-12
    assert abs(
        float(aggregate["without_global_fallback"]["session_pair_overlap_rate"])
        - 0.16634799235181644
    ) < 1e-12

    with (root / "session_tree_loads.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        detail_rows = list(csv.DictReader(handle))
    assert len(detail_rows) == 1117
    assert Counter(row["tree_type"] for row in detail_rows) == {
        "scene": 780,
        "entity": 337,
    }


def check_async_core_snapshot() -> None:
    root = ROOT / "implementation" / "async_core"
    tree_source = (root / "bplustree.py").read_text(encoding="utf-8")
    forest_source = (root / "forest.py").read_text(encoding="utf-8")
    assert "class AsyncRWLock" in tree_source
    assert "async with self._rwlock.read_lock()" in tree_source
    assert "async with self._rwlock.write_lock()" in tree_source
    assert "async def _run_tree_inserts" in forest_source
    assert "async def _take_dirty_tree_snapshot" in forest_source
    assert "tmp.rename(target)" in forest_source


def main() -> None:
    check_baseline_implementations()
    check_mem0()
    check_gemma()
    check_zep()
    check_zep_native_budget()
    check_write_path_traces()
    check_runtime_configs()
    check_judge_sensitivity()
    check_qwen_embed_main_protocol()
    check_public_judge_three_backbone()
    check_semantic_audit()
    check_author_adjudicated_audit()
    check_deepseek_cost_probe()
    check_write_conflicts()
    check_async_core_snapshot()
    print("revision release verification: PASS")


if __name__ == "__main__":
    main()
