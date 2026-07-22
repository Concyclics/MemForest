#!/usr/bin/env python3
"""Validate and summarize the expanded model-assisted semantic audits.

These outputs are review aids, not human gold.  The script preserves every
sample selected before review and emits unresolved rows for author sign-off.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPRO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPRO_ROOT / "results/semantic_audit/expanded"
TABLE_DIR = OUTPUT_DIR / "source"
AUDIT_DIR = OUTPUT_DIR / "raw"


PUBLIC_PREFERRED = {
    "35a27287::k50",
    "54026fce::k50",
    "54026fce::k200",
    "d24813b1::k50",
    "d24813b1::k200",
    "c4a1ceb8::k50",
    "c4a1ceb8::k200",
}

# These disagreement cases remained interpretation-sensitive after the first
# independent review.  Four agreement cases also need author judgment because
# a complete but terse answer and a partially personalized answer are not
# treated consistently across judge policies.
AMBIGUOUS_DISAGREEMENTS = {
    "06f04340::k50",
    "06f04340::k200",
    "35a27287::k50",
    "54026fce::k50",
    "54026fce::k200",
    "55241a1f::k50",
    "d24813b1::k50",
    "d24813b1::k200",
}
AMBIGUOUS_AGREEMENTS = {
    "1d4e3b97::k200",
    "505af2f5::k200",
    "8a2466db::k50",
    "a89d7624::k50",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def split_ids(value: str | list[str]) -> set[str]:
    if isinstance(value, str):
        return {item for item in value.split(";") if item}
    return {str(item) for item in value}


def validate_audits(path: Path, expected: int) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != expected or any(row.get("error") for row in rows):
        raise ValueError(f"{path}: expected {expected} successful rows")
    by_key = {row["key"]: row for row in rows}
    if len(by_key) != expected:
        raise ValueError(f"{path}: duplicate keys")
    return by_key


def summarize_temporal() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source = read_csv(TABLE_DIR / "temporal_conflict_author_review_249.csv")
    if len(source) != 249:
        raise ValueError("Temporal source must contain 249 rows")
    audits = validate_audits(AUDIT_DIR / "temporal_independent_audit_249.jsonl", 249)

    output: list[dict[str, Any]] = []
    signoff: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    benchmark_counts: Counter[str] = Counter()
    for row in source:
        key = f"{row['benchmark']}::{row['qid']}"
        audit = audits[key]["audit"]
        candidate_ids = {item["fact_id"] for item in json.loads(row["candidate_facts_json"])}
        proposed_gold = split_ids(row["proposed_gold_fact_ids"])
        proposed_conflict = split_ids(row["proposed_conflict_fact_ids"])
        independent_gold = split_ids(audit["gold_fact_ids"])
        independent_conflict = split_ids(audit["conflict_fact_ids"])
        unknown = (independent_gold | independent_conflict) - candidate_ids
        if unknown:
            raise ValueError(f"{key}: unknown independent fact IDs {sorted(unknown)}")
        overlap = independent_gold & independent_conflict
        type_exact = row["proposed_temporal_type"] == audit["temporal_type"]
        gold_exact = proposed_gold == independent_gold
        conflict_exact = proposed_conflict == independent_conflict
        fully_upheld = bool(
            audit.get("proposed_label_valid") is True
            and audit.get("confidence") == "high"
            and type_exact
            and gold_exact
            and conflict_exact
            and not overlap
        )
        record = {
            "key": key,
            "benchmark": row["benchmark"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "selection_reason": row["selection_reason"],
            "double_label_required": row["double_label_required"],
            "proposed_temporal_type": row["proposed_temporal_type"],
            "independent_temporal_type": audit["temporal_type"],
            "proposed_gold_fact_ids": ";".join(sorted(proposed_gold)),
            "independent_gold_fact_ids": ";".join(sorted(independent_gold)),
            "proposed_conflict_fact_ids": ";".join(sorted(proposed_conflict)),
            "independent_conflict_fact_ids": ";".join(sorted(independent_conflict)),
            "type_exact_match": type_exact,
            "gold_ids_exact_match": gold_exact,
            "conflict_ids_exact_match": conflict_exact,
            "proposed_label_valid": audit.get("proposed_label_valid"),
            "independent_gold_conflict_overlap": ";".join(sorted(overlap)),
            "confidence": audit.get("confidence"),
            "model_assisted_fully_upheld": fully_upheld,
            "author_signoff_required": not fully_upheld,
            "independent_notes": audit.get("notes", ""),
        }
        output.append(record)
        type_counts[audit["temporal_type"]] += 1
        benchmark_counts[row["benchmark"]] += 1
        if not fully_upheld:
            signoff.append({
                **record,
                "author_temporal_type": "",
                "author_gold_fact_ids": "",
                "author_conflict_fact_ids": "",
                "author_notes": "",
            })

    summary = {
        "rows": 249,
        "selection_frozen_before_independent_review": True,
        "double_label_rows": sum(row["double_label_required"].lower() == "true" for row in source),
        "model_assisted_fully_upheld": sum(row["model_assisted_fully_upheld"] for row in output),
        "author_signoff_required": len(signoff),
        "type_exact_match": sum(row["type_exact_match"] for row in output),
        "gold_ids_exact_match": sum(row["gold_ids_exact_match"] for row in output),
        "conflict_ids_exact_match": sum(row["conflict_ids_exact_match"] for row in output),
        "by_benchmark": dict(sorted(benchmark_counts.items())),
        "independent_type_counts": dict(sorted(type_counts.items())),
    }
    return output, signoff, summary


def summarize_entity() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source = read_csv(TABLE_DIR / "entity_routing_author_review_300.csv")
    if len(source) != 300:
        raise ValueError("Entity source must contain 300 rows")
    audits = validate_audits(AUDIT_DIR / "entity_independent_audit_300.jsonl", 300)

    output: list[dict[str, Any]] = []
    strata: dict[str, Counter[str]] = defaultdict(Counter)
    for row in source:
        key = f"{row['model']}::{row['qid']}::{row['fact_id']}"
        audit = audits[key]["audit"]
        active = bool(row["active_entity_keys"].strip())
        record = {
            "key": key,
            "model": row["model"],
            "qid": row["qid"],
            "question_type": row["question_type"],
            "fact_id": row["fact_id"],
            "fact_text": row["fact_text"],
            "audit_stratum": row["audit_stratum"],
            "extracted_entities": row["extracted_entities"],
            "active_entity_keys": row["active_entity_keys"],
            "has_active_entity": active,
            "independent_expected_entities": " | ".join(audit["expected_entities"]),
            "independent_precision_verdict": audit["precision_verdict"],
            "independent_recall_verdict": audit["recall_verdict"],
            "confidence": audit["confidence"],
            "independent_notes": audit["notes"],
        }
        output.append(record)
        strata[row["audit_stratum"]][f"precision_{audit['precision_verdict']}"] += 1
        strata[row["audit_stratum"]][f"recall_{audit['recall_verdict']}"] += 1

    active_rows = [row for row in output if row["has_active_entity"]]
    precision_exceptions = [
        row for row in active_rows if row["independent_precision_verdict"] != "PASS"
    ]
    signoff = [{
        **row,
        "author_precision_verdict": "",
        "author_expected_entities": "",
        "author_notes": "",
    } for row in precision_exceptions]
    precision_pass = sum(row["independent_precision_verdict"] == "PASS" for row in active_rows)
    recall_pass = sum(row["independent_recall_verdict"] == "PASS" for row in active_rows)
    recall_partial = sum(row["independent_recall_verdict"] == "PARTIAL" for row in active_rows)
    summary = {
        "rows": 300,
        "selection_frozen_before_independent_review": True,
        "active_key_rows": len(active_rows),
        "active_precision_pass": precision_pass,
        "active_precision_pass_rate": precision_pass / len(active_rows),
        "active_precision_author_signoff_required": len(signoff),
        "active_full_recall_pass": recall_pass,
        "active_partial_recall": recall_partial,
        "active_recall_fail": len(active_rows) - recall_pass - recall_partial,
        "by_stratum": {key: dict(value) for key, value in sorted(strata.items())},
    }
    return output, signoff, summary


def summarize_judge() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source = read_csv(TABLE_DIR / "judge_calibration_author_review_120.csv")
    if len(source) != 120:
        raise ValueError("Judge source must contain 120 rows")
    audits = validate_audits(AUDIT_DIR / "judge_independent_audit_120.jsonl", 120)

    output: list[dict[str, Any]] = []
    signoff: list[dict[str, Any]] = []
    for row in source:
        key = f"{row['qid']}::k{row['cutoff']}"
        audit = audits[key]["audit"]
        disagreement = row["audit_group"] == "disagreement"
        if disagreement:
            codex_preferred = "public" if key in PUBLIC_PREFERRED else "strict"
            codex_label = row[f"{codex_preferred}_label"]
            ambiguous = key in AMBIGUOUS_DISAGREEMENTS
            basis = "prior_independent_disagreement_adjudication"
        else:
            if row["strict_label"] != row["public_label"]:
                raise ValueError(f"{key}: agreement row has different labels")
            codex_preferred = "both"
            codex_label = row["strict_label"]
            ambiguous = key in AMBIGUOUS_AGREEMENTS
            basis = "strict_public_agreement_plus_independent_review"
        record = {
            "key": key,
            "qid": row["qid"],
            "cutoff": row["cutoff"],
            "question_type": row["question_type"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "generated_answer": row["generated_answer"],
            "audit_group": row["audit_group"],
            "selection_stratum": row["selection_stratum"],
            "strict_label": row["strict_label"],
            "public_label": row["public_label"],
            "qwen_adjudicated_label": audit["adjudicated_label"],
            "qwen_preferred_judge": audit["preferred_judge"],
            "qwen_confidence": audit["confidence"],
            "qwen_error_type": audit["error_type"],
            "codex_independent_adjudicated_label": codex_label,
            "codex_independent_preferred_judge": codex_preferred,
            "adjudication_basis": basis,
            "stable_without_author_signoff": not ambiguous,
            "author_signoff_required": ambiguous,
            "qwen_notes": audit["notes"],
        }
        output.append(record)
        if ambiguous:
            signoff.append({
                **record,
                "author_adjudicated_label": "",
                "author_preferred_judge": "",
                "author_error_type": "",
                "author_notes": "",
            })

    groups = Counter(row["audit_group"] for row in output)
    stable = [row for row in output if row["stable_without_author_signoff"]]
    summary = {
        "rows": 120,
        "selection_frozen_before_independent_review": True,
        "agreement_rows": groups["agreement"],
        "disagreement_rows": groups["disagreement"],
        "stable_without_author_signoff": len(stable),
        "author_signoff_required": len(signoff),
        "stable_strict_preferred": sum(
            row["codex_independent_preferred_judge"] == "strict" for row in stable
        ),
        "stable_public_preferred": sum(
            row["codex_independent_preferred_judge"] == "public" for row in stable
        ),
        "stable_both_judges_agree": sum(
            row["codex_independent_preferred_judge"] == "both" for row in stable
        ),
        "qwen_matches_codex_label": sum(
            row["qwen_adjudicated_label"] == row["codex_independent_adjudicated_label"]
            for row in output
        ),
    }
    return output, signoff, summary


def main() -> None:
    temporal, temporal_signoff, temporal_summary = summarize_temporal()
    entity, entity_signoff, entity_summary = summarize_entity()
    judge, judge_signoff, judge_summary = summarize_judge()

    write_csv(OUTPUT_DIR / "temporal_independent_review_249.csv", temporal)
    write_csv(OUTPUT_DIR / "temporal_author_signoff_required_249.csv", temporal_signoff)
    write_csv(OUTPUT_DIR / "entity_independent_review_300.csv", entity)
    write_csv(OUTPUT_DIR / "entity_author_signoff_required_300.csv", entity_signoff)
    write_csv(OUTPUT_DIR / "judge_independent_review_120.csv", judge)
    write_csv(OUTPUT_DIR / "judge_author_signoff_required_120.csv", judge_signoff)

    summary = {
        "created": "2026-07-22",
        "provenance": {
            "human_gold": False,
            "description": (
                "Frozen stratified samples independently reviewed with Qwen3-30B "
                "and Codex-assisted adjudication; unresolved rows require author sign-off."
            ),
        },
        "temporal": temporal_summary,
        "entity_routing": entity_summary,
        "judge_calibration": judge_summary,
    }
    output_path = OUTPUT_DIR / "expanded_semantic_audit_summary.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
