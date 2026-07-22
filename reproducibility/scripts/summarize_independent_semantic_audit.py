#!/usr/bin/env python3
"""Validate and summarize the revision's independent semantic audits.

The generated records are explicitly model/Codex-assisted review artifacts,
not author-verified human gold labels.  Rows that still need author judgment
are emitted separately so manuscript claims cannot silently outrun evidence.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPRO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = REPRO_ROOT / "results/semantic_audit"
RAW_DIR = AUDIT_DIR / "raw"
TABLE_DIR = AUDIT_DIR / "source"
JUDGE_SOURCE = (
    REPRO_ROOT / "results/mem0_corrected"
    / "top50_top200_control/per_question_majority.csv"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def split_ids(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [item for item in value.split(";") if item]
    return [str(item) for item in value if item]


def compact_fact_diff(row: dict[str, str], ids: set[str]) -> str:
    facts = {item["fact_id"]: item["fact_text"] for item in json.loads(row["candidate_facts_json"])}
    return " || ".join(f"{fact_id}: {facts.get(fact_id, '<missing>')}" for fact_id in sorted(ids))


def temporal_review() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source = read_csv(TABLE_DIR / "temporal_conflict_author_review_79.csv")
    audits = read_jsonl(RAW_DIR / "temporal_independent_audit.jsonl")
    by_key = {row["key"]: row for row in audits}
    if len(source) != 79 or len(by_key) != 79 or any(row.get("error") for row in audits):
        raise ValueError("Temporal audit is incomplete or contains request errors")

    output: list[dict[str, Any]] = []
    signoff: list[dict[str, Any]] = []
    for row in source:
        key = f"{row['benchmark']}::{row['qid']}"
        audit = by_key[key]["audit"]
        candidate_ids = {item["fact_id"] for item in json.loads(row["candidate_facts_json"])}
        proposed_gold = set(split_ids(row["proposed_gold_fact_ids"]))
        proposed_conflict = set(split_ids(row["proposed_conflict_fact_ids"]))
        audit_gold = set(split_ids(audit["gold_fact_ids"]))
        audit_conflict = set(split_ids(audit["conflict_fact_ids"]))
        unknown = (audit_gold | audit_conflict) - candidate_ids
        if unknown:
            raise ValueError(f"{key}: audit emitted unknown fact IDs: {sorted(unknown)}")
        overlap = audit_gold & audit_conflict

        type_agree = row["proposed_temporal_type"] == audit["temporal_type"]
        gold_agree = proposed_gold == audit_gold
        conflict_agree = proposed_conflict == audit_conflict
        needs_signoff = not (
            audit.get("proposed_label_valid") is True
            and audit.get("confidence") == "high"
            and type_agree and gold_agree and conflict_agree and not overlap
        )
        record = {
            "key": key,
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "proposed_temporal_type": row["proposed_temporal_type"],
            "independent_temporal_type": audit["temporal_type"],
            "proposed_gold_fact_ids": ";".join(sorted(proposed_gold)),
            "independent_gold_fact_ids": ";".join(sorted(audit_gold)),
            "proposed_conflict_fact_ids": ";".join(sorted(proposed_conflict)),
            "independent_conflict_fact_ids": ";".join(sorted(audit_conflict)),
            "temporal_type_valid": audit.get("temporal_type_valid"),
            "gold_ids_valid": audit.get("gold_ids_valid"),
            "conflict_ids_valid": audit.get("conflict_ids_valid"),
            "proposed_label_valid": audit.get("proposed_label_valid"),
            "confidence": audit.get("confidence"),
            "type_exact_match": type_agree,
            "gold_ids_exact_match": gold_agree,
            "conflict_ids_exact_match": conflict_agree,
            "independent_gold_conflict_overlap": ";".join(sorted(overlap)),
            "author_signoff_required": needs_signoff,
            "independent_notes": audit.get("notes", ""),
        }
        output.append(record)
        if needs_signoff:
            changed = proposed_gold ^ audit_gold | proposed_conflict ^ audit_conflict
            signoff.append({
                **record,
                "changed_fact_texts": compact_fact_diff(row, changed),
                "author_temporal_type": "",
                "author_gold_fact_ids": "",
                "author_conflict_fact_ids": "",
                "author_notes": "",
            })

    summary = {
        "rows": len(output),
        "high_confidence": sum(row["confidence"] == "high" for row in output),
        "provisional_fully_upheld": sum(not row["author_signoff_required"] for row in output),
        "author_signoff_required": len(signoff),
        "type_exact_match": sum(row["type_exact_match"] for row in output),
        "gold_ids_exact_match": sum(row["gold_ids_exact_match"] for row in output),
        "conflict_ids_exact_match": sum(row["conflict_ids_exact_match"] for row in output),
    }
    return output, signoff, summary


def entity_review() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source = read_csv(TABLE_DIR / "entity_routing_author_review_200.csv")
    audits = read_jsonl(RAW_DIR / "entity_independent_audit.jsonl")
    by_key = {row["key"]: row for row in audits}
    if len(source) != 200 or len(by_key) != 200 or any(row.get("error") for row in audits):
        raise ValueError("Entity audit is incomplete or contains request errors")

    output: list[dict[str, Any]] = []
    strata: dict[str, Counter[str]] = defaultdict(Counter)
    for row in source:
        key = f"{row['model']}::{row['qid']}::{row['fact_id']}"
        audit = by_key[key]["audit"]
        precision = audit["precision_verdict"]
        recall = audit["recall_verdict"]
        stratum = row["audit_stratum"]
        strata[stratum][f"precision_{precision}"] += 1
        strata[stratum][f"recall_{recall}"] += 1
        output.append({
            "key": key,
            "model": row["model"],
            "qid": row["qid"],
            "question_type": row["question_type"],
            "fact_id": row["fact_id"],
            "fact_text": row["fact_text"],
            "audit_stratum": stratum,
            "extracted_entities": row["extracted_entities"],
            "active_entity_keys": row["active_entity_keys"],
            "independent_expected_entities": " | ".join(audit["expected_entities"]),
            "independent_precision_verdict": precision,
            "independent_recall_verdict": recall,
            "confidence": audit["confidence"],
            "independent_notes": audit["notes"],
        })

    active = [row for row in output if row["active_entity_keys"].strip()]
    precision_pass = sum(row["independent_precision_verdict"] == "PASS" for row in active)
    recall_nonzero = sum(row["independent_recall_verdict"] in {"PASS", "PARTIAL"} for row in active)
    precision_failures = [row for row in active if row["independent_precision_verdict"] != "PASS"]
    ambiguous_recall = [row for row in active if row["independent_recall_verdict"] == "FAIL"]
    signoff_pool = precision_failures + [row for row in ambiguous_recall if row not in precision_failures]
    signoff = []
    for row in signoff_pool:
        signoff.append({
            **row,
            "author_precision_verdict": "",
            "author_recall_verdict": "",
            "author_expected_entities": "",
            "author_notes": "",
        })
    summary = {
        "rows": len(output),
        "active_key_rows": len(active),
        "active_precision_pass": precision_pass,
        "active_precision_pass_rate": precision_pass / len(active),
        "active_recall_pass_or_partial": recall_nonzero,
        "active_recall_pass_or_partial_rate": recall_nonzero / len(active),
        "active_full_recall_pass": sum(row["independent_recall_verdict"] == "PASS" for row in active),
        "author_signoff_required": len(signoff),
        "by_stratum": {key: dict(value) for key, value in sorted(strata.items())},
    }
    return output, signoff, summary


PUBLIC_PREFERRED = {
    "35a27287::k50",
    "54026fce::k50", "54026fce::k200",
    "d24813b1::k50", "d24813b1::k200",
    "c4a1ceb8::k50", "c4a1ceb8::k200",
}

AMBIGUOUS_JUDGE_KEYS = {
    "06f04340::k50", "06f04340::k200",
    "35a27287::k50",
    "54026fce::k50", "54026fce::k200",
    "55241a1f::k50",
    "d24813b1::k50", "d24813b1::k200",
}


def judge_review() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    audits = read_jsonl(RAW_DIR / "judge_independent_audit.jsonl")
    by_key = {row["key"]: row for row in audits}
    source: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in read_csv(JUDGE_SOURCE):
        source[f"{row['qid']}::k{row['cutoff']}"][row["judge_arm"]] = row
    disagreement_keys = {
        key for key, arms in source.items()
        if set(arms) == {"strict", "public"}
        and arms["strict"]["majority_label"] != arms["public"]["majority_label"]
    }
    if len(disagreement_keys) != 40 or set(by_key) != disagreement_keys:
        raise ValueError("Judge audit does not exactly cover the 40 disagreements")
    if any(row.get("error") for row in audits):
        raise ValueError("Judge audit contains request errors")

    output: list[dict[str, Any]] = []
    signoff: list[dict[str, Any]] = []
    for key in sorted(disagreement_keys):
        arms = source[key]
        base = arms["strict"]
        preferred = "public" if key in PUBLIC_PREFERRED else "strict"
        adjudicated = arms[preferred]["majority_label"]
        audit = by_key[key]["audit"]
        record = {
            "key": key,
            "qid": base["qid"],
            "cutoff": base["cutoff"],
            "question_type": base["question_type"],
            "question": base["question"],
            "gold_answer": base["gold_answer"],
            "generated_answer": base["generated_answer"],
            "strict_label": arms["strict"]["majority_label"],
            "public_label": arms["public"]["majority_label"],
            "qwen_preferred_judge": audit["preferred_judge"],
            "qwen_adjudicated_label": audit["adjudicated_label"],
            "codex_independent_preferred_judge": preferred,
            "codex_independent_adjudicated_label": adjudicated,
            "author_signoff_required": key in AMBIGUOUS_JUDGE_KEYS,
            "qwen_confidence": audit["confidence"],
            "qwen_error_type": audit["error_type"],
            "qwen_notes": audit["notes"],
        }
        output.append(record)
        if record["author_signoff_required"]:
            signoff.append({
                **record,
                "author_preferred_judge": "",
                "author_adjudicated_label": "",
                "author_notes": "",
            })

    public_only = [row for row in output if row["public_label"] == "CORRECT"]
    strict_only = [row for row in output if row["strict_label"] == "CORRECT"]
    summary = {
        "rows": len(output),
        "codex_prefers_strict": sum(row["codex_independent_preferred_judge"] == "strict" for row in output),
        "codex_prefers_public": sum(row["codex_independent_preferred_judge"] == "public" for row in output),
        "author_signoff_required": len(signoff),
        "public_only_accepts": len(public_only),
        "public_only_accepts_upheld": sum(row["codex_independent_preferred_judge"] == "public" for row in public_only),
        "strict_only_accepts": len(strict_only),
        "strict_only_accepts_upheld": sum(row["codex_independent_preferred_judge"] == "strict" for row in strict_only),
    }
    return output, signoff, summary


def main() -> None:
    temporal, temporal_signoff, temporal_summary = temporal_review()
    entity, entity_signoff, entity_summary = entity_review()
    judge, judge_signoff, judge_summary = judge_review()

    write_csv(AUDIT_DIR / "temporal_independent_review.csv", temporal)
    write_csv(AUDIT_DIR / "temporal_author_signoff_required.csv", temporal_signoff)
    write_csv(AUDIT_DIR / "entity_independent_review.csv", entity)
    write_csv(AUDIT_DIR / "entity_author_signoff_required.csv", entity_signoff)
    write_csv(AUDIT_DIR / "judge_independent_review.csv", judge)
    write_csv(AUDIT_DIR / "judge_author_signoff_required.csv", judge_signoff)

    summary = {
        "provenance": {
            "semantic_reviewer": "Qwen3-30B-A3B-Instruct-2507-FP8",
            "policy_adjudication": "Codex independent review",
            "human_gold": False,
            "generated_at": "2026-07-22",
        },
        "temporal": temporal_summary,
        "entity_routing": entity_summary,
        "judge_disagreements": judge_summary,
    }
    (AUDIT_DIR / "semantic_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
