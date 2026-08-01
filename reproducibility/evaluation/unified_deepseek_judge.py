#!/usr/bin/env python3
"""Unified DeepSeek judge for Gemma baseline/MemForest response files.

This script intentionally judges existing responses only. It does not rebuild
memory, rerun retrieval, or regenerate answers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI


ACCURACY_PROMPT = """
Your task is to label an answer to a question as CORRECT or WRONG. You will be given:
(1) a question,
(2) a gold answer,
(3) a generated answer.

Be generous with grading: as long as the generated answer touches the same fact, event, person, place, item, or time reference as the gold answer, count it as CORRECT.

For time-related questions, be especially generous:
- treat equivalent date formats as CORRECT, such as "May 7th, 2023" vs "7 May 2023"
- treat relative and absolute time references as CORRECT if they refer to the same date or time period
- treat a more specific answer as CORRECT when it clearly instantiates the same time period as the gold answer

Do not penalize extra explanation, markdown formatting, or a short reasoning trace if the underlying answer is correct.
Only label WRONG when the generated answer contradicts the gold answer, misses the key fact, answers a different question, or says the information is unavailable when the answer is actually present.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Return JSON only with a single key "label" whose value is either "CORRECT" or "WRONG".
""".strip()


APPENDIX_LONGMEMEVAL_PROMPT = """
Your task is to label an answer to a LongMemEval question as CORRECT or WRONG.

You will be given:
(1) question type
(2) question
(3) gold answer
(4) generated answer

Be generous with grading:
(1) accept semantically equivalent answers;
(2) accept equivalent relative and absolute time expressions;
(3) accept more specific but consistent answers;
(4) mark WRONG only for contradiction, missing key facts, answering a different question, or incorrect abstention.

Question type: {question_type}
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Return JSON only with {{"label": "CORRECT"}} or {{"label": "WRONG"}}.
""".strip()


APPENDIX_LOCOMO_PROMPT = """
Your task is to label an answer to a question as CORRECT or WRONG.

You will be given:
(1) question
(2) gold answer
(3) generated answer

Be generous with grading:
(1) accept semantically equivalent answers;
(2) accept equivalent relative and absolute time expressions;
(3) accept more specific but consistent answers;
(4) mark WRONG only for contradiction, missing key facts, non-answer, or incorrect abstention.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Return JSON only with {{"label": "CORRECT"}} or {{"label": "WRONG"}}.
""".strip()


DEFAULT_TARGETS = {
    "mem0": {
        "longmemeval": [
            "gemma_8001_full_runs/full_20260618_015525_tmux_mem0shards64_memoryos_lightmem/mem0_sharded_v2/mem0/shards/*/output/answer_results.json",
        ],
        "locomo": [
            "gemma_8001_supplement_runs/supp_20260618_142020_mem0_memoryos_shards64_workers1/mem0_locomo64/mem0_locomo/shards/*/output/answer_results.json",
            "gemma_8001_supplement_runs/supp_20260618_142020_mem0_memoryos_shards64_workers1/mem0_locomo64_recover_shard019_20260618_210400_retry4_8006/mem0_locomo/shards/*/output/answer_results.json",
        ],
    },
    "evermemos": {
        "longmemeval": [
            "gemma_8001_supplement_runs/evermemos_longmemeval_gemma_8001_20260619_231558/evermemos_longmemeval/answer_results.json",
        ],
        "locomo": [
            "gemma_8001_supplement_runs/evermemos_locomo_gemma_8001_20260619_203418/evermemos_locomo/answer_results.json",
        ],
    },
    "lightmem": {
        "longmemeval": [
            "gemma_8001_full_runs/full_20260618_015525_tmux_mem0shards64_memoryos_lightmem/lightmem/lightmem/longmemeval_s_cleaned_google__gemma-4-12B-it_20260618_015525/result_*.json",
        ],
        "locomo": [
            "gemma_8001_supplement_runs/supp_20260618_141439_recovery_after_8001_8003_restart/lightmem_locomo/lightmem_locomo/results/sample_*.json",
        ],
    },
    "memoryos": {
        "longmemeval": [
            "gemma_8001_supplement_runs/supp_20260618_142020_mem0_memoryos_shards64_workers1/memoryos_lme64_fixed/memoryos_longmemeval/shards/*/output/item_*.json",
        ],
        "locomo": [
            "gemma_8001_supplement_runs/supp_20260618_142020_mem0_memoryos_shards64_workers1/memoryos_locomo64_v2/memoryos_locomo/shards/*/output/all_loco_results_gemma.json",
        ],
    },
    "mempalace": {
        "longmemeval": [
            "gemma_8001_supplement_runs/mempalace_longmemeval_gemma_8001_20260622_170723/answers_gemma/answers.jsonl",
        ],
        "locomo": [
            "gemma_8001_supplement_runs/mempalace_locomo_gemma_8001_20260622_161827/answers_gemma/answers.jsonl",
        ],
    },
    "memforest_agent": {
        "longmemeval": [
            "gemma_8001_supplement_runs/memforest_longmemeval_gemma_8001_full500_20260620_165747/memforest/eval_500q_gemma_8001/per_question.jsonl",
            "gemma_8001_unified_deepseek_judge_20260623/recovery_20260623/memforest_agent_lme_missing_answer/per_question.jsonl",
        ],
        "locomo": [
            "gemma_8001_supplement_runs/memforest_locomo_gemma_8001_20260622_150209/memforest/eval_locomo_gemma_8001/per_question.jsonl",
            "gemma_8001_unified_deepseek_judge_20260623/recovery_20260623/memforest_agent_locomo_missing/per_question.jsonl",
            "gemma_8001_cat5_runs/cat5_20260623/memforest_agent_cat5/per_question.jsonl",
        ],
    },
    "memforest_embed_browse": {
        "longmemeval": [
            "gemma_8001_supplement_runs/memforest_longmemeval_gemma_8001_full500_20260620_165747/memforest/eval_500q_gemma_8001_embed_browse_answer/per_question.jsonl",
        ],
        "locomo": [
            "gemma_8001_supplement_runs/memforest_locomo_gemma_8001_20260622_150209/memforest/eval_locomo_gemma_8001_embed_browse_answer/per_question.jsonl",
            "gemma_8001_cat5_runs/cat5_20260623/memforest_embed_cat5_answer/per_question.jsonl",
        ],
    },
}


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def expand_patterns(root: Path, patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    return sorted(set(files))


def load_datasets(root: Path) -> dict[str, Any]:
    lme = read_json(root / "data/longmemeval_s_cleaned.json")
    locomo = read_json(root / "data/locomo10_real.json")
    lme_by_id = {str(r["question_id"]): r for r in lme}
    locomo_by_id = {str(r["question_id"]): r for r in locomo}
    original_locomo_path = root / "data/locomo10.json"
    if original_locomo_path.exists():
        original_locomo = read_json(original_locomo_path)
        for sample in original_locomo:
            sid = str(sample.get("sample_id") or "")
            if not sid:
                continue
            for qa_idx, qa in enumerate(sample.get("qa", [])):
                qid = f"{sid}_qa{qa_idx:03d}"
                row = locomo_by_id.get(qid)
                if not row:
                    continue
                adversarial_answer = qa.get("adversarial_answer")
                if adversarial_answer is not None:
                    row["adversarial_answer"] = adversarial_answer
                    if not str(row.get("answer") or "").strip():
                        row["answer"] = adversarial_answer
    sample_order: list[str] = []
    for r in locomo:
        sid = str(r.get("locomo_sample_id"))
        if sid and sid not in sample_order:
            sample_order.append(sid)
    locomo_by_internal_id: dict[str, dict] = {}
    for r in locomo:
        sid = str(r.get("locomo_sample_id"))
        if sid in sample_order:
            sample_idx = sample_order.index(sid)
            qa_idx = int(r.get("locomo_qa_idx", -1))
            locomo_by_internal_id[f"locomo_{sample_idx}_qa{qa_idx}"] = r
    locomo_by_qgold: dict[tuple[str, str], list[dict]] = defaultdict(list)
    lme_by_qgold: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in lme:
        lme_by_qgold[(norm_text(r.get("question")), norm_text(r.get("answer")))].append(r)
    for r in locomo:
        locomo_by_qgold[(norm_text(r.get("question")), norm_text(r.get("answer")))].append(r)
    return {
        "longmemeval": lme,
        "locomo": locomo,
        "longmemeval_by_id": lme_by_id,
        "locomo_by_id": locomo_by_id,
        "locomo_by_internal_id": locomo_by_internal_id,
        "longmemeval_by_qgold": lme_by_qgold,
        "locomo_by_qgold": locomo_by_qgold,
    }


def resolve_dataset_row(
    benchmark: str,
    row: dict,
    datasets: dict[str, Any],
    *,
    sample_id: str | None = None,
    question_index: int | None = None,
) -> dict | None:
    qid = row.get("qid") or row.get("question_id")
    by_id = datasets[f"{benchmark}_by_id"]
    if qid and str(qid) in by_id:
        return by_id[str(qid)]
    if benchmark == "locomo" and qid and str(qid) in datasets["locomo_by_internal_id"]:
        return datasets["locomo_by_internal_id"][str(qid)]
    if sample_id and question_index is not None:
        qid2 = f"{sample_id}_qa{question_index:03d}"
        ds = by_id.get(qid2)
        if ds:
            return ds

    question = row.get("question")
    gold = (
        row.get("gold_answer")
        or row.get("golden_answer")
        or row.get("ground_truth")
        or row.get("reference")
        or row.get("original_answer")
    )
    by_qgold = datasets[f"{benchmark}_by_qgold"]
    candidates = by_qgold.get((norm_text(question), norm_text(gold)), [])
    if len(candidates) == 1:
        return candidates[0]
    if sample_id and candidates:
        filtered = [r for r in candidates if r.get("locomo_sample_id") == sample_id]
        if len(filtered) == 1:
            return filtered[0]
    return candidates[0] if candidates else None


def is_locomo_eval_row(dataset_row: dict) -> bool:
    return str(dataset_row.get("question_type", "")) != "category_5"


def locomo_scope_allows(dataset_row: dict, scope: str) -> bool:
    is_cat5 = str(dataset_row.get("question_type", "")) == "category_5"
    if scope == "cat5":
        return is_cat5
    if scope == "all":
        return True
    return not is_cat5


def make_record(
    *,
    method: str,
    benchmark: str,
    dataset_row: dict,
    answer: Any,
    source_path: Path,
    source_status: str = "ok",
) -> dict:
    return {
        "method": method,
        "benchmark": benchmark,
        "qid": str(dataset_row["question_id"]),
        "question_type": dataset_row.get("question_type"),
        "question": dataset_row.get("question"),
        "gold_answer": dataset_row.get("answer"),
        "generated_answer": str(answer or "").strip(),
        "source_path": str(source_path),
        "source_status": source_status,
    }


def collect_answer_results(
    method: str,
    benchmark: str,
    files: list[Path],
    datasets: dict[str, Any],
    locomo_scope: str = "noncat5",
) -> list[dict]:
    records = []
    for path in files:
        rows = read_json(path)
        if isinstance(rows, dict):
            rows = [rows]
        for row in rows:
            ds = resolve_dataset_row(benchmark, row, datasets)
            if not ds:
                continue
            if benchmark == "locomo" and not locomo_scope_allows(ds, locomo_scope):
                continue
            records.append(make_record(
                method=method,
                benchmark=benchmark,
                dataset_row=ds,
                answer=row.get("answer"),
                source_path=path,
            ))
    return records


def collect_memoryos(
    method: str,
    benchmark: str,
    files: list[Path],
    datasets: dict[str, Any],
    locomo_scope: str = "noncat5",
) -> list[dict]:
    records = []
    for path in files:
        obj = read_json(path)
        rows = obj if isinstance(obj, list) else [obj]
        for idx, row in enumerate(rows):
            ds = resolve_dataset_row(
                benchmark,
                row,
                datasets,
                sample_id=row.get("sample_id"),
                question_index=idx if benchmark == "locomo" else None,
            )
            if not ds:
                continue
            if benchmark == "locomo" and not locomo_scope_allows(ds, locomo_scope):
                continue
            answer = row.get("answer") if benchmark == "longmemeval" else row.get("system_answer")
            records.append(make_record(
                method=method,
                benchmark=benchmark,
                dataset_row=ds,
                answer=answer,
                source_path=path,
                source_status=str(row.get("status") or "ok"),
            ))
    return records


def collect_lightmem(
    method: str,
    benchmark: str,
    files: list[Path],
    datasets: dict[str, Any],
    locomo_scope: str = "noncat5",
) -> list[dict]:
    records = []
    for path in files:
        obj = read_json(path)
        if benchmark == "longmemeval":
            ds = resolve_dataset_row(benchmark, obj, datasets)
            if not ds:
                continue
            records.append(make_record(
                method=method,
                benchmark=benchmark,
                dataset_row=ds,
                answer=obj.get("generated_answer"),
                source_path=path,
            ))
            continue
        sample_id = obj.get("sample_id")
        for row in obj.get("results", []):
            ds = resolve_dataset_row(
                benchmark,
                row,
                datasets,
                sample_id=sample_id,
                question_index=row.get("question_index"),
            )
            if not ds or not locomo_scope_allows(ds, locomo_scope):
                continue
            records.append(make_record(
                method=method,
                benchmark=benchmark,
                dataset_row=ds,
                answer=row.get("prediction"),
                source_path=path,
            ))
    return records


def collect_mempalace(
    method: str,
    benchmark: str,
    files: list[Path],
    datasets: dict[str, Any],
    locomo_scope: str = "noncat5",
) -> list[dict]:
    records = []
    for path in files:
        for row in iter_jsonl(path):
            ds = resolve_dataset_row(benchmark, row, datasets)
            if not ds:
                continue
            if benchmark == "locomo" and not locomo_scope_allows(ds, locomo_scope):
                continue
            records.append(make_record(
                method=method,
                benchmark=benchmark,
                dataset_row=ds,
                answer=row.get("model_answer"),
                source_path=path,
                source_status="error" if row.get("error") else "ok",
            ))
    return records


def collect_memforest(
    method: str,
    benchmark: str,
    files: list[Path],
    datasets: dict[str, Any],
    locomo_scope: str = "noncat5",
) -> list[dict]:
    records = []
    for path in files:
        for row in iter_jsonl(path):
            ds = resolve_dataset_row(benchmark, row, datasets)
            if not ds:
                continue
            if benchmark == "locomo" and not locomo_scope_allows(ds, locomo_scope):
                continue
            status = "skip" if row.get("skip") else "ok"
            records.append(make_record(
                method=method,
                benchmark=benchmark,
                dataset_row=ds,
                answer=row.get("model_answer") or row.get("answer"),
                source_path=path,
                source_status=status,
            ))
    return records


def dedupe_records(records: list[dict]) -> list[dict]:
    best: dict[tuple[str, str, str], dict] = {}
    rank = {"ok": 3, "None": 3, "error": 2, "skip": 1}
    for rec in records:
        key = (rec["method"], rec["benchmark"], rec["qid"])
        prev = best.get(key)
        if not prev:
            best[key] = rec
            continue
        prev_score = rank.get(str(prev.get("source_status")), 0) + bool(prev.get("generated_answer"))
        rec_score = rank.get(str(rec.get("source_status")), 0) + bool(rec.get("generated_answer"))
        if rec_score >= prev_score:
            best[key] = rec
    return sorted(best.values(), key=lambda r: (r["method"], r["benchmark"], r["qid"]))


def collect_all(
    root: Path,
    datasets: dict[str, Any],
    benchmarks: set[str],
    locomo_scope: str,
) -> tuple[list[dict], dict]:
    records = []
    manifest = {}
    collectors = {
        "mem0": collect_answer_results,
        "evermemos": collect_answer_results,
        "memoryos": collect_memoryos,
        "lightmem": collect_lightmem,
        "mempalace": collect_mempalace,
        "memforest_agent": collect_memforest,
        "memforest_embed_browse": collect_memforest,
    }
    for method, by_bench in DEFAULT_TARGETS.items():
        manifest[method] = {}
        for benchmark, patterns in by_bench.items():
            if benchmark not in benchmarks:
                continue
            files = expand_patterns(root, patterns)
            manifest[method][benchmark] = {
                "patterns": patterns,
                "files": [str(p) for p in files],
            }
            records.extend(collectors[method](method, benchmark, files, datasets, locomo_scope))
    return dedupe_records(records), manifest


def coverage_rows(records: list[dict], datasets: dict[str, Any], benchmarks: set[str], locomo_scope: str) -> list[dict]:
    expected = {
        "longmemeval": {str(r["question_id"]) for r in datasets["longmemeval"]},
        "locomo": {
            str(r["question_id"])
            for r in datasets["locomo"]
            if locomo_scope_allows(r, locomo_scope)
        },
    }
    got: dict[tuple[str, str], set[str]] = defaultdict(set)
    empty: Counter[tuple[str, str]] = Counter()
    status_counts: Counter[tuple[str, str, str]] = Counter()
    for rec in records:
        key = (rec["method"], rec["benchmark"])
        got[key].add(rec["qid"])
        if not rec.get("generated_answer"):
            empty[key] += 1
        status_counts[(rec["method"], rec["benchmark"], str(rec.get("source_status")))] += 1
    rows = []
    for method in DEFAULT_TARGETS:
        for benchmark in ("longmemeval", "locomo"):
            if benchmark not in benchmarks:
                continue
            key = (method, benchmark)
            rows.append({
                "method": method,
                "benchmark": benchmark,
                "expected": len(expected[benchmark]),
                "records": len(got[key]),
                "missing": len(expected[benchmark] - got[key]),
                "empty_answer": empty[key],
                "status_counts": {
                    status: count
                    for (m, b, status), count in status_counts.items()
                    if (m, b) == key
                },
            })
    return rows


def cache_key(judge_model: str, rec: dict, prompt_version: str) -> str:
    raw = json.dumps({
        "prompt_version": prompt_version,
        "judge_model": judge_model,
        "question": norm_text(rec["question"]),
        "gold_answer": norm_text(rec["gold_answer"]),
        "generated_answer": norm_text(rec["generated_answer"]),
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict) -> None:
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done(path: Path) -> dict[tuple[str, str, str], dict]:
    done = {}
    if not path.exists():
        return done
    for row in iter_jsonl(path):
        done[(row["method"], row["benchmark"], row["qid"])] = row
    return done


def parse_label(content: str) -> str:
    try:
        obj = json.loads(content)
        label = str(obj.get("label", "")).upper().strip()
        if label in {"CORRECT", "WRONG"}:
            return label
    except Exception:
        pass
    upper = content.upper()
    if "CORRECT" in upper and "WRONG" not in upper:
        return "CORRECT"
    return "WRONG"


def render_prompt(rec: dict, prompt_version: str) -> str:
    if prompt_version == "appendix":
        if rec["benchmark"] == "longmemeval":
            return APPENDIX_LONGMEMEVAL_PROMPT.format(
                question_type=rec.get("question_type") or "",
                question=rec["question"],
                gold_answer=rec["gold_answer"],
                generated_answer=rec["generated_answer"],
            )
        return APPENDIX_LOCOMO_PROMPT.format(
            question=rec["question"],
            gold_answer=rec["gold_answer"],
            generated_answer=rec["generated_answer"],
        )
    return ACCURACY_PROMPT.format(
        question=rec["question"],
        gold_answer=rec["gold_answer"],
        generated_answer=rec["generated_answer"],
    )


def judge_one(client: OpenAI, model: str, rec: dict, prompt_version: str, retries: int = 4) -> dict:
    if not rec.get("generated_answer"):
        return {**rec, "judge_label": "WRONG", "judge_raw": "", "judge_error": "empty_answer"}
    prompt = render_prompt(rec, prompt_version)
    last_error = ""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = resp.choices[0].message.content or "{}"
            return {
                **rec,
                "judge_label": parse_label(content),
                "judge_raw": content,
                "judge_error": "",
                "prompt_version": prompt_version,
            }
        except Exception as exc:
            last_error = str(exc)
            time.sleep(min(20, 2 ** attempt))
    return {**rec, "judge_label": "WRONG", "judge_raw": "", "judge_error": last_error, "prompt_version": prompt_version}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(judged_path: Path, summary_json: Path, summary_csv: Path, benchmarks: set[str]) -> list[dict]:
    counts: Counter[tuple[str, str, str]] = Counter()
    errors: Counter[tuple[str, str]] = Counter()
    total: Counter[tuple[str, str]] = Counter()
    for row in iter_jsonl(judged_path):
        key = (row["method"], row["benchmark"])
        total[key] += 1
        counts[(row["method"], row["benchmark"], row.get("judge_label", "WRONG"))] += 1
        if row.get("judge_error"):
            errors[key] += 1
    rows = []
    for method in DEFAULT_TARGETS:
        for benchmark in ("longmemeval", "locomo"):
            if benchmark not in benchmarks:
                continue
            key = (method, benchmark)
            n = total[key]
            correct = counts[(method, benchmark, "CORRECT")]
            wrong = counts[(method, benchmark, "WRONG")]
            rows.append({
                "method": method,
                "benchmark": benchmark,
                "n": n,
                "correct": correct,
                "wrong": wrong,
                "accuracy": (correct / n) if n else None,
                "judge_errors": errors[key],
            })
    write_json(summary_json, rows)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--judge-url", default="https://api.deepseek.com/v1")
    ap.add_argument("--judge-model", default="deepseek-chat")
    ap.add_argument("--judge-api-key", default=os.getenv("DEEPSEEK_API_KEY"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--prompt-version", choices=("unified", "appendix"), default="unified")
    ap.add_argument("--locomo-scope", choices=("noncat5", "cat5", "all"), default="noncat5")
    ap.add_argument("--benchmarks", nargs="+", choices=("longmemeval", "locomo"), default=("longmemeval", "locomo"))
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args()

    if not args.judge_api_key and not args.collect_only:
        raise SystemExit("Missing --judge-api-key or DEEPSEEK_API_KEY")

    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = set(args.benchmarks)
    datasets = load_datasets(root)
    records, manifest = collect_all(root, datasets, benchmarks, args.locomo_scope)
    coverage = coverage_rows(records, datasets, benchmarks, args.locomo_scope)
    write_json(out_dir / "input_manifest.json", manifest)
    write_json(out_dir / "coverage.json", coverage)
    write_jsonl(out_dir / "records.jsonl", records)
    write_json(out_dir / "judge_prompt_manifest.json", {
        "prompt_version": args.prompt_version,
        "benchmarks": sorted(benchmarks),
        "locomo_scope": args.locomo_scope,
        "source": "reproducibility/evaluation/unified_deepseek_judge.py" if args.prompt_version == "appendix" else "MemoryForest/scripts/rerun_answer_passk.py",
        "longmemeval_prompt": APPENDIX_LONGMEMEVAL_PROMPT if args.prompt_version == "appendix" else ACCURACY_PROMPT,
        "locomo_prompt": APPENDIX_LOCOMO_PROMPT if args.prompt_version == "appendix" else ACCURACY_PROMPT,
    })

    if args.collect_only:
        print(json.dumps(coverage, ensure_ascii=False, indent=2))
        return

    judged_path = out_dir / "judged.jsonl"
    done = load_done(judged_path)
    todo = [r for r in records if (r["method"], r["benchmark"], r["qid"]) not in done]
    writer = JsonlWriter(judged_path)
    client = OpenAI(api_key=args.judge_api_key, base_url=args.judge_url)

    print(f"records={len(records)} done={len(done)} todo={len(todo)} workers={args.workers} prompt={args.prompt_version}")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(judge_one, client, args.judge_model, rec, args.prompt_version) for rec in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            writer.append(row)
            if i % 100 == 0 or i == len(futs):
                print(f"judged {i}/{len(futs)}")

    summarize(judged_path, out_dir / "summary.json", out_dir / "summary.csv", benchmarks)


if __name__ == "__main__":
    main()
