#!/usr/bin/env python3
"""Normalize native-unit, fully expanded Qwen MemForest-Embed answers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED = {"longmemeval": 500, "locomo": 1986}
NATIVE_TOP_K = 10
MAX_CONTEXT_CHARS = 60_000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    output: list[dict[str, Any]] = []
    sources: dict[str, dict[str, Any]] = {}
    answer_path = args.answers.resolve()
    answer_rows = read_jsonl(answer_path)
    retrieval_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    for benchmark in EXPECTED:
        rows = [row for row in answer_rows if str(row.get("benchmark")) == benchmark]
        if len(rows) != EXPECTED[benchmark]:
            raise ValueError(
                f"{benchmark}: expected {EXPECTED[benchmark]} rows, found {len(rows)}"
            )
        qids = [str(row["qid"]) for row in rows]
        if len(set(qids)) != len(qids):
            raise ValueError(f"{benchmark}: duplicate qids")
        for row in rows:
            if str(row.get("method")) != "memforest_embed_browse":
                raise ValueError(f"{benchmark}/{row['qid']}: unexpected method")
            if str(row.get("prompt_version")) != "memforest_default_v1":
                raise ValueError(f"{benchmark}/{row['qid']}: unexpected answer prompt")
            if row.get("error"):
                raise ValueError(f"{benchmark}/{row['qid']}: answer error: {row['error']}")
            responses = list(row.get("responses") or [])
            if len(responses) != 1 or int(responses[0].get("sample_index") or 0) != 0:
                raise ValueError(f"{benchmark}/{row['qid']}: not deterministic pass@1")
            answer = str(responses[0].get("answer") or "").strip()
            if not answer:
                raise ValueError(f"{benchmark}/{row['qid']}: blank answer")
            context_chars = int(row.get("context_chars") or -1)
            if context_chars < 0 or context_chars > MAX_CONTEXT_CHARS:
                raise ValueError(f"{benchmark}/{row['qid']}: invalid context length")

            retrieval_path = Path(str(row.get("source_path") or "")).resolve()
            if retrieval_path not in retrieval_cache:
                retrieval_cache[retrieval_path] = {
                    str(item["qid"]): item for item in read_jsonl(retrieval_path)
                }
            retrieval_rows = retrieval_cache[retrieval_path]
            retrieval = retrieval_rows.get(str(row["qid"]))
            if retrieval is None:
                raise ValueError(f"{benchmark}/{row['qid']}: retrieval row missing")
            fact_ids = list(retrieval.get("fact_ids") or [])
            if (
                not fact_ids
                or int(retrieval.get("n_facts") or -1) != len(fact_ids)
                or len(set(fact_ids)) != len(fact_ids)
            ):
                raise ValueError(f"{benchmark}/{row['qid']}: invalid expanded facts")
            output.append(
                {
                    "model": args.model,
                    "method": "memforest_embed_browse",
                    "sample_index": 0,
                    "benchmark": benchmark,
                    "qid": str(row["qid"]),
                    "question_type": str(row["question_type"]),
                    "question": str(row["question"]),
                    "gold_answer": str(row["gold_answer"]),
                    "generated_answer": answer,
                    "answer_prompt": "memforest_default_v1",
                    "retrieval_unit": "memtree_browse_unit",
                    "native_top_k": NATIVE_TOP_K,
                    "context_expansion": "full",
                    "expanded_tree_count": int(retrieval["n_recalled_trees"]),
                    "expanded_fact_count": len(fact_ids),
                    "context_chars": context_chars,
                }
            )
        retrieval_sources = sorted({Path(row["source_path"]).resolve() for row in rows})
        sources[benchmark] = {
            "answer_source": args.answers.name,
            "answer_sha256": sha256(answer_path),
            "retrieval_sources": [path.name for path in retrieval_sources],
            "retrieval_sha256": [sha256(path) for path in retrieval_sources],
            "rows": len(rows),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in sorted(output, key=lambda item: (item["benchmark"], item["qid"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "protocol_id": "qwen_memforest_embed_native_top10_full_expand_v2_20260731",
        "model": args.model,
        "answer_prompt": "memforest_default_v1",
        "retrieval_unit": "memtree_browse_unit",
        "native_top_k": NATIVE_TOP_K,
        "context_expansion": "full",
        "max_context_chars": MAX_CONTEXT_CHARS,
        "rows": len(output),
        "sources": sources,
        "output_sha256": sha256(args.out),
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
