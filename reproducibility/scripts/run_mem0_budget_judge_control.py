#!/usr/bin/env python3
"""Run the frozen Mem0 top-50/top-200 x strict/public-judge control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import runpy
import statistics
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI
from qdrant_client import QdrantClient, models


CUTOFFS = (50, 200)
JUDGE_ARMS = ("strict", "public")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def shard_number(path: Path) -> int:
    return int(path.name.rsplit("_", 1)[1])


def load_frozen_metadata(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    old_searches: dict[str, dict[str, Any]] = {}
    source_files = []
    shard_dirs = sorted(
        [path for path in run_root.glob("shard_*") if path.is_dir()],
        key=shard_number,
    )
    for shard in shard_dirs:
        answer_path = shard / "answer_results.json"
        search_path = shard / "search_results.json"
        answers = json.loads(answer_path.read_text(encoding="utf-8"))
        searches = json.loads(search_path.read_text(encoding="utf-8"))
        if len(answers) != len(searches):
            raise ValueError(f"count mismatch in {shard}")
        for search in searches:
            conversation_id = str(search["conversation_id"])
            if conversation_id in old_searches:
                raise ValueError(f"duplicate conversation: {conversation_id}")
            old_searches[conversation_id] = search
        for answer in answers:
            conversation_id = str(answer["conversation_id"])
            old_search = old_searches[conversation_id]
            if old_search["query"] != answer["question"]:
                raise ValueError(f"question mismatch: {conversation_id}")
            old_results = old_search.get("results") or []
            if len(old_results) != 20:
                raise ValueError(f"{conversation_id}: expected frozen top-20")
            rows.append(
                {
                    "qid": str(answer["question_id"]),
                    "conversation_id": conversation_id,
                    "question": str(answer["question"]),
                    "gold_answer": str(answer["golden_answer"]),
                    "question_type": str(answer.get("category") or ""),
                    "metadata": answer.get("metadata") or {},
                    "old_top20_ids": [
                        str((result.get("metadata") or {}).get("id"))
                        for result in old_results
                    ],
                    "old_top20_scores": [float(result["score"]) for result in old_results],
                    "old_top20_contents": [str(result["content"]) for result in old_results],
                }
            )
        source_files.extend(
            [
                {"path": str(answer_path.resolve()), "sha256": sha256(answer_path)},
                {"path": str(search_path.resolve()), "sha256": sha256(search_path)},
            ]
        )
    rows.sort(key=lambda row: int(row["conversation_id"].rsplit("_", 1)[1]))
    if len(rows) != 500:
        raise ValueError(f"expected 500 frozen rows, got {len(rows)}")
    if len({row["qid"] for row in rows}) != 500:
        raise ValueError("frozen qids are not unique")
    validation = {
        "rows": len(rows),
        "shards": len(shard_dirs),
        "question_types": dict(Counter(row["question_type"] for row in rows)),
        "source_files": source_files,
    }
    return rows, validation


def embedding_batches(
    questions: list[str],
    *,
    base_url: str,
    model: str,
    api_key: str,
    dimensions: int,
    workers: int,
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    client = OpenAI(api_key=api_key, base_url=base_url)
    available = {item.id for item in client.models.list().data}
    if model not in available:
        raise ValueError(f"embedding model {model!r} not served: {sorted(available)}")
    def embed_one(index: int, question: str) -> tuple[int, list[float], dict[str, Any]]:
        began = time.perf_counter()
        response = client.embeddings.create(
            model=model,
            input=question,
            dimensions=dimensions,
        )
        if len(response.data) != 1:
            raise ValueError(f"embedding query {index}: expected one vector")
        usage = response.usage
        return index, response.data[0].embedding, {
            "index": index,
            "count": 1,
            "seconds": time.perf_counter() - began,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    completed_rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(embed_one, index, question): index
            for index, question in enumerate(questions)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            completed_rows.append(future.result())
            if completed % 50 == 0 or completed == len(questions):
                print(f"embedded={completed}/{len(questions)}", flush=True)
    completed_rows.sort(key=lambda item: item[0])
    vectors = [item[1] for item in completed_rows]
    usage_rows = [item[2] for item in completed_rows]
    return vectors, usage_rows


def display_timestamp(value: str) -> str:
    if not value:
        return value
    return datetime.fromisoformat(value).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def qdrant_result(point: Any) -> dict[str, Any]:
    payload = point.payload or {}
    created_at = str(payload.get("created_at") or "")
    memory = str(payload.get("data") or payload.get("memory") or "")
    user_id = str(payload.get("user_id") or "")
    return {
        "content": f"{display_timestamp(created_at)}: {memory}",
        "score": float(point.score),
        "user_id": user_id,
        "metadata": {
            "id": str(point.id),
            "created_at": created_at,
            "created_at_display": display_timestamp(created_at),
            "memory": memory,
            "user_id": user_id,
        },
    }


def retrieve_stage(args: argparse.Namespace, output_dir: Path) -> None:
    frozen, validation = load_frozen_metadata(args.run_root)
    vectors_path = output_dir / "query_embeddings.jsonl"
    if vectors_path.exists() and args.resume:
        vector_rows = read_jsonl(vectors_path)
        if len(vector_rows) != 500:
            raise ValueError("saved query embeddings are incomplete")
        vectors = [row["embedding"] for row in vector_rows]
        embedding_usage = []
    else:
        vectors, embedding_usage = embedding_batches(
            [row["question"] for row in frozen],
            base_url=args.embedding_base_url,
            model=args.embedding_model,
            api_key=args.embedding_api_key,
            dimensions=args.embedding_dimensions,
            workers=args.embedding_workers,
        )
        write_jsonl(
            vectors_path,
            [
                {
                    "qid": row["qid"],
                    "conversation_id": row["conversation_id"],
                    "embedding": vector,
                }
                for row, vector in zip(frozen, vectors)
            ],
        )

    by_conversation = {row["conversation_id"]: row for row in frozen}
    vector_by_conversation = {
        row["conversation_id"]: vector for row, vector in zip(frozen, vectors)
    }
    retrieved: dict[str, dict[str, Any]] = {}
    stores = sorted(args.stores_root.glob("shard_*/qdrant"), key=lambda p: shard_number(p.parent))
    if len(stores) != 128:
        raise ValueError(f"expected 128 stores, got {len(stores)}")
    for store_index, qdrant_dir in enumerate(stores, 1):
        client = QdrantClient(path=str(qdrant_dir))
        try:
            collections = client.get_collections().collections
            if len(collections) != 1:
                raise ValueError(f"{qdrant_dir}: expected one collection")
            collection_name = collections[0].name
            offset = None
            user_by_conversation: dict[str, set[str]] = defaultdict(set)
            while True:
                points, offset = client.scroll(
                    collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    conversation_id = str((point.payload or {}).get("conversation_id") or "")
                    user_id = str((point.payload or {}).get("user_id") or "")
                    user_by_conversation[conversation_id].add(user_id)
                if offset is None:
                    break
            for conversation_id, user_ids in sorted(user_by_conversation.items()):
                if conversation_id not in by_conversation:
                    raise ValueError(f"unexpected conversation in store: {conversation_id}")
                if len(user_ids) != 1:
                    raise ValueError(f"{conversation_id}: expected one user, got {user_ids}")
                user_id = next(iter(user_ids))
                response = client.query_points(
                    collection_name=collection_name,
                    query=vector_by_conversation[conversation_id],
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="user_id",
                                match=models.MatchValue(value=user_id),
                            )
                        ]
                    ),
                    limit=200,
                    with_payload=True,
                    with_vectors=False,
                )
                results = [qdrant_result(point) for point in response.points]
                frozen_row = by_conversation[conversation_id]
                new_ids = [result["metadata"]["id"] for result in results[:20]]
                old_ids = frozen_row["old_top20_ids"]
                new_by_id = {
                    result["metadata"]["id"]: result for result in results[:20]
                }
                old_score_by_id = dict(zip(old_ids, frozen_row["old_top20_scores"]))
                old_content_by_id = dict(zip(old_ids, frozen_row["old_top20_contents"]))
                shared_ids = set(new_by_id) & set(old_ids)
                score_deltas = [
                    abs(float(new_by_id[point_id]["score"]) - old_score_by_id[point_id])
                    for point_id in shared_ids
                ]
                content_matches = all(
                    new_by_id[point_id]["content"] == old_content_by_id[point_id]
                    for point_id in shared_ids
                )
                if not content_matches:
                    raise ValueError(f"{conversation_id}: memory content drift for a shared top-20 ID")
                retrieved[conversation_id] = {
                    "qid": frozen_row["qid"],
                    "conversation_id": conversation_id,
                    "question": frozen_row["question"],
                    "gold_answer": frozen_row["gold_answer"],
                    "question_type": frozen_row["question_type"],
                    "metadata": frozen_row["metadata"],
                    "user_id": user_id,
                    "available_memories": len(user_ids) and client.count(
                        collection_name,
                        count_filter=models.Filter(
                            must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
                        ),
                        exact=True,
                    ).count,
                    "results": results,
                    "top10_ordered_exact": new_ids[:10] == old_ids[:10],
                    "top10_set_overlap": len(set(new_ids[:10]) & set(old_ids[:10])),
                    "top20_ordered_exact": new_ids == old_ids,
                    "top20_set_overlap": len(shared_ids),
                    "top20_max_score_delta_by_id": max(score_deltas, default=None),
                    "top20_content_match_by_id": content_matches,
                }
        finally:
            client.close()
        if store_index % 16 == 0 or store_index == len(stores):
            print(f"stores={store_index}/{len(stores)} retrieved={len(retrieved)}/500", flush=True)
    ordered = [retrieved[row["conversation_id"]] for row in frozen]
    if len(ordered) != 500:
        raise ValueError(f"retrieval incomplete: {len(ordered)}")
    top10_exact = sum(row["top10_ordered_exact"] for row in ordered)
    top20_exact = sum(row["top20_ordered_exact"] for row in ordered)
    min_top10_overlap = min(row["top10_set_overlap"] for row in ordered)
    min_top20_overlap = min(row["top20_set_overlap"] for row in ordered)
    if min_top10_overlap < 9 or min_top20_overlap < 18:
        raise RuntimeError(
            "retrieval implementation drift is too large to extend the frozen run: "
            f"min top-10 overlap={min_top10_overlap}/10, "
            f"min top-20 overlap={min_top20_overlap}/20"
        )
    write_jsonl(output_dir / "retrieval_top200.jsonl", ordered)
    write_json(
        output_dir / "retrieval_manifest.json",
        {
            "status": "complete",
            "protocol": "frozen corrected Mem0 state; direct Qdrant cosine query",
            "run_root": str(args.run_root.resolve()),
            "stores_root": str(args.stores_root.resolve()),
            "source_validation": validation,
            "embedding_model": args.embedding_model,
            "embedding_base_url": args.embedding_base_url,
            "embedding_dimensions": args.embedding_dimensions,
            "embedding_usage": embedding_usage,
            "rows": len(ordered),
            "validation_policy": (
                "Re-query the retained stores with the same model/configuration and quantify "
                "numerical ranking drift against the frozen top-20. Require at least 9/10 and "
                "18/20 set overlap for every question; do not splice frozen and new rankings."
            ),
            "top10_ordered_exact_matches": top10_exact,
            "top20_ordered_exact_matches": top20_exact,
            "min_top10_set_overlap": min_top10_overlap,
            "mean_top10_set_overlap": statistics.mean(row["top10_set_overlap"] for row in ordered),
            "min_top20_set_overlap": min_top20_overlap,
            "mean_top20_set_overlap": statistics.mean(row["top20_set_overlap"] for row in ordered),
            "max_top20_score_delta_by_id": max(
                row["top20_max_score_delta_by_id"]
                for row in ordered
                if row["top20_max_score_delta_by_id"] is not None
            ),
            "top20_content_match_by_id": all(row["top20_content_match_by_id"] for row in ordered),
            "mismatch_examples": [
                {
                    "qid": row["qid"],
                    "conversation_id": row["conversation_id"],
                    "top10_set_overlap": row["top10_set_overlap"],
                    "top20_set_overlap": row["top20_set_overlap"],
                }
                for row in ordered
                if not row["top20_ordered_exact"]
            ][:20],
            "retrieval_sha256": sha256(output_dir / "retrieval_top200.jsonl"),
            "runner_sha256": sha256(Path(__file__)),
        },
    )


def numbered_context(results: list[dict[str, Any]], cutoff: int) -> str:
    return "\n\n".join(
        f"{index}. {result['content']}" for index, result in enumerate(results[:cutoff], 1)
    )


def build_question(row: dict[str, Any]) -> str:
    question = row["question"]
    metadata = row.get("metadata") or {}
    if "all_options" in metadata:
        options = "\n".join(f"{key} {value}" for key, value in metadata["all_options"].items())
        question = f"""{question}

OPTIONS:
{options}

IMPORTANT: This is a multiple-choice question. You MUST analyze the context and select the BEST option. In your FINAL ANSWER, return ONLY the option letter like (a), (b), (c), or (d), nothing else."""
    return question


def clean_answer(value: str) -> str:
    answer = (value or "").strip()
    if "FINAL ANSWER:" in answer:
        answer = answer.split("FINAL ANSWER:")[-1].strip()
    return answer


def answer_one(
    client: OpenAI,
    task: dict[str, Any],
    *,
    model: str,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": task["prompt"]}],
                temperature=0,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            raw = response.choices[0].message.content or ""
            answer = clean_answer(raw)
            if not answer:
                raise ValueError("empty answer")
            usage = response.usage
            return {
                **{key: value for key, value in task.items() if key != "prompt"},
                "answer": answer,
                "raw_response": raw,
                "answer_model": model,
                "answer_seconds": time.perf_counter() - started,
                "api_attempts": attempt,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                },
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    return {
        **{key: value for key, value in task.items() if key != "prompt"},
        "answer": "",
        "raw_response": "",
        "answer_model": model,
        "answer_seconds": time.perf_counter() - started,
        "api_attempts": retries,
        "usage": {},
        "error": last_error,
    }


def answer_stage(args: argparse.Namespace, output_dir: Path) -> None:
    retrieval_rows = read_jsonl(output_dir / "retrieval_top200.jsonl")
    if len(retrieval_rows) != 500:
        raise ValueError("retrieval output is incomplete")
    prompt_payload = yaml.safe_load(args.prompt_yaml.read_text(encoding="utf-8"))
    template = prompt_payload["online_api"]["default"]["answer_prompt_mem0"]
    tasks = []
    for row in retrieval_rows:
        for cutoff in CUTOFFS:
            context = numbered_context(row["results"], cutoff)
            reference_date = (row.get("metadata") or {}).get("question_date")
            prompt_context = f"[Reference date: {reference_date}]\n\n{context}" if reference_date else context
            prompt = template.format(context=prompt_context, question=build_question(row))
            tasks.append(
                {
                    "qid": row["qid"],
                    "conversation_id": row["conversation_id"],
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "question_type": row["question_type"],
                    "question_date": reference_date or "",
                    "cutoff": cutoff,
                    "available_memories": row["available_memories"],
                    "effective_results": min(cutoff, len(row["results"])),
                    "context": context,
                    "context_chars": len(context),
                    "selected_result_ids": [
                        result["metadata"]["id"] for result in row["results"][:cutoff]
                    ],
                    "prompt": prompt,
                }
            )
    attempts_path = output_dir / "answer_attempts.jsonl"
    successful = {
        (str(row["qid"]), int(row["cutoff"])): row
        for row in read_jsonl(attempts_path)
        if not row.get("error") and row.get("answer")
    } if args.resume else {}
    if not args.resume:
        attempts_path.unlink(missing_ok=True)
    todo = [task for task in tasks if (task["qid"], task["cutoff"]) not in successful]
    client = OpenAI(api_key=args.answer_api_key, base_url=args.answer_base_url)
    available = {item.id for item in client.models.list().data}
    if args.answer_model not in available:
        raise ValueError(f"answer model not served: {args.answer_model}; available={sorted(available)}")
    lock = threading.Lock()
    errors = 0
    with ThreadPoolExecutor(max_workers=args.answer_workers) as pool:
        futures = {
            pool.submit(
                answer_one,
                client,
                task,
                model=args.answer_model,
                max_tokens=args.answer_max_tokens,
                timeout=args.answer_timeout,
                retries=args.retries,
            ): task
            for task in todo
        }
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            errors += bool(row.get("error"))
            with lock, attempts_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if completed % 50 == 0 or completed == len(todo):
                print(f"answers={completed}/{len(todo)} errors={errors}", flush=True)
    successful = {
        (str(row["qid"]), int(row["cutoff"])): row
        for row in read_jsonl(attempts_path)
        if not row.get("error") and row.get("answer")
    }
    ordered = [successful[(task["qid"], task["cutoff"])] for task in tasks if (task["qid"], task["cutoff"]) in successful]
    write_jsonl(output_dir / "answers.jsonl", ordered)
    write_json(
        output_dir / "answer_manifest.json",
        {
            "status": "complete" if len(ordered) == 1000 else "incomplete",
            "rows": len(ordered),
            "errors_this_run": errors,
            "cutoffs": list(CUTOFFS),
            "answer_model": args.answer_model,
            "answer_base_url": args.answer_base_url,
            "answer_workers": args.answer_workers,
            "answer_prompt_path": str(args.prompt_yaml.resolve()),
            "answer_prompt_sha256": sha256(args.prompt_yaml),
            "mean_prompt_tokens": statistics.mean(
                row["usage"]["prompt_tokens"] for row in ordered if row["usage"].get("prompt_tokens") is not None
            ) if ordered else None,
            "answers_sha256": sha256(output_dir / "answers.jsonl"),
            "runner_sha256": sha256(Path(__file__)),
        },
    )
    if len(ordered) != 1000:
        raise RuntimeError(f"answer stage incomplete: {len(ordered)}/1000")


def parse_strict_label(value: str) -> str:
    text = (value or "").strip()
    try:
        label = str(json.loads(text).get("label", "")).upper().strip()
    except (json.JSONDecodeError, AttributeError):
        label = ""
    if label in {"CORRECT", "WRONG"}:
        return label
    upper = text.upper()
    if "CORRECT" in upper and "WRONG" not in upper:
        return "CORRECT"
    if "WRONG" in upper and "CORRECT" not in upper:
        return "WRONG"
    raise ValueError(f"unparseable strict label: {text[:200]!r}")


def parse_public_label(value: str) -> str:
    text = (value or "").strip().lower()
    if text == "yes" or text.startswith("yes"):
        return "CORRECT"
    if text == "no" or text.startswith("no"):
        return "WRONG"
    raise ValueError(f"unparseable public label: {text[:200]!r}")


def judge_one(
    client: OpenAI,
    task: dict[str, Any],
    *,
    prompt: str,
    system_prompt: str,
    arm: str,
    run: int,
    model: str,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": 0,
                "timeout": timeout,
            }
            if arm == "strict":
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content or ""
            label = parse_strict_label(raw) if arm == "strict" else parse_public_label(raw)
            usage = response.usage
            return {
                **task,
                "judge_arm": arm,
                "judge_run": run,
                "judge_model": model,
                "judge_label": label,
                "judge_raw": raw,
                "judge_seconds": time.perf_counter() - started,
                "api_attempts": attempt,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                },
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    return {
        **task,
        "judge_arm": arm,
        "judge_run": run,
        "judge_model": model,
        "judge_label": "",
        "judge_raw": "",
        "judge_seconds": time.perf_counter() - started,
        "api_attempts": retries,
        "usage": {},
        "error": last_error,
    }


def judge_stage(args: argparse.Namespace, output_dir: Path) -> None:
    answers = read_jsonl(output_dir / "answers.jsonl")
    if len(answers) != 1000:
        raise ValueError("answer output is incomplete")
    strict_namespace = runpy.run_path(str(args.strict_judge_source))
    public_namespace = runpy.run_path(str(args.public_judge_source))
    strict_prompt = str(strict_namespace["JUDGE_PROMPT"])
    public_prompt = str(public_namespace["JUDGE_PROMPT"])
    strict_system_prompt = str(strict_namespace.get("JUDGE_SYSTEM_PROMPT", ""))
    public_system_prompt = str(public_namespace.get("JUDGE_SYSTEM_PROMPT", ""))
    tasks = []
    for row in answers:
        base = {
            "qid": row["qid"],
            "cutoff": int(row["cutoff"]),
            "question_type": row["question_type"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "generated_answer": row["answer"],
            "question_date": row.get("question_date") or "",
        }
        for arm in JUDGE_ARMS:
            prompt = (
                strict_prompt.format(**base)
                if arm == "strict"
                else public_prompt.format(
                    question_type=base["question_type"],
                    question_id=base["qid"],
                    question=base["question"],
                    answer=base["gold_answer"],
                    response=base["generated_answer"],
                    question_date=base["question_date"],
                )
            )
            system_prompt = strict_system_prompt if arm == "strict" else public_system_prompt
            for run in range(1, 4):
                tasks.append(({**base}, arm, run, prompt, system_prompt))
    attempts_path = output_dir / "judge_attempts.jsonl"
    successful = {
        (str(row["qid"]), int(row["cutoff"]), str(row["judge_arm"]), int(row["judge_run"])): row
        for row in read_jsonl(attempts_path)
        if not row.get("error") and row.get("judge_label") in {"CORRECT", "WRONG"}
    } if args.resume else {}
    if not args.resume:
        attempts_path.unlink(missing_ok=True)
    todo = [
        item for item in tasks
        if (item[0]["qid"], item[0]["cutoff"], item[1], item[2]) not in successful
    ]
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is required and is not written to artifacts")
    client = OpenAI(api_key=api_key, base_url=args.judge_base_url)
    lock = threading.Lock()
    errors = 0
    with ThreadPoolExecutor(max_workers=args.judge_workers) as pool:
        futures = {
            pool.submit(
                judge_one,
                client,
                task,
                prompt=prompt,
                system_prompt=system_prompt,
                arm=arm,
                run=run,
                model=args.judge_model,
                timeout=args.judge_timeout,
                retries=args.retries,
            ): (task, arm, run)
            for task, arm, run, prompt, system_prompt in todo
        }
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            errors += bool(row.get("error"))
            with lock, attempts_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if completed % 100 == 0 or completed == len(todo):
                print(f"judges={completed}/{len(todo)} errors={errors}", flush=True)
    successful = {
        (str(row["qid"]), int(row["cutoff"]), str(row["judge_arm"]), int(row["judge_run"])): row
        for row in read_jsonl(attempts_path)
        if not row.get("error") and row.get("judge_label") in {"CORRECT", "WRONG"}
    }
    ordered = [
        successful[(task["qid"], task["cutoff"], arm, run)]
        for task, arm, run, _, _ in tasks
        if (task["qid"], task["cutoff"], arm, run) in successful
    ]
    write_jsonl(output_dir / "judgments.jsonl", ordered)
    write_json(
        output_dir / "judge_manifest.json",
        {
            "status": "complete" if len(ordered) == 6000 else "incomplete",
            "rows": len(ordered),
            "errors_this_run": errors,
            "judge_model": args.judge_model,
            "judge_base_url": args.judge_base_url,
            "judge_workers": args.judge_workers,
            "runs_per_answer": 3,
            "strict_prompt_source": str(args.strict_judge_source.resolve()),
            "strict_prompt_sha256": hashlib.sha256(strict_prompt.encode()).hexdigest(),
            "strict_system_prompt_sha256": hashlib.sha256(strict_system_prompt.encode()).hexdigest(),
            "public_prompt_source": str(args.public_judge_source.resolve()),
            "public_prompt_sha256": hashlib.sha256(public_prompt.encode()).hexdigest(),
            "public_system_prompt_sha256": hashlib.sha256(public_system_prompt.encode()).hexdigest(),
            "keys_recorded": False,
            "judgments_sha256": sha256(output_dir / "judgments.jsonl"),
            "runner_sha256": sha256(Path(__file__)),
        },
    )
    if len(ordered) != 6000:
        raise RuntimeError(f"judge stage incomplete: {len(ordered)}/6000")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap(values: list[float], seed: str, repetitions: int = 10000) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    draws = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(repetitions)]
    return percentile(draws, 0.025), percentile(draws, 0.975)


def aggregate_stage(output_dir: Path) -> None:
    judgments = read_jsonl(output_dir / "judgments.jsonl")
    answers = read_jsonl(output_dir / "answers.jsonl")
    if len(judgments) != 6000 or len(answers) != 1000:
        raise ValueError("cannot aggregate incomplete outputs")
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in judgments:
        grouped[(str(row["qid"]), int(row["cutoff"]), str(row["judge_arm"]))].append(row)
    majority = []
    for key, rows in sorted(grouped.items()):
        if len(rows) != 3 or len({row["judge_run"] for row in rows}) != 3:
            raise ValueError(f"{key}: expected three votes")
        correct_votes = sum(row["judge_label"] == "CORRECT" for row in rows)
        first = rows[0]
        majority.append(
            {
                "qid": key[0],
                "cutoff": key[1],
                "judge_arm": key[2],
                "question_type": first["question_type"],
                "question": first["question"],
                "gold_answer": first["gold_answer"],
                "generated_answer": first["generated_answer"],
                "correct_votes": correct_votes,
                "majority_label": "CORRECT" if correct_votes >= 2 else "WRONG",
            }
        )
    if len(majority) != 2000:
        raise ValueError(f"expected 2000 majority rows, got {len(majority)}")
    write_csv(output_dir / "per_question_majority.csv", majority)

    summary = []
    for cutoff in CUTOFFS:
        for arm in JUDGE_ARMS:
            rows = [row for row in majority if row["cutoff"] == cutoff and row["judge_arm"] == arm]
            for category in ("overall", "temporal-reasoning"):
                selected = rows if category == "overall" else [row for row in rows if row["question_type"] == category]
                labels = [int(row["majority_label"] == "CORRECT") for row in selected]
                low, high = paired_bootstrap(labels, f"cell:{cutoff}:{arm}:{category}")
                summary.append(
                    {
                        "cutoff": cutoff,
                        "judge_arm": arm,
                        "scope": category,
                        "correct": sum(labels),
                        "total": len(labels),
                        "accuracy": sum(labels) / len(labels),
                        "bootstrap_ci95_low": low,
                        "bootstrap_ci95_high": high,
                    }
                )
    write_csv(output_dir / "accuracy_summary.csv", summary)

    by_key = {
        (row["qid"], row["cutoff"], row["judge_arm"]): int(row["majority_label"] == "CORRECT")
        for row in majority
    }
    metadata = {(row["qid"], row["cutoff"], row["judge_arm"]): row for row in majority}
    comparisons = []
    for arm in JUDGE_ARMS:
        for scope in ("overall", "temporal-reasoning"):
            qids = sorted(
                row["qid"] for row in majority
                if row["cutoff"] == 50 and row["judge_arm"] == arm
                and (scope == "overall" or row["question_type"] == scope)
            )
            diffs = [by_key[(qid, 200, arm)] - by_key[(qid, 50, arm)] for qid in qids]
            low, high = paired_bootstrap(diffs, f"budget:{arm}:{scope}")
            comparisons.append(
                {
                    "comparison": "top200_minus_top50",
                    "judge_arm": arm,
                    "scope": scope,
                    "n": len(diffs),
                    "delta": statistics.mean(diffs),
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                }
            )
    for cutoff in CUTOFFS:
        for scope in ("overall", "temporal-reasoning"):
            qids = sorted(
                row["qid"] for row in majority
                if row["cutoff"] == cutoff and row["judge_arm"] == "strict"
                and (scope == "overall" or row["question_type"] == scope)
            )
            diffs = [by_key[(qid, cutoff, "public")] - by_key[(qid, cutoff, "strict")] for qid in qids]
            low, high = paired_bootstrap(diffs, f"judge:{cutoff}:{scope}")
            comparisons.append(
                {
                    "comparison": "public_minus_strict",
                    "cutoff": cutoff,
                    "scope": scope,
                    "n": len(diffs),
                    "delta": statistics.mean(diffs),
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                }
            )
    write_csv(output_dir / "paired_comparisons.csv", comparisons)

    answer_by_key = {(row["qid"], row["cutoff"]): row for row in answers}
    disagreements = []
    for cutoff in CUTOFFS:
        for qid in sorted({row["qid"] for row in majority if row["cutoff"] == cutoff}):
            strict = metadata[(qid, cutoff, "strict")]
            public = metadata[(qid, cutoff, "public")]
            if strict["majority_label"] == public["majority_label"]:
                continue
            disagreements.append(
                {
                    "qid": qid,
                    "cutoff": cutoff,
                    "question_type": strict["question_type"],
                    "question": strict["question"],
                    "gold_answer": strict["gold_answer"],
                    "generated_answer": strict["generated_answer"],
                    "strict_label": strict["majority_label"],
                    "public_label": public["majority_label"],
                    "context": answer_by_key[(qid, cutoff)]["context"],
                    "author_label": "",
                    "author_notes": "",
                }
            )
    write_csv(output_dir / "judge_disagreements_author_audit.csv", disagreements)

    context_summary = []
    for cutoff in CUTOFFS:
        rows = [row for row in answers if row["cutoff"] == cutoff]
        context_summary.append(
            {
                "cutoff": cutoff,
                "rows": len(rows),
                "mean_context_chars": statistics.mean(row["context_chars"] for row in rows),
                "mean_prompt_tokens": statistics.mean(row["usage"]["prompt_tokens"] for row in rows),
                "mean_completion_tokens": statistics.mean(row["usage"]["completion_tokens"] for row in rows),
                "max_prompt_tokens": max(row["usage"]["prompt_tokens"] for row in rows),
                "stores_exhausted": sum(row["available_memories"] <= cutoff for row in rows),
            }
        )
    write_csv(output_dir / "context_summary.csv", context_summary)
    write_json(
        output_dir / "aggregate_manifest.json",
        {
            "status": "complete",
            "answers": len(answers),
            "judgments": len(judgments),
            "majority_rows": len(majority),
            "disagreements": len(disagreements),
            "official_reference": {
                "commit": "edcd6f1d42400837b1fcb6997716f1769dc51a37",
                "top50": {"overall": 0.904, "temporal-reasoning": 0.925},
                "top200": {"overall": 0.934, "temporal-reasoning": 0.932},
            },
            "runner_sha256": sha256(Path(__file__)),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("retrieve", "answer", "judge", "aggregate", "all"), default="all")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stores-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-yaml", type=Path, required=True)
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-api-key", default="EMPTY")
    parser.add_argument("--embedding-dimensions", type=int, default=1024)
    parser.add_argument("--embedding-workers", type=int, default=128)
    parser.add_argument("--answer-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--answer-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--answer-api-key", default="EMPTY")
    parser.add_argument("--answer-workers", type=int, default=128)
    parser.add_argument("--answer-max-tokens", type=int, default=512)
    parser.add_argument("--answer-timeout", type=float, default=300)
    parser.add_argument("--strict-judge-source", type=Path, required=True)
    parser.add_argument("--public-judge-source", type=Path, required=True)
    parser.add_argument("--judge-base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--judge-model", default="deepseek-chat")
    parser.add_argument("--judge-workers", type=int, default=64)
    parser.add_argument("--judge-timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stages = ("retrieve", "answer", "judge", "aggregate") if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"stage={stage}", flush=True)
        if stage == "retrieve":
            retrieve_stage(args, args.output_dir)
        elif stage == "answer":
            answer_stage(args, args.output_dir)
        elif stage == "judge":
            judge_stage(args, args.output_dir)
        elif stage == "aggregate":
            aggregate_stage(args.output_dir)


if __name__ == "__main__":
    main()
