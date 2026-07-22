#!/usr/bin/env python3
"""Profile how much of each frozen Mem0 store a top-k query can expose."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from qdrant_client import QdrantClient


def percentile(values: list[int], fraction: float) -> float:
    """Return a linearly interpolated percentile for a sorted integer list."""
    if not values:
        raise ValueError("percentile requires at least one value")
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def summarize(values: Iterable[int], budgets: list[int]) -> dict:
    ordered = sorted(values)
    result = {
        "n": len(ordered),
        "min": ordered[0],
        "p10": percentile(ordered, 0.10),
        "p25": percentile(ordered, 0.25),
        "median": statistics.median(ordered),
        "mean": statistics.mean(ordered),
        "p75": percentile(ordered, 0.75),
        "p90": percentile(ordered, 0.90),
        "max": ordered[-1],
    }
    result["budgets"] = {
        str(k): {
            "questions_retrieving_all_memories": sum(value <= k for value in ordered),
            "questions_retrieving_all_memories_fraction": statistics.mean(
                value <= k for value in ordered
            ),
            "mean_store_fraction": statistics.mean(
                min(k, value) / value for value in ordered
            ),
            "median_store_fraction": statistics.median(
                min(k, value) / value for value in ordered
            ),
        }
        for k in budgets
    }
    return result


def load_memory_counts(stores_root: Path) -> tuple[Counter, dict[str, set[str]], int]:
    counts: Counter = Counter()
    users: dict[str, set[str]] = defaultdict(set)
    point_keys: Counter = Counter()
    qdrant_dirs = sorted(stores_root.glob("shard_*/qdrant"))
    if not qdrant_dirs:
        raise FileNotFoundError(f"No shard_*/qdrant stores found under {stores_root}")

    for qdrant_dir in qdrant_dirs:
        client = QdrantClient(path=str(qdrant_dir))
        try:
            for collection in client.get_collections().collections:
                offset = None
                while True:
                    points, offset = client.scroll(
                        collection.name,
                        limit=1000,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                    for point in points:
                        conversation_id = point.payload.get("conversation_id")
                        if not conversation_id:
                            raise ValueError(
                                f"Point {point.id} in {collection.name} has no conversation_id"
                            )
                        counts[conversation_id] += 1
                        users[conversation_id].add(point.payload.get("user_id", ""))
                        point_keys[(conversation_id, str(point.id))] += 1
                    if offset is None:
                        break
        finally:
            client.close()

    duplicate_point_keys = sum(count > 1 for count in point_keys.values())
    return counts, users, duplicate_point_keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stores-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--budgets", type=int, nargs="+", default=[10, 30, 50, 100, 200])
    args = parser.parse_args()

    counts, users, duplicate_point_keys = load_memory_counts(args.stores_root)
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    expected_ids = {f"longmemeval_{index}" for index in range(len(dataset))}
    actual_ids = set(counts)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"Store/dataset mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    per_question = []
    by_type: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(dataset):
        conversation_id = f"longmemeval_{index}"
        memory_count = counts[conversation_id]
        question_type = row["question_type"]
        by_type[question_type].append(memory_count)
        per_question.append(
            {
                "conversation_id": conversation_id,
                "question_id": row["question_id"],
                "question_type": question_type,
                "memory_count": memory_count,
                "retrieval_fractions": {
                    str(k): min(k, memory_count) / memory_count for k in args.budgets
                },
            }
        )

    output = {
        "source": {
            "stores_root": str(args.stores_root),
            "dataset": str(args.dataset),
            "budgets": args.budgets,
        },
        "validation": {
            "conversations": len(counts),
            "memory_points": sum(counts.values()),
            "multi_user_conversations": sum(len(value) > 1 for value in users.values()),
            "duplicate_conversation_point_keys": duplicate_point_keys,
        },
        "overall": summarize(counts.values(), args.budgets),
        "by_question_type": {
            question_type: summarize(values, args.budgets)
            for question_type, values in sorted(by_type.items())
        },
        "per_question": per_question,
    }

    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
