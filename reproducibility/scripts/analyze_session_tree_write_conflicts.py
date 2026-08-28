#!/usr/bin/env python3
"""Audit session-level MemTree write concentration in matched build traces.

Each canonical fact is attributed to its first source session and joined to
the final entity/scene trees whose leaves contain that fact.  A session-tree
collision means that at least two canonical facts from one source session are
routed to the same tree.  This is an upper-bound structural-conflict proxy for
per-fact insertion; the released builder deduplicates affected tree IDs and
dirty ancestors before the LLM refresh.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


DEFAULT_QIDS = ("28dc39ac", "bc8a6e93", "7e00a6cb")
GLOBAL_FALLBACK_IDS = {"entity:user", "tree:entity:user"}


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * q + 0.999999)))
    return ordered[index]


def load_fact_sessions(run_root: Path, qid: str) -> dict[str, str]:
    path = run_root / "workdir_measured" / qid / "fact_store" / "facts.jsonl"
    out: dict[str, str] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            fact_id = str(row["fact_id"])
            session_id = str(row.get("first_session_id") or "")
            if not session_id:
                occurrences = row.get("occurrences") or []
                if occurrences:
                    session_id = str(occurrences[0].get("session_id") or "")
            if session_id:
                out[fact_id] = session_id
    return out


def load_fact_trees(
    run_root: Path,
    qid: str,
) -> tuple[dict[str, set[str]], dict[str, str]]:
    root = run_root / "workdir_measured" / qid / "build" / "trees"
    fact_trees: dict[str, set[str]] = defaultdict(set)
    tree_types: dict[str, str] = {}
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text())
        tree_id = str(payload.get("tree_id") or path.stem)
        tree_types[tree_id] = str(payload.get("tree_type") or path.parent.name)
        nodes = payload.get("nodes") or {}
        node_rows = nodes.values() if isinstance(nodes, dict) else nodes
        for node in node_rows:
            if int(node.get("level") or 0) != 0:
                continue
            for fact_id in node.get("child_ids") or []:
                fact_trees[str(fact_id)].add(tree_id)
    return fact_trees, tree_types


def summarize_counts(
    qid: str,
    session_tree_counts: dict[tuple[str, str], int],
    *,
    include_global_fallback: bool,
    pair_within_prefixed_stream: bool = False,
) -> dict[str, object]:
    selected = {
        key: count
        for key, count in session_tree_counts.items()
        if include_global_fallback or key[1] not in GLOBAL_FALLBACK_IDS
    }
    by_session: dict[str, set[str]] = defaultdict(set)
    for (session_id, tree_id), count in selected.items():
        if count > 0:
            by_session[session_id].add(tree_id)

    loads = list(selected.values())
    total_assignments = sum(loads)
    collision_pairs = sum(count > 1 for count in loads)
    excess_assignments = sum(max(0, count - 1) for count in loads)
    session_ids = sorted(by_session)

    session_pairs = 0
    session_pairs_with_overlap = 0
    shared_tree_counts: list[int] = []
    for index, left in enumerate(session_ids):
        for right in session_ids[index + 1 :]:
            if pair_within_prefixed_stream:
                left_stream = left.split(":", 1)[0]
                right_stream = right.split(":", 1)[0]
                if left_stream != right_stream:
                    continue
            session_pairs += 1
            shared = by_session[left] & by_session[right]
            shared_tree_counts.append(len(shared))
            session_pairs_with_overlap += bool(shared)

    return {
        "qid": qid,
        "scope": "all" if include_global_fallback else "without_global_fallback",
        "sessions": len(session_ids),
        "session_tree_pairs": len(loads),
        "fact_tree_assignments": total_assignments,
        "mean_trees_per_session": mean(len(by_session[sid]) for sid in session_ids) if session_ids else 0.0,
        "mean_facts_per_touched_tree": mean(loads) if loads else 0.0,
        "median_facts_per_touched_tree": median(loads) if loads else 0.0,
        "p95_facts_per_touched_tree": percentile(loads, 0.95),
        "max_facts_per_touched_tree": max(loads, default=0),
        "collision_session_tree_pairs": collision_pairs,
        "collision_pair_rate": collision_pairs / len(loads) if loads else 0.0,
        "excess_assignment_rate": excess_assignments / total_assignments if total_assignments else 0.0,
        "session_pairs": session_pairs,
        "session_pairs_with_shared_tree": session_pairs_with_overlap,
        "session_pair_overlap_rate": session_pairs_with_overlap / session_pairs if session_pairs else 0.0,
        "mean_shared_trees_per_session_pair": mean(shared_tree_counts) if shared_tree_counts else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Matched-build root containing workdir_measured/<qid>.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for summary.csv, session_tree_loads.csv, and README.md.",
    )
    parser.add_argument(
        "--qids",
        nargs="+",
        default=list(DEFAULT_QIDS),
        help="Independent question/user streams to aggregate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    aggregate_counts: dict[tuple[str, str], int] = Counter()

    for qid in args.qids:
        fact_sessions = load_fact_sessions(run_root, qid)
        fact_trees, tree_types = load_fact_trees(run_root, qid)
        session_tree_counts: dict[tuple[str, str], int] = Counter()
        for fact_id, session_id in fact_sessions.items():
            for tree_id in fact_trees.get(fact_id, set()):
                session_tree_counts[(session_id, tree_id)] += 1
                aggregate_counts[(f"{qid}:{session_id}", tree_id)] += 1

        for (session_id, tree_id), count in sorted(session_tree_counts.items()):
            detail_rows.append(
                {
                    "qid": qid,
                    "session_id": session_id,
                    "tree_id": tree_id,
                    "tree_type": tree_types.get(tree_id, ""),
                    "is_global_fallback": tree_id in GLOBAL_FALLBACK_IDS,
                    "fact_count": count,
                    "same_tree_collision": count > 1,
                }
            )

        summary_rows.append(
            summarize_counts(qid, session_tree_counts, include_global_fallback=True)
        )
        summary_rows.append(
            summarize_counts(qid, session_tree_counts, include_global_fallback=False)
        )

    summary_rows.append(
        summarize_counts(
            "aggregate",
            aggregate_counts,
            include_global_fallback=True,
            pair_within_prefixed_stream=True,
        )
    )
    summary_rows.append(
        summarize_counts(
            "aggregate",
            aggregate_counts,
            include_global_fallback=False,
            pair_within_prefixed_stream=True,
        )
    )

    detail_path = output_dir / "session_tree_loads.csv"
    with detail_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    aggregate = [row for row in summary_rows if row["qid"] == "aggregate"]
    aggregate_by_scope = {str(row["scope"]): row for row in aggregate}
    specific_overlap = float(
        aggregate_by_scope["without_global_fallback"]["session_pair_overlap_rate"]
    )
    lines = [
        "# Session-level MemTree write concentration",
        "",
        "Source streams: " + ", ".join(f"`{qid}`" for qid in args.qids) + ".",
        "Inputs are the saved canonical fact store and final entity/scene trees "
        "under `workdir_measured/<qid>`.",
        "",
        "A collision is two or more canonical facts from one source session routed "
        "to the same final tree. The metric is an upper-bound conflict proxy for "
        "per-fact insertion; the released builder deduplicates affected tree IDs "
        "and dirty ancestors before the LLM refresh.",
        "Session-pair overlap is computed only within each independent question/user "
        "stream before the three streams are aggregated.",
        "For tree-level concurrent writes, session-pair overlap is the relevant "
        f"potential lock-conflict proxy: excluding the global fallback, "
        f"{100 * specific_overlap:.1f}% of pairs share a specific entity/scene "
        f"tree and {100 * (1.0 - specific_overlap):.1f}% are disjoint. The "
        "global `entity:user` tree is a separable hotspot that can be partitioned "
        "into independently built subforests and consolidated through migration/merge.",
        "",
        "| Scope | Sessions | Trees/session | Facts/touched tree (mean/median/p95/max) | Collision pairs | Excess assignments | Session-pair overlap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['scope']} | {row['sessions']} | "
            f"{float(row['mean_trees_per_session']):.2f} | "
            f"{float(row['mean_facts_per_touched_tree']):.2f}/"
            f"{float(row['median_facts_per_touched_tree']):.1f}/"
            f"{int(row['p95_facts_per_touched_tree'])}/"
            f"{int(row['max_facts_per_touched_tree'])} | "
            f"{100 * float(row['collision_pair_rate']):.1f}% | "
            f"{100 * float(row['excess_assignment_rate']):.1f}% | "
            f"{100 * float(row['session_pair_overlap_rate']):.1f}% |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
