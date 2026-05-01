"""Batch extraction runner for LongMemEval-style datasets."""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from src.api.client import OpenAIChatClient
from src.config import MemForestConfig, load_default_config
from src.extraction.manager import ExtractionManager
from src.extraction.pipeline import ChunkExtractionPipeline, JsonExtractionBackend
from src.logger import ApiCallLogger, ExtractionLogger
from src.prompt import build_extraction_prompt_manager
from src.utils.time import parse_timestamp_to_unix
from src.utils.types import ExtractionRequest


class MeasuredJsonBackend(JsonExtractionBackend):
    """OpenAI-backed extraction backend that records per-call timing."""

    def __init__(
        self,
        *,
        chat_client: OpenAIChatClient,
        model_name: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> None:
        self.chat_client = chat_client
        self.model_name = model_name
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.top_p = float(top_p)
        self.calls: list[dict[str, Any]] = []
        self._calls_lock = threading.Lock()

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        result = self.chat_client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            step_label="extraction",
            trace=trace,
        )
        with self._calls_lock:
            self.calls.append(
                {
                    "kind": "unified_extraction",
                    "elapsed_sec": time.perf_counter() - started,
                    "request_id": None if trace is None else trace.get("request_id"),
                    "session_id": None if trace is None else trace.get("session_id"),
                    "cell_id": None if trace is None else trace.get("cell_id"),
                    "cell_index": None if trace is None else trace.get("cell_index"),
                    "system_chars": len(system_prompt),
                    "user_chars": len(user_prompt),
                    "response_chars": len(json.dumps(result, ensure_ascii=False)),
                }
            )
        return result


def run_longmemeval_parallel(
    *,
    dataset_path: str | Path,
    question_ids: list[str],
    output_dir: str | Path,
    config: MemForestConfig | None = None,
    max_workers: int | None = None,
    max_inflight: int | None = None,
    max_sessions_per_question: int = 1,
    show_progress: bool = True,
    pipeline_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Run extraction over selected LongMemEval questions with one shared manager."""

    config = config or load_default_config()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    items_by_qid = {item["question_id"]: item for item in dataset}
    selected = [items_by_qid[qid] for qid in question_ids if qid in items_by_qid]
    if len(selected) != len(question_ids):
        missing = sorted(set(question_ids) - set(items_by_qid))
        raise KeyError(f"Missing question ids in dataset: {missing}")

    inflight = max(1, int(max_inflight or max_workers or config.extraction.max_inflight_requests))
    manager, backend = _build_runtime_bundle(
        config=config,
        inflight=inflight,
        pipeline_factory=pipeline_factory,
    )

    all_requests: list[ExtractionRequest] = []
    question_session_lists: list[tuple[dict[str, Any], list[str], list[list[Any]]]] = []
    for item in selected:
        haystack_ids = list(item.get("haystack_session_ids", []))
        idx_by_sid = {sid: i for i, sid in enumerate(haystack_ids)}
        matched = [sid for sid in item.get("answer_session_ids", []) if sid in idx_by_sid][: max(1, max_sessions_per_question)]

        session_turns_list: list[list[Any]] = []
        for sid in matched:
            sess_idx = idx_by_sid[sid]
            turns = _build_turns(
                session_id=sid,
                session_date=item["haystack_dates"][sess_idx],
                session_turns=item["haystack_sessions"][sess_idx],
            )
            all_requests.append(
                ExtractionRequest(
                    session_id=sid,
                    turns=turns,
                    request_id=f"{item['question_id']}:{sid}",
                    metadata={
                        "question_id": item["question_id"],
                        "question_type": item["question_type"],
                    },
                )
            )
            session_turns_list.append(turns)
        question_session_lists.append((item, matched, session_turns_list))

    batch_started = time.perf_counter()
    batch_results = manager.extract_requests(
        all_requests,
        show_progress=show_progress,
        progress_label="extraction",
    )
    batch_elapsed = time.perf_counter() - batch_started
    result_by_req = {req.request_id: res for req, res in zip(all_requests, batch_results)}

    results: list[dict[str, Any]] = []
    for item, matched, session_turns_list in question_session_lists:
        session_summaries: list[dict[str, Any]] = []
        all_facts: list[str] = []
        for sid, turns in zip(matched, session_turns_list):
            req_id = f"{item['question_id']}:{sid}"
            result = result_by_req.get(req_id)
            if result is None:
                continue
            items = result.memory_items
            all_facts.extend(memory_item.fact_text for memory_item in items)
            session_summaries.append(
                {
                    "session_id": sid,
                    "turn_count": len(turns),
                    "cell_count": len(result.cells),
                    "item_count": len(items),
                    "exact_detail_count": sum(1 for memory_item in items if memory_item.detail_level == "exact"),
                    "time_bearing_count": sum(
                        1
                        for memory_item in items
                        if memory_item.time_text or memory_item.time_start is not None or memory_item.time_end is not None
                    ),
                    "role_preview": [memory_item.semantic_role for memory_item in items[:12]],
                    "origin_preview": [memory_item.origin for memory_item in items[:12]],
                    "fact_preview": [memory_item.fact_text for memory_item in items[:12]],
                    "topic_preview": [topic for memory_item in items[:6] for topic in memory_item.topics[:2]][:12],
                    "attribute_key_preview": [attr for memory_item in items[:6] for attr in memory_item.attribute_keys[:2]][:12],
                    "domain_key_preview": [key for memory_item in items[:6] for key in memory_item.domain_keys[:2]][:12],
                    "collection_key_preview": [key for memory_item in items[:6] for key in memory_item.collection_keys[:2]][:12],
                }
            )

        gold_answer = str(item.get("answer", "")).strip()
        results.append(
            {
                "question_id": item["question_id"],
                "question_type": item["question_type"],
                "question": item["question"],
                "gold_answer": gold_answer,
                "answer_session_ids": matched,
                "session_count": len(matched),
                "cell_count": sum(row["cell_count"] for row in session_summaries),
                "item_count": sum(row["item_count"] for row in session_summaries),
                "avg_items_per_cell": (
                    sum(row["item_count"] for row in session_summaries) / sum(row["cell_count"] for row in session_summaries)
                    if sum(row["cell_count"] for row in session_summaries)
                    else 0.0
                ),
                "exact_detail_count": sum(row["exact_detail_count"] for row in session_summaries),
                "time_bearing_count": sum(row["time_bearing_count"] for row in session_summaries),
                "gold_exact_hit_in_facts": _normalize_text(gold_answer) in _normalize_text(" ".join(all_facts)),
                "gold_fuzzy_hit_in_facts": _fuzzy_hit(gold_answer, all_facts),
                "temporal_anchor_count": (_ta := _count_temporal_anchors(all_facts)),
                "has_two_temporal_anchors": _ta >= 2,
                "elapsed_sec": 0.0,
                "api_call_count": 0,
                "api_call_sec": 0.0,
                "all_fact_texts": all_facts,
                "session_summaries": session_summaries,
            }
        )

    results.sort(key=lambda row: question_ids.index(row["question_id"]))
    total_api_calls, total_api_call_sec = _backend_call_stats(backend)
    summary = _summarize_parallel_results(
        results,
        worker_count=inflight,
        total_elapsed_sec=batch_elapsed,
        total_api_calls=total_api_calls,
        total_api_call_sec=total_api_call_sec,
    )
    (output_root / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _build_runtime_bundle(
    *,
    config: MemForestConfig,
    inflight: int,
    pipeline_factory: Callable[[], Any] | None,
) -> tuple[ExtractionManager, Any]:
    if pipeline_factory is not None:
        raw = pipeline_factory()
        manager, _, backend = _coerce_worker_bundle(raw, max_inflight_requests=inflight)
        if manager is None:
            pipeline = raw[0]
            if not hasattr(pipeline, "extract_session"):
                raise TypeError("pipeline_factory must provide a pipeline compatible with ExtractionManager or expose extract_session().")
            return _SequentialRequestAdapter(pipeline), backend
        return manager, backend

    api_logger = ApiCallLogger(
        config.logger.api.path,
        enabled=config.logger.api.enabled,
    )
    extraction_logger = ExtractionLogger(
        config.logger.extraction.path,
        enabled=config.logger.extraction.enabled,
    )
    chat_client = OpenAIChatClient(config.api.llm, api_logger=api_logger)
    backend = MeasuredJsonBackend(
        chat_client=chat_client,
        model_name=config.api.llm.model_name,
        temperature=config.extraction.temperature,
        max_tokens=config.extraction.max_tokens,
        top_p=config.extraction.top_p,
    )
    pipeline = ChunkExtractionPipeline(
        backend=backend,
        chunking=config.extraction.chunking,
        prompt_manager=build_extraction_prompt_manager(),
        max_items_per_cell=config.extraction.max_items_per_cell,
        max_topics_per_item=config.extraction.max_topics_per_item,
        max_attribute_keys_per_item=config.extraction.max_attribute_keys_per_item,
        max_domain_keys_per_item=config.extraction.max_domain_keys_per_item,
        max_collection_keys_per_item=config.extraction.max_collection_keys_per_item,
    )
    manager = ExtractionManager(
        pipeline=pipeline,
        max_inflight_requests=inflight,
        extraction_logger=extraction_logger,
    )
    return manager, backend


def _coerce_worker_bundle(
    raw: Any,
    *,
    max_inflight_requests: int,
) -> tuple[ExtractionManager | None, Any, Any]:
    if not isinstance(raw, tuple):
        raise TypeError("pipeline_factory must return a tuple.")
    if len(raw) == 3:
        manager, pipeline, backend = raw
        return manager, pipeline, backend
    if len(raw) != 2:
        raise ValueError("pipeline_factory must return (pipeline, backend) or (manager, pipeline, backend).")

    pipeline, backend = raw
    if hasattr(pipeline, "build_cells") and hasattr(pipeline, "extract_cell_request"):
        manager = ExtractionManager(
            pipeline=pipeline,
            max_inflight_requests=max_inflight_requests,
        )
        return manager, pipeline, backend
    return None, pipeline, backend


class _SequentialRequestAdapter:
    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def extract_requests(
        self,
        requests: list[ExtractionRequest],
        *,
        show_progress: bool = False,
        progress_label: str = "extract",
    ) -> list[Any]:
        results: list[Any] = []
        total = len(requests)
        started = time.perf_counter()
        for idx, request in enumerate(requests, start=1):
            results.append(self.pipeline.extract_session(request.session_id, request.turns))
            if show_progress:
                elapsed = time.perf_counter() - started
                pct = (idx / total) * 100.0 if total else 100.0
                print(
                    f"\r[{progress_label}] requests {idx}/{total} ({pct:5.1f}%) | elapsed {elapsed:6.1f}s",
                    end="",
                    flush=True,
                )
        if show_progress and total:
            print(flush=True)
        return results


def _backend_call_stats(backend: Any) -> tuple[int, float]:
    calls = getattr(backend, "calls", None)
    if not isinstance(calls, list):
        return 0, 0.0
    return len(calls), float(sum(float(row.get("elapsed_sec", 0.0)) for row in calls if isinstance(row, dict)))


def _build_turns(*, session_id: str, session_date: str, session_turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_ts = parse_timestamp_to_unix(session_date)
    turns: list[dict[str, Any]] = []
    for turn_idx, turn in enumerate(session_turns):
        role = str(turn.get("role", "unknown"))
        speaker_name = "User" if role == "user" else "Assistant" if role == "assistant" else role.title()
        listener_name = "Assistant" if role == "user" else "User"
        ts = datetime.fromtimestamp(base_ts, tz=timezone.utc) + timedelta(seconds=turn_idx)
        turns.append(
            {
                "role": role,
                "speaker_name": speaker_name,
                "listener_name": listener_name,
                "content": turn.get("content", ""),
                "timestamp": ts.isoformat(),
                "content_id": f"{session_id}#turn_{turn_idx:04d}",
            }
        )
    return turns


def _normalize_text(text: str) -> str:
    lowered = str(text or "").lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "i", "my", "me", "you", "your",
    "it", "its", "and", "or", "not", "no", "in", "on", "at", "to", "of", "for",
    "with", "that", "this", "they", "their", "be", "have", "has", "had", "do",
    "did", "will", "would", "could", "should", "may", "might", "about", "from",
    "by", "as", "so", "but", "if", "than", "also", "just", "we", "our", "us",
})


def _fuzzy_hit(gold: str, facts: list[str]) -> bool:
    """Return True if gold answer has meaningful token overlap with any extracted fact.

    Criteria (any one suffices):
    - Exact substring match (same as gold_exact_hit)
    - Token Jaccard >= 0.25 between gold and any single fact (after stop-word removal)
    - Gold tokens are a subset of fact tokens (gold fully covered by a single fact)
    - Recall-based: |overlap| / |gold_tokens| >= 0.50 (for longer gold answers like preferences)
    """
    gold_norm = _normalize_text(gold)
    facts_concat = _normalize_text(" ".join(facts))
    if gold_norm and gold_norm in facts_concat:
        return True
    gold_tokens = {w for w in gold_norm.split() if w not in _STOP_WORDS and len(w) >= 2}
    if not gold_tokens:
        return False
    for fact in facts:
        fact_tokens = {w for w in _normalize_text(fact).split() if w not in _STOP_WORDS and len(w) >= 2}
        if not fact_tokens:
            continue
        overlap = gold_tokens & fact_tokens
        if not overlap:
            continue
        jaccard = len(overlap) / len(gold_tokens | fact_tokens)
        if jaccard >= 0.25:
            return True
        if gold_tokens <= fact_tokens:
            return True
        # Recall-based criterion: gold is well-represented in the fact
        if len(gold_tokens) >= 4 and len(overlap) / len(gold_tokens) >= 0.50:
            return True
    return False


_TEMPORAL_PATTERN = re.compile(
    r'\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
    r'jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?'
    r'|20\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b',
    re.IGNORECASE,
)


def _count_temporal_anchors(facts: list[str]) -> int:
    """Count facts that contain both a date reference and a user-owned event."""
    count = 0
    for fact in facts:
        if _TEMPORAL_PATTERN.search(fact) and re.search(r'\buser\b', fact, re.IGNORECASE):
            count += 1
    return count


def _summarize_parallel_results(
    results: list[dict[str, Any]],
    *,
    worker_count: int,
    total_elapsed_sec: float,
    total_api_calls: int,
    total_api_call_sec: float,
) -> dict[str, Any]:
    return {
        "question_count": len(results),
        "worker_count": worker_count,
        "total_elapsed_sec": total_elapsed_sec,
        "total_api_calls": total_api_calls,
        "total_api_call_sec": total_api_call_sec,
        "avg_question_sec": total_elapsed_sec / len(results) if results else 0.0,
        "avg_api_calls_per_question": total_api_calls / len(results) if results else 0.0,
        "total_item_count": sum(row["item_count"] for row in results),
        "avg_items_per_cell": (
            sum(row["item_count"] for row in results) / sum(row["cell_count"] for row in results)
            if sum(row["cell_count"] for row in results)
            else 0.0
        ),
        "exact_detail_item_count": sum(row["exact_detail_count"] for row in results),
        "time_bearing_item_count": sum(row["time_bearing_count"] for row in results),
        "questions_with_gold_hit": sum(1 for row in results if row["gold_exact_hit_in_facts"]),
        "questions_with_gold_fuzzy_hit": sum(1 for row in results if row.get("gold_fuzzy_hit_in_facts", False)),
        "temporal_questions_with_two_anchors": sum(
            1 for row in results
            if row.get("question_type") == "temporal-reasoning" and row.get("has_two_temporal_anchors", False)
        ),
        "questions_with_two_temporal_anchors": sum(1 for row in results if row.get("has_two_temporal_anchors", False)),
    }
