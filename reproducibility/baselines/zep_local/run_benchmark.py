#!/usr/bin/env python3
"""Run the resumable Graphiti/Neo4j Zep Local revision benchmark.

The implementation imports the public MemoryData adapter at its pinned commit,
but owns dataset conversion, persistence, concurrency, and instrumentation so a
completed graph can be reused for retrieval, answering, and re-judging.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import hashlib
import json
import logging
import os
import queue
import re
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


WORKSPACE = Path(__file__).resolve().parents[3]
GRAPHITI_ROOT = Path(
    os.getenv("ZEP_LOCAL_GRAPHITI_ROOT", WORKSPACE / "external/graphiti-0.24.1")
).resolve()
MEMORYDATA_ROOT = Path(
    os.getenv("ZEP_LOCAL_MEMORYDATA_ROOT", WORKSPACE / "external/MemoryData")
).resolve()
for source_root in (str(MEMORYDATA_ROOT), str(GRAPHITI_ROOT)):
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

# Graphiti probes optional properties while a namespace is still sparse. Neo4j
# reports those probes as notifications; they are expected and otherwise dwarf
# the benchmark's actionable build/error logs.
logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("neo4j.notifications").disabled = True

from graphiti_core.nodes import EpisodeType  # noqa: E402
from graphiti_core.utils.maintenance import clear_data  # noqa: E402
from methods.zep_local.main import (  # noqa: E402
    ANSWER_SYSTEM_PROMPT,
    GraphitiLocalMemory,
    _extract_openai_message_text,
    _is_multiple_choice_question,
    _should_disable_qwen3_thinking,
    build_answer_context,
    extract_retrieved_facts,
)


ADAPTER_COMMIT = "c63391c128e33eedb91115edf689f12acf4bbc63"
GRAPHITI_COMMIT = "d2654003ffc11821bce73c493162a40181b23504"
GRAPHITI_VERSION = "0.24.1"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIM = 1024
RETRIEVAL_LIMIT = 10
DATE_PATTERN = re.compile(r"\s*\([A-Za-z]{3}\)\s*")


MODELS = {
    "qwen4b": "Qwen/Qwen3-4B-Instruct-2507-FP8",
    "qwen30b": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "gemma": "google/gemma-4-12B-it",
}


CALL_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "zep_local_call_context", default={}
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(encoded.encode("utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def sanitize_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned[:180] or "default"


def parse_benchmark_date(value: str) -> datetime:
    normalized = DATE_PATTERN.sub(" ", str(value).strip())
    parsed = datetime.strptime(normalized, "%Y/%m/%d %H:%M")
    return parsed.replace(tzinfo=timezone.utc)


def usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = int(
        getattr(usage, "prompt_tokens", None)
        or getattr(usage, "input_tokens", None)
        or 0
    )
    output_tokens = int(
        getattr(usage, "completion_tokens", None)
        or getattr(usage, "output_tokens", None)
        or 0
    )
    total_tokens = int(getattr(usage, "total_tokens", None) or input_tokens + output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


class JsonlSink:
    """Non-blocking append-only JSONL sink for high-concurrency call traces."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=100_000)
        self._thread = threading.Thread(target=self._run, name=f"jsonl-{path.name}", daemon=True)
        self._thread.start()

    def append(self, value: dict[str, Any]) -> None:
        self._queue.put(value)

    def _run(self) -> None:
        with self.path.open("a", encoding="utf-8", buffering=1024 * 1024) as handle:
            pending = 0
            while True:
                item = self._queue.get()
                if item is None:
                    self._queue.task_done()
                    break
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                pending += 1
                if pending >= 32:
                    handle.flush()
                    pending = 0
                self._queue.task_done()
            handle.flush()

    def close(self) -> None:
        self._queue.put(None)
        self._queue.join()
        self._thread.join(timeout=30)


class InstrumentedCreate:
    def __init__(self, create: Any, sink: JsonlSink, call_kind: str, activity_file: Path):
        self._create = create
        self._sink = sink
        self._call_kind = call_kind
        self._activity_file = activity_file

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        call_id = uuid.uuid4().hex
        started_utc = utc_now()
        started = time.perf_counter()
        context = dict(CALL_CONTEXT.get())
        messages = kwargs.get("messages") or []
        inputs = kwargs.get("input")
        prompt_material = messages if messages else inputs
        prompt_json = json.dumps(prompt_material, ensure_ascii=False, default=str)
        base = {
            "call_id": call_id,
            "kind": self._call_kind,
            "started_at": started_utc,
            "model": kwargs.get("model"),
            "prompt_hash": sha256_bytes(prompt_json.encode("utf-8")),
            "prompt_chars": len(prompt_json),
            **context,
        }
        if inputs is not None:
            base["input_count"] = len(inputs) if isinstance(inputs, list) else 1
        self._activity_file.parent.mkdir(parents=True, exist_ok=True)
        self._activity_file.touch()
        try:
            response = await self._create(*args, **kwargs)
            self._sink.append(
                {
                    **base,
                    **usage_dict(response),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "status": "ok",
                    "finished_at": utc_now(),
                }
            )
            return response
        except Exception as exc:
            self._sink.append(
                {
                    **base,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:2000],
                    "finished_at": utc_now(),
                }
            )
            raise
        finally:
            self._activity_file.touch()


class ResourceProxy:
    def __init__(self, original: Any, create: InstrumentedCreate):
        self._original = original
        self.create = create

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class ChatProxy:
    def __init__(self, original: Any, create: InstrumentedCreate):
        self._original = original
        self.completions = ResourceProxy(original.completions, create)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class ClientProxy:
    def __init__(
        self,
        original: Any,
        sink: JsonlSink,
        call_kind: str,
        activity_file: Path,
        *,
        embeddings: bool = False,
    ):
        self._original = original
        if embeddings:
            self.embeddings = ResourceProxy(
                original.embeddings,
                InstrumentedCreate(original.embeddings.create, sink, call_kind, activity_file),
            )
        else:
            self.chat = ChatProxy(
                original.chat,
                InstrumentedCreate(
                    original.chat.completions.create, sink, call_kind, activity_file
                ),
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


@dataclass(frozen=True)
class Episode:
    uuid: str
    name: str
    body: str
    source_description: str
    reference_time: datetime


@dataclass(frozen=True)
class MemoryGroup:
    source_id: str
    namespace: str
    sessions: list[list[dict[str, Any]]]
    session_ids: list[str]
    dates: list[str]

    def episodes(self) -> list[Episode]:
        result: list[Episode] = []
        for session_index, (session, session_id, date_text) in enumerate(
            zip(self.sessions, self.session_ids, self.dates, strict=True)
        ):
            base_time = parse_benchmark_date(date_text)
            for turn_index, message in enumerate(session):
                speaker = str(message.get("speaker") or message.get("role") or "Speaker").strip()
                content = str(message.get("content") or "").strip()
                if not content:
                    continue
                episode_name = f"{self.source_id}_{session_id}_turn_{turn_index:04d}"
                episode_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"zep-local:{self.namespace}:{episode_name}"))
                result.append(
                    Episode(
                        uuid=episode_uuid,
                        name=episode_name,
                        body=f"\n{speaker}: {content}",
                        source_description=f"{session_id} at {date_text}",
                        reference_time=base_time
                        + timedelta(microseconds=session_index * 10_000 + turn_index),
                    )
                )
        return result


class DatasetView:
    def __init__(self, benchmark: str, data_root: Path, namespace_prefix: str):
        self.benchmark = benchmark
        self.data_root = data_root
        self.path = data_root / (
            "longmemeval_s_cleaned.json" if benchmark == "longmemeval" else "locomo10_real.json"
        )
        self.rows: list[dict[str, Any]] = read_json(self.path)
        if benchmark == "locomo":
            self._recover_locomo_adversarial_answers()
        self.groups = self._build_groups(namespace_prefix)

    def _recover_locomo_adversarial_answers(self) -> None:
        original_path = self.data_root / "locomo10.json"
        if not original_path.exists():
            return
        original = read_json(original_path)
        recovered: dict[str, Any] = {}
        for sample in original:
            sample_id = str(sample.get("sample_id") or "")
            for qa_index, qa in enumerate(sample.get("qa", [])):
                recovered[f"{sample_id}_qa{qa_index:03d}"] = qa.get("adversarial_answer")
        for row in self.rows:
            if str(row.get("question_type")) != "category_5":
                continue
            answer = recovered.get(str(row.get("question_id")))
            if answer is not None:
                row["answer"] = answer

    def _build_groups(self, namespace_prefix: str) -> list[MemoryGroup]:
        if self.benchmark == "longmemeval":
            source_rows = [(str(row["question_id"]), row) for row in self.rows]
        else:
            unique: dict[str, dict[str, Any]] = {}
            hashes: dict[str, str] = {}
            for row in self.rows:
                sample_id = str(row["locomo_sample_id"])
                session_hash = stable_hash(
                    [row.get("haystack_sessions"), row.get("haystack_dates"), row.get("haystack_session_ids")]
                )
                if sample_id in hashes and hashes[sample_id] != session_hash:
                    raise ValueError(f"LoCoMo session mismatch for {sample_id}")
                hashes[sample_id] = session_hash
                unique.setdefault(sample_id, row)
            source_rows = list(unique.items())
        groups = []
        for source_id, row in source_rows:
            namespace = sanitize_id(f"{namespace_prefix}_{self.benchmark}_{source_id}")
            groups.append(
                MemoryGroup(
                    source_id=source_id,
                    namespace=namespace,
                    sessions=list(row["haystack_sessions"]),
                    session_ids=[str(value) for value in row["haystack_session_ids"]],
                    dates=[str(value) for value in row["haystack_dates"]],
                )
            )
        return groups

    def rows_for_groups(self, group_ids: set[str]) -> list[dict[str, Any]]:
        if self.benchmark == "longmemeval":
            return [row for row in self.rows if str(row["question_id"]) in group_ids]
        return [row for row in self.rows if str(row["locomo_sample_id"]) in group_ids]


class ZepRuntime:
    def __init__(self, args: argparse.Namespace, run_dir: Path):
        self.args = args
        self.run_dir = run_dir
        calls_dir = run_dir / "calls"
        self.llm_sink = JsonlSink(calls_dir / "llm_calls.jsonl")
        self.embedding_sink = JsonlSink(calls_dir / "embedding_calls.jsonl")
        self.activity_dir = run_dir / "activity"
        self.memory = GraphitiLocalMemory(
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            llm_model=MODELS[args.model_key],
            llm_small_model=MODELS[args.model_key],
            llm_api_key="EMPTY",
            llm_base_url=args.llm_url,
            llm_temperature=0.0,
            llm_max_tokens=512,
            episode_max_chars=3000,
            embedding_model_name=EMBEDDING_MODEL,
            embedding_api_key="EMPTY",
            embedding_base_url=args.embedding_url,
            embedding_dim=EMBEDDING_DIM,
            answer_model=MODELS[args.model_key],
            answer_api_key="EMPTY",
            answer_base_url=args.llm_url,
            answer_temperature=0.0,
            answer_max_tokens=200,
        )
        invalid_path = run_dir / "errors" / "graphiti_json_failures.jsonl"
        invalid_path.parent.mkdir(parents=True, exist_ok=True)
        type(self.memory.graphiti.llm_client)._DEBUG_LOG_PATH = invalid_path
        self._instrument_clients()

    def _instrument_clients(self) -> None:
        llm_activity = self.activity_dir / "llm.activity"
        embedding_activity = self.activity_dir / "embedding.activity"
        llm_client = self.memory.graphiti.llm_client
        llm_client.client = ClientProxy(
            llm_client.client, self.llm_sink, "graphiti_llm", llm_activity
        )
        cross_encoder = self.memory.graphiti.cross_encoder
        cross_encoder.client = ClientProxy(
            cross_encoder.client, self.llm_sink, "graphiti_rerank", llm_activity
        )
        embedder = self.memory.graphiti.embedder
        embedder.client = ClientProxy(
            embedder.client,
            self.embedding_sink,
            "graphiti_embedding",
            embedding_activity,
            embeddings=True,
        )
        self.memory.answer_client = ClientProxy(
            self.memory.answer_client, self.llm_sink, "answer", llm_activity
        )

    async def close(self) -> None:
        try:
            await self.memory.close()
        finally:
            self.llm_sink.close()
            self.embedding_sink.close()


def graph_records(result: Any) -> list[Any]:
    if isinstance(result, tuple):
        return list(result[0])
    if hasattr(result, "records"):
        return list(result.records)
    return list(result)


async def graph_stats(runtime: ZepRuntime, namespace: str) -> dict[str, int]:
    node_result = await runtime.memory.graphiti.driver.execute_query(
        """
        MATCH (n)
        WHERE n.group_id = $group_id
        RETURN count(n) AS nodes,
               count(CASE WHEN n:Episodic THEN 1 END) AS episodes,
               count(CASE WHEN n:Entity THEN 1 END) AS entities,
               count(CASE WHEN n:Community THEN 1 END) AS communities
        """,
        group_id=namespace,
    )
    edge_result = await runtime.memory.graphiti.driver.execute_query(
        """
        MATCH (a)-[r]->(b)
        WHERE a.group_id = $group_id OR b.group_id = $group_id
        RETURN count(r) AS relationships
        """,
        group_id=namespace,
    )
    node_records = graph_records(node_result)
    edge_records = graph_records(edge_result)
    node = dict(node_records[0]) if node_records else {}
    edge = dict(edge_records[0]) if edge_records else {}
    return {
        "nodes": int(node.get("nodes", 0)),
        "episodes": int(node.get("episodes", 0)),
        "entities": int(node.get("entities", 0)),
        "communities": int(node.get("communities", 0)),
        "relationships": int(edge.get("relationships", 0)),
    }


def group_marker(run_dir: Path, group: MemoryGroup) -> Path:
    return run_dir / "build" / "groups" / f"{sanitize_id(group.source_id)}.complete.json"


def group_progress_marker(run_dir: Path, group: MemoryGroup) -> Path:
    return run_dir / "build" / "progress" / f"{sanitize_id(group.source_id)}.json"


async def stored_episode_names(runtime: ZepRuntime, namespace: str) -> list[str]:
    result = await runtime.memory.graphiti.driver.execute_query(
        """
        MATCH (e:Episodic)
        WHERE e.group_id = $group_id
        RETURN e.name AS name
        ORDER BY e.valid_at, e.name
        """,
        group_id=namespace,
    )
    return [str(record["name"]) for record in graph_records(result)]


async def build_one_group(
    runtime: ZepRuntime,
    group: MemoryGroup,
    protocol_hash: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    marker = group_marker(runtime.run_dir, group)
    progress_marker = group_progress_marker(runtime.run_dir, group)
    episodes = group.episodes()
    if marker.exists():
        existing = read_json(marker)
        if (
            existing.get("protocol_hash") == protocol_hash
            and int(existing.get("expected_episodes", -1)) == len(episodes)
        ):
            stats = await graph_stats(runtime, group.namespace)
            if stats["episodes"] == len(episodes):
                return {"source_id": group.source_id, "status": "reused", **stats}
    async with semaphore:
        started = time.perf_counter()
        resume_index = 0
        if progress_marker.exists():
            progress = read_json(progress_marker)
            completed_episodes = int(progress.get("completed_episodes") or 0)
            if (
                progress.get("protocol_hash") == protocol_hash
                and progress.get("namespace") == group.namespace
                and int(progress.get("expected_episodes") or -1) == len(episodes)
                and 0 <= completed_episodes <= len(episodes)
            ):
                stored_names = await stored_episode_names(runtime, group.namespace)
                expected_names = [episode.name for episode in episodes[:completed_episodes]]
                if stored_names == expected_names:
                    resume_index = completed_episodes

        if resume_index:
            print(
                f"build resume {group.source_id} from episode "
                f"{resume_index}/{len(episodes)}",
                flush=True,
            )

        if resume_index == 0:
            token = CALL_CONTEXT.set(
                {
                    "stage": "build_clear",
                    "benchmark": runtime.args.benchmark,
                    "model_key": runtime.args.model_key,
                    "namespace": group.namespace,
                    "source_id": group.source_id,
                }
            )
            try:
                await clear_data(runtime.memory.graphiti.driver, [group.namespace])
            finally:
                CALL_CONTEXT.reset(token)
            atomic_json(
                progress_marker,
                {
                    "source_id": group.source_id,
                    "namespace": group.namespace,
                    "protocol_hash": protocol_hash,
                    "expected_episodes": len(episodes),
                    "completed_episodes": 0,
                    "updated_at": utc_now(),
                },
            )

        for episode_index, episode in enumerate(episodes[resume_index:], start=resume_index):
            token = CALL_CONTEXT.set(
                {
                    "stage": "build",
                    "benchmark": runtime.args.benchmark,
                    "model_key": runtime.args.model_key,
                    "namespace": group.namespace,
                    "source_id": group.source_id,
                    "episode_index": episode_index,
                    "episode_uuid": episode.uuid,
                }
            )
            try:
                await runtime.memory.graphiti.add_episode(
                    name=episode.name,
                    episode_body=episode.body,
                    source_description=episode.source_description,
                    reference_time=episode.reference_time,
                    source=EpisodeType.message,
                    group_id=group.namespace,
                )
            finally:
                CALL_CONTEXT.reset(token)
            atomic_json(
                progress_marker,
                {
                    "source_id": group.source_id,
                    "namespace": group.namespace,
                    "protocol_hash": protocol_hash,
                    "expected_episodes": len(episodes),
                    "completed_episodes": episode_index + 1,
                    "last_episode_name": episode.name,
                    "updated_at": utc_now(),
                },
            )

        stats = await graph_stats(runtime, group.namespace)
        if stats["episodes"] != len(episodes):
            raise RuntimeError(
                f"episode count mismatch for {group.source_id}: {stats['episodes']} != {len(episodes)}"
            )
        completed = {
            "source_id": group.source_id,
            "namespace": group.namespace,
            "protocol_hash": protocol_hash,
            "expected_episodes": len(episodes),
            "completed_at": utc_now(),
            "wall_seconds": round(time.perf_counter() - started, 6),
            "stats": stats,
        }
        atomic_json(marker, completed)
        progress_marker.unlink(missing_ok=True)
        status = "resumed" if resume_index else "built"
        return {"source_id": group.source_id, "status": status, **stats}


async def run_build(
    runtime: ZepRuntime,
    view: DatasetView,
    groups: list[MemoryGroup],
    protocol_hash: str,
) -> None:
    semaphore = asyncio.Semaphore(runtime.args.concurrency)
    tasks = [
        asyncio.create_task(build_one_group(runtime, group, protocol_hash, semaphore))
        for group in groups
    ]
    completed = 0
    failures: list[dict[str, str]] = []
    for future in asyncio.as_completed(tasks):
        try:
            result = await future
            completed += 1
            print(
                f"build {completed}/{len(tasks)} {result['source_id']} {result['status']} "
                f"episodes={result['episodes']}",
                flush=True,
            )
        except Exception as exc:
            failures.append({"error_type": type(exc).__name__, "error": str(exc)})
            print(f"build failure {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    summary = {
        "benchmark": view.benchmark,
        "model_key": runtime.args.model_key,
        "groups": len(groups),
        "completed": completed,
        "failures": failures,
        "finished_at": utc_now(),
    }
    atomic_json(runtime.run_dir / "build" / "summary.json", summary)
    if failures:
        raise RuntimeError(f"{len(failures)} build groups failed")


def item_path(run_dir: Path, stage: str, qid: str) -> Path:
    return run_dir / stage / "items" / f"{sanitize_id(qid)}.json"


def namespace_for_row(view: DatasetView, namespace_prefix: str, row: dict[str, Any]) -> str:
    source_id = (
        str(row["question_id"])
        if view.benchmark == "longmemeval"
        else str(row["locomo_sample_id"])
    )
    return sanitize_id(f"{namespace_prefix}_{view.benchmark}_{source_id}")


async def query_one(
    runtime: ZepRuntime,
    view: DatasetView,
    namespace_prefix: str,
    row: dict[str, Any],
    protocol_hash: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    qid = str(row["question_id"])
    output = item_path(runtime.run_dir, "query", qid)
    if output.exists():
        existing = read_json(output)
        if existing.get("protocol_hash") == protocol_hash:
            return {"qid": qid, "status": "reused"}
    namespace = namespace_for_row(view, namespace_prefix, row)
    async with semaphore:
        started = time.perf_counter()
        token = CALL_CONTEXT.set(
            {
                "stage": "query",
                "benchmark": view.benchmark,
                "model_key": runtime.args.model_key,
                "namespace": namespace,
                "qid": qid,
            }
        )
        try:
            results = await runtime.memory.search(str(row["question"]), namespace)
        finally:
            CALL_CONTEXT.reset(token)
        retrieved = extract_retrieved_facts(results)
        context = build_answer_context(retrieved)
        record = {
            "protocol_hash": protocol_hash,
            "method": f"zep_local_{runtime.args.model_key}",
            "benchmark": view.benchmark,
            "qid": qid,
            "question_type": row.get("question_type"),
            "question": row.get("question"),
            "gold_answer": row.get("answer"),
            "question_date": row.get("question_date"),
            "namespace": namespace,
            "retrieved": retrieved,
            "retrieved_counts": {key.lower(): len(values) for key, values in retrieved.items()},
            "context": context,
            "context_chars": len(context),
            "latency_seconds": round(time.perf_counter() - started, 6),
            "created_at": utc_now(),
        }
        atomic_json(output, record)
        return {"qid": qid, "status": "queried", "context_chars": len(context)}


async def run_query(
    runtime: ZepRuntime,
    view: DatasetView,
    namespace_prefix: str,
    rows: list[dict[str, Any]],
    protocol_hash: str,
) -> None:
    semaphore = asyncio.Semaphore(runtime.args.concurrency)
    tasks = [
        asyncio.create_task(
            query_one(runtime, view, namespace_prefix, row, protocol_hash, semaphore)
        )
        for row in rows
    ]
    failures = []
    completed = 0
    for future in asyncio.as_completed(tasks):
        try:
            result = await future
            completed += 1
            if completed % 25 == 0 or completed == len(tasks):
                print(f"query {completed}/{len(tasks)} last={result['qid']}", flush=True)
        except Exception as exc:
            failures.append({"error_type": type(exc).__name__, "error": str(exc)})
            print(f"query failure {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    atomic_json(
        runtime.run_dir / "query" / "summary.json",
        {"rows": len(rows), "completed": completed, "failures": failures, "finished_at": utc_now()},
    )
    if failures:
        raise RuntimeError(f"{len(failures)} query rows failed")


async def answer_from_context(
    runtime: ZepRuntime,
    question: str,
    context: str,
    reference_date: str | None,
) -> str:
    if not context.strip():
        return "Insufficient context to answer."
    system_prompt = ANSWER_SYSTEM_PROMPT
    user_suffix = "Answer briefly and directly based only on the context."
    if _is_multiple_choice_question(question):
        system_prompt += (
            "\nIf the question is multiple-choice, reply with exactly one uppercase letter: "
            "A, B, C, or D. Do not explain your answer."
        )
        user_suffix = "Reply with exactly one uppercase letter: A, B, C, or D."
    request: dict[str, Any] = {
        "model": MODELS[runtime.args.model_key],
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"# REFERENCE DATE\n{reference_date or 'unknown'}\n\n"
                    f"# CONTEXT\n{context}\n\n"
                    f"# QUESTION\n{question}\n\n{user_suffix}"
                ),
            },
        ],
        "max_tokens": 200,
        "temperature": 0.0,
    }
    if _should_disable_qwen3_thinking(MODELS[runtime.args.model_key]):
        request["extra_body"] = {
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    response = await runtime.memory.answer_client.chat.completions.create(**request)
    return _extract_openai_message_text(response.choices[0].message) or "No response generated"


async def answer_one(
    runtime: ZepRuntime,
    row: dict[str, Any],
    protocol_hash: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    qid = str(row["question_id"])
    query_path = item_path(runtime.run_dir, "query", qid)
    if not query_path.exists():
        raise FileNotFoundError(f"missing query context for {qid}")
    query_record = read_json(query_path)
    output = item_path(runtime.run_dir, "answers", qid)
    if output.exists():
        existing = read_json(output)
        if existing.get("protocol_hash") == protocol_hash:
            return {"qid": qid, "status": "reused"}
    async with semaphore:
        started = time.perf_counter()
        token = CALL_CONTEXT.set(
            {
                "stage": "answer",
                "benchmark": runtime.args.benchmark,
                "model_key": runtime.args.model_key,
                "namespace": query_record["namespace"],
                "qid": qid,
            }
        )
        try:
            answer = await answer_from_context(
                runtime,
                str(row["question"]),
                str(query_record["context"]),
                str(row.get("question_date") or "") or None,
            )
        finally:
            CALL_CONTEXT.reset(token)
        record = {
            "protocol_hash": protocol_hash,
            "method": f"zep_local_{runtime.args.model_key}",
            "benchmark": runtime.args.benchmark,
            "qid": qid,
            "question_type": row.get("question_type"),
            "question": row.get("question"),
            "gold_answer": row.get("answer"),
            "reference_date": row.get("question_date"),
            "model": MODELS[runtime.args.model_key],
            "prompt_version": "memorydata_zep_local_schema_aware",
            "context_source": str(query_path),
            "context_chars": query_record["context_chars"],
            "responses": [{"sample_index": 0, "answer": answer.strip(), "reason": ""}],
            "latency_seconds": round(time.perf_counter() - started, 6),
            "created_at": utc_now(),
        }
        atomic_json(output, record)
        return {"qid": qid, "status": "answered"}


async def run_answer(
    runtime: ZepRuntime,
    rows: list[dict[str, Any]],
    protocol_hash: str,
) -> None:
    semaphore = asyncio.Semaphore(runtime.args.concurrency)
    tasks = [
        asyncio.create_task(answer_one(runtime, row, protocol_hash, semaphore)) for row in rows
    ]
    failures = []
    completed = 0
    for future in asyncio.as_completed(tasks):
        try:
            result = await future
            completed += 1
            if completed % 25 == 0 or completed == len(tasks):
                print(f"answer {completed}/{len(tasks)} last={result['qid']}", flush=True)
        except Exception as exc:
            failures.append({"error_type": type(exc).__name__, "error": str(exc)})
            print(f"answer failure {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    atomic_json(
        runtime.run_dir / "answers" / "summary.json",
        {"rows": len(rows), "completed": completed, "failures": failures, "finished_at": utc_now()},
    )
    if failures:
        raise RuntimeError(f"{len(failures)} answer rows failed")


def aggregate_answers(run_dir: Path) -> Path:
    items = sorted((run_dir / "answers" / "items").glob("*.json"))
    output = run_dir / "answers" / "answers_pass1.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for path in items:
            handle.write(json.dumps(read_json(path), ensure_ascii=False) + "\n")
    atomic_json(
        run_dir / "answers" / "aggregate_manifest.json",
        {"rows": len(items), "output": str(output), "created_at": utc_now()},
    )
    return output


def select_groups(view: DatasetView, args: argparse.Namespace) -> list[MemoryGroup]:
    groups = view.groups
    if args.source_id:
        wanted = set(args.source_id)
        groups = [group for group in groups if group.source_id in wanted]
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    if args.max_sessions is not None:
        groups = [
            MemoryGroup(
                source_id=group.source_id,
                namespace=group.namespace,
                sessions=group.sessions[: args.max_sessions],
                session_ids=group.session_ids[: args.max_sessions],
                dates=group.dates[: args.max_sessions],
            )
            for group in groups
        ]
    return groups


def build_manifest(
    args: argparse.Namespace,
    view: DatasetView,
    groups: list[MemoryGroup],
    rows: list[dict[str, Any]],
    run_dir: Path,
) -> tuple[str, str, str]:
    dataset_hash = sha256_file(view.path)
    group_shape = [
        {
            "source_id": group.source_id,
            "sessions": len(group.sessions),
            "episodes": len(group.episodes()),
        }
        for group in groups
    ]
    build_protocol = {
        "adapter_commit": ADAPTER_COMMIT,
        "graphiti_commit": GRAPHITI_COMMIT,
        "graphiti_version": GRAPHITI_VERSION,
        "model": MODELS[args.model_key],
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "dataset_hash": dataset_hash,
        "group_shape": group_shape,
        "timestamp_policy": "session timestamp plus deterministic microsecond turn offset",
        "episode_policy": "one dialogue turn per add_episode",
    }
    build_hash = stable_hash(build_protocol)
    query_protocol = {
        "build_hash": build_hash,
        "search": "Graphiti COMBINED_HYBRID_SEARCH_CROSS_ENCODER",
        "limit_per_channel": RETRIEVAL_LIMIT,
        "context_schema": ["Edges", "Nodes", "Episodes", "Communities"],
    }
    query_hash = stable_hash(query_protocol)
    answer_protocol = {
        "query_hash": query_hash,
        "answer_prompt_hash": sha256_bytes(ANSWER_SYSTEM_PROMPT.encode("utf-8")),
        "reference_date_policy": "explicit benchmark question_date in user prompt",
        "temperature": 0.0,
        "max_tokens": 200,
    }
    answer_hash = stable_hash(answer_protocol)
    manifest = {
        "created_at": utc_now(),
        "workspace": str(WORKSPACE),
        "run_dir": str(run_dir),
        "benchmark": args.benchmark,
        "model_key": args.model_key,
        "model": MODELS[args.model_key],
        "llm_url": args.llm_url,
        "embedding_url": args.embedding_url,
        "neo4j_uri": args.neo4j_uri,
        "concurrency": args.concurrency,
        "dataset_path": str(view.path),
        "dataset_hash": dataset_hash,
        "dataset_dependencies": {
            str(view.path): dataset_hash,
            **(
                {
                    str(view.data_root / "locomo10.json"): sha256_file(
                        view.data_root / "locomo10.json"
                    )
                }
                if view.benchmark == "locomo"
                and (view.data_root / "locomo10.json").exists()
                else {}
            ),
        },
        "group_count": len(groups),
        "question_row_count": len(rows),
        "group_shape": group_shape,
        "neo4j_version": "5.26.2",
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "build_protocol": build_protocol,
        "build_protocol_hash": build_hash,
        "query_protocol": query_protocol,
        "query_protocol_hash": query_hash,
        "answer_protocol": answer_protocol,
        "answer_protocol_hash": answer_hash,
        "secrets_persisted": False,
    }
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        immutable = ("benchmark", "model_key", "dataset_hash", "build_protocol_hash")
        mismatches = [key for key in immutable if existing.get(key) != manifest.get(key)]
        if mismatches:
            raise RuntimeError(f"run manifest mismatch: {', '.join(mismatches)}")
        manifest["created_at"] = existing.get("created_at", manifest["created_at"])
    atomic_json(manifest_path, manifest)
    return build_hash, query_hash, answer_hash


async def async_main(
    args: argparse.Namespace,
    runtime: ZepRuntime,
    view: DatasetView,
    groups: list[MemoryGroup],
    rows: list[dict[str, Any]],
    namespace_prefix: str,
    build_hash: str,
    query_hash: str,
    answer_hash: str,
) -> None:
    try:
        if args.stage in {"build", "run"}:
            await run_build(runtime, view, groups, build_hash)
        if args.stage in {"query", "run"}:
            await run_query(runtime, view, namespace_prefix, rows, query_hash)
        if args.stage in {"answer", "run"}:
            await run_answer(runtime, rows, answer_hash)
            print(aggregate_answers(runtime.run_dir))
    finally:
        await runtime.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("build", "query", "answer", "aggregate", "run"), default="run")
    parser.add_argument("--benchmark", choices=("longmemeval", "locomo"), required=True)
    parser.add_argument("--model-key", choices=tuple(MODELS), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-root", type=Path, default=WORKSPACE / "data")
    parser.add_argument("--llm-url", required=True)
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--neo4j-uri", required=True)
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="zep-local-revision")
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--source-id", action="append")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    return args


def main() -> None:
    args = parse_args()
    namespace_prefix = sanitize_id(f"zep_{args.model_key}_{args.run_name}")
    run_dir = args.run_root.resolve() / args.model_key / args.benchmark
    run_dir.mkdir(parents=True, exist_ok=True)
    view = DatasetView(args.benchmark, args.data_root.resolve(), namespace_prefix)
    groups = select_groups(view, args)
    group_ids = {group.source_id for group in groups}
    rows = view.rows_for_groups(group_ids)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    build_hash, query_hash, answer_hash = build_manifest(args, view, groups, rows, run_dir)
    if args.stage == "aggregate":
        print(aggregate_answers(run_dir))
        return
    # MemoryData initializes Neo4j on a private event loop. Run the complete
    # concurrent pipeline on that same loop so async driver futures never cross
    # event-loop boundaries.
    runtime = ZepRuntime(args, run_dir)
    try:
        runtime.memory._run_sync(
            async_main(
                args,
                runtime,
                view,
                groups,
                rows,
                namespace_prefix,
                build_hash,
                query_hash,
                answer_hash,
            )
        )
    finally:
        if not runtime.memory._loop.is_closed():
            runtime.memory._loop.close()


if __name__ == "__main__":
    main()
