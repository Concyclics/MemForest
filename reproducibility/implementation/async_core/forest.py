"""Forest write control layer for session/actor/scene trees and atomic facts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import uuid
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from memory_forest.api.prompts_manager import PromptManager
from memory_forest.core.async_utils import run_blocking
from memory_forest.core.bplustree import AsyncBPlusTree
from memory_forest.core.entity_resolver import EntityResolver
from memory_forest.core.extraction import extract_atomic_facts_raw, extract_vector_labels
from memory_forest.core.fact_canonicalizer import FactCanonicalizer
from memory_forest.core.fact_manager import FactManager, RootSummaryIndexManager
from memory_forest.core.models import AtomicFact, BPlusNodeMeta, ContentLeaf, RawContent
from memory_forest.core.utils import unix_to_time_text

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass
class DialogueInput:
    speaker_id: str
    speaker_name: str
    listener_name: str
    content: str
    speaker_tag: str = "user"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DialogueInsertRequest:
    session_id: str
    dialogue: DialogueInput
    timestamp: float


@dataclass
class DialogueInsertResult:
    session_id: str
    content_id: str
    inserted_key: Tuple[float, int]
    fact_ids: List[str]
    extracted_fact_count: int
    accepted_fact_count: int = 0
    duplicate_fact_count: int = 0
    extraction_error: Optional[str] = None


@dataclass
class PendingWriteReceipt:
    write_id: str
    session_id: str
    content_id: str
    inserted_key: Tuple[float, int]
    queued_at: float
    state: str
    error: Optional[str] = None


@dataclass
class BatchInsertResult:
    total: int
    succeeded: int
    failed: int
    results: List[DialogueInsertResult]
    errors: List[str]


@dataclass
class ForestTimingEvent:
    stage: str
    tree_type: Optional[str]
    tree_id: Optional[str]
    duration_sec: float
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForestSnapshotReport:
    snapshot_dir: str
    saved_tree_counts: Dict[str, int]
    loaded_tree_counts: Dict[str, int]
    warnings: List[str]


@dataclass
class ForestUpdateReport:
    drained_jobs: int
    flushed_tree_count: int
    flushed_tree_ids: List[str]
    failed_jobs: List[str]
    failed_trees: List[str]
    duration_sec: float


@dataclass
class TreeGraphEdge:
    neighbor_tree_id: str
    raw_weight: int
    normalized_weight: float


@dataclass
class TreeGraphNodeStats:
    total_outgoing_raw_weight: int = 0


@dataclass
class TreeGraph:
    adjacency: Dict[str, List[TreeGraphEdge]] = field(default_factory=dict)
    node_stats: Dict[str, TreeGraphNodeStats] = field(default_factory=dict)
    last_built_from_fact_count: int = 0



@dataclass
class _PendingWriteJob:
    receipt: PendingWriteReceipt
    task: asyncio.Task[None]


@dataclass
class _PreparedTurn:
    session_id: str
    req: DialogueInsertRequest
    key: Tuple[float, int]
    content_id: str
    raw: RawContent


def _normalize_entity_key(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return text.strip().lower()


def _singularize_entity_key(text: Any) -> str:
    raw = _normalize_entity_key(text)
    if not raw:
        return ""

    tokens = raw.split()
    if not tokens:
        return raw

    token = tokens[-1]
    if len(token) < 4 or not re.fullmatch(r"[a-z]+", token):
        return raw
    if token in {"news", "series", "species", "headquarters"}:
        return raw
    if token.endswith(("ss", "us", "is", "ous", "ics")):
        return raw

    singular = token
    if token.endswith("ies") and len(token) > 4:
        if len(token) >= 5 and token[-4] not in {"a", "e", "i", "o", "u"}:
            singular = token[:-3] + "y"
        else:
            singular = token[:-1]
    elif re.search(r"(ches|shes|xes|zes|oes)$", token):
        singular = token[:-2]
    elif token.endswith("ses"):
        singular = token
    elif token.endswith("s"):
        singular = token[:-1]

    if not singular or singular == token:
        return raw
    tokens[-1] = singular
    return " ".join(tokens).strip()


def _entity_key_tokens(text: Any) -> List[str]:
    raw = _normalize_entity_key(text)
    if not raw:
        return []
    return [tok for tok in re.findall(r"[a-z0-9]+", raw) if tok]


def _scene_key_tokens(text: Any) -> List[str]:
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "for",
        "to",
        "in",
        "on",
        "at",
        "with",
        "by",
        "from",
    }
    return [tok for tok in _entity_key_tokens(text) if tok not in stop]


def _strip_snapshot_digest_suffix(text: str) -> str:
    raw = str(text).strip()
    match = re.match(r"^(?P<base>.+)_[0-9a-f]{10}$", raw)
    if not match:
        return raw
    return str(match.group("base")).strip()


def _session_tree_id(session_id: str) -> str:
    return f"session:{session_id}"


def _actor_tree_id(actor_key: str) -> str:
    return f"actor:{actor_key}"


def _scene_tree_id(scene_key: str) -> str:
    return f"scene:{scene_key}"


def _entity_resolution_cache_key(fact: AtomicFact) -> str:
    actors = sorted(
        {
            str(x).strip().lower()
            for x in (getattr(fact, "actors_raw", []) or [])
            if str(x).strip()
        }
    )
    scene = str(getattr(fact, "scene_raw", "") or "").strip().lower()
    speaker = str(getattr(fact, "source_speaker_name", "") or "").strip().lower()
    return json.dumps(
        {
            "actors": actors,
            "scene": scene,
            "speaker": speaker,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


class SessionForestManager:
    """Maintain one session B+ tree per session_id."""

    def __init__(
        self,
        prompt_manager: PromptManager,
        order: int = 4,
        leaf_capacity: int = 1,
        summary_parallel_limit: int = 8,
        summary_executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        self.prompt_manager = prompt_manager
        self.order = int(order)
        self.leaf_capacity = max(1, int(leaf_capacity))
        self.summary_parallel_limit = int(summary_parallel_limit)
        self.summary_executor = summary_executor
        self.trees: Dict[str, AsyncBPlusTree] = {}
        self._map_lock = asyncio.Lock()

    async def get_tree(self, session_id: str) -> AsyncBPlusTree:
        key = str(session_id)
        async with self._map_lock:
            tree = self.trees.get(key)
            if tree is None:
                tree = AsyncBPlusTree(
                    order=self.order,
                    leaf_capacity=self.leaf_capacity,
                    prompt_manager=self.prompt_manager,
                    summary_parallel_limit=self.summary_parallel_limit,
                    summary_prompt_key="node_summarizer",
                    summary_executor=self.summary_executor,
                )
                self.trees[key] = tree
            return tree


class ActorForestManager:
    """Maintain one actor B+ tree per normalized actor key."""

    def __init__(
        self,
        prompt_manager: PromptManager,
        order: int = 4,
        leaf_capacity: int = 1,
        summary_parallel_limit: int = 8,
        summary_executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        self.prompt_manager = prompt_manager
        self.order = int(order)
        self.leaf_capacity = max(1, int(leaf_capacity))
        self.summary_parallel_limit = int(summary_parallel_limit)
        self.summary_executor = summary_executor
        self.trees: Dict[str, AsyncBPlusTree] = {}
        self._map_lock = asyncio.Lock()

    async def get_tree(self, actor_key: str) -> AsyncBPlusTree:
        key = _normalize_entity_key(actor_key)
        async with self._map_lock:
            tree = self.trees.get(key)
            if tree is None:
                tree = AsyncBPlusTree(
                    order=self.order,
                    leaf_capacity=self.leaf_capacity,
                    prompt_manager=self.prompt_manager,
                    summary_parallel_limit=self.summary_parallel_limit,
                    summary_prompt_key="actor_node_summarizer",
                    summary_context={"focal_actor": key},
                    summary_executor=self.summary_executor,
                )
                self.trees[key] = tree
            return tree


class SceneForestManager:
    """Maintain one scene B+ tree per normalized scene key."""

    def __init__(
        self,
        prompt_manager: PromptManager,
        order: int = 4,
        leaf_capacity: int = 1,
        summary_parallel_limit: int = 8,
        summary_executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        self.prompt_manager = prompt_manager
        self.order = int(order)
        self.leaf_capacity = max(1, int(leaf_capacity))
        self.summary_parallel_limit = int(summary_parallel_limit)
        self.summary_executor = summary_executor
        self.trees: Dict[str, AsyncBPlusTree] = {}
        self._map_lock = asyncio.Lock()

    async def get_tree(self, scene_key: str) -> AsyncBPlusTree:
        key = _normalize_entity_key(scene_key)
        async with self._map_lock:
            tree = self.trees.get(key)
            if tree is None:
                tree = AsyncBPlusTree(
                    order=self.order,
                    leaf_capacity=self.leaf_capacity,
                    prompt_manager=self.prompt_manager,
                    summary_parallel_limit=self.summary_parallel_limit,
                    summary_prompt_key="scene_node_summarizer",
                    summary_context={"focal_scene": key},
                    summary_executor=self.summary_executor,
                )
                self.trees[key] = tree
            return tree


class Forest:
    """
    Write control layer:
    session tree + atomic facts + actor/scene trees.
    """

    def __init__(
        self,
        *,
        prompt_manager: PromptManager,
        fact_manager: FactManager,
        config: Optional[Dict[str, Any]] = None,
        root_index_manager: Optional[RootSummaryIndexManager] = None,
        tree_order: int = 4,
        timing_callback: Optional[Callable[[ForestTimingEvent], None]] = None,
    ) -> None:
        cfg = config or {}
        self.config = cfg
        self.prompt_manager = prompt_manager
        self.fact_manager = fact_manager
        concurrency_cfg = cfg.get("concurrency", {}) if isinstance(cfg.get("concurrency"), dict) else {}
        extraction_workers = int(concurrency_cfg.get("extraction_workers", 16))
        summary_workers = int(concurrency_cfg.get("summary_workers", 8))
        self._extraction_pool = ThreadPoolExecutor(max_workers=max(1, extraction_workers))
        self._summary_pool = ThreadPoolExecutor(max_workers=max(1, summary_workers))

        summary_parallel_limit = int(cfg.get("summarization", {}).get("max_parallel_llm", 8))
        tree_cfg = cfg.get("tree", {}) if isinstance(cfg.get("tree"), dict) else {}
        order_cfg = tree_cfg.get("order", {}) if isinstance(tree_cfg.get("order"), dict) else {}
        leaf_cfg = tree_cfg.get("leaf_capacity", {}) if isinstance(tree_cfg.get("leaf_capacity"), dict) else {}
        session_order = int(order_cfg.get("session", tree_order))
        actor_order = int(order_cfg.get("actor", tree_order))
        scene_order = int(order_cfg.get("scene", tree_order))
        session_leaf_capacity = int(leaf_cfg.get("session", 1))
        actor_leaf_capacity = int(leaf_cfg.get("actor", 1))
        scene_leaf_capacity = int(leaf_cfg.get("scene", 1))
        self.sessions = SessionForestManager(
            prompt_manager=prompt_manager,
            order=session_order,
            leaf_capacity=session_leaf_capacity,
            summary_parallel_limit=summary_parallel_limit,
            summary_executor=self._summary_pool,
        )
        self.actors = ActorForestManager(
            prompt_manager=prompt_manager,
            order=actor_order,
            leaf_capacity=actor_leaf_capacity,
            summary_parallel_limit=summary_parallel_limit,
            summary_executor=self._summary_pool,
        )
        self.scenes = SceneForestManager(
            prompt_manager=prompt_manager,
            order=scene_order,
            leaf_capacity=scene_leaf_capacity,
            summary_parallel_limit=summary_parallel_limit,
            summary_executor=self._summary_pool,
        )
        root_base_dir = str(self.fact_manager.facts_path.parent)
        self.root_index_manager = root_index_manager or RootSummaryIndexManager(
            base_dir=root_base_dir,
            vector_dim=self.fact_manager.vector_dim,
        )
        self.fact_canonicalizer = FactCanonicalizer(config=cfg)
        self.entity_resolver = EntityResolver(
            root_index_manager=self.root_index_manager,
            prompt_manager=self.prompt_manager,
            config=cfg.get("entity_resolution", {})
            if isinstance(cfg.get("entity_resolution"), dict)
            else {},
        )

        extraction_cfg = cfg.get("extraction", {}) if isinstance(cfg.get("extraction"), dict) else {}
        self.history_slide_window = int(
            extraction_cfg.get(
                "history_slide_window",
                extraction_cfg.get("history_window", 4),
            )
        )
        self.history_text_max_chars = int(extraction_cfg.get("history_text_max_chars", 0))
        self.timestamp_epsilon_sec = float(extraction_cfg.get("timestamp_epsilon_sec", 1e-3))
        self.max_parallel_sessions = int(concurrency_cfg.get("max_parallel_sessions", 8))
        self.max_extraction_tasks = int(concurrency_cfg.get("max_extraction_tasks", 4))
        self.max_parallel_tree_inserts = int(concurrency_cfg.get("max_parallel_tree_inserts", 16))
        self.max_parallel_tree_updates = int(concurrency_cfg.get("max_parallel_tree_updates", 16))
        self._timing_callback = timing_callback
        memory_cfg = cfg.get("memory", {}) if isinstance(cfg.get("memory"), dict) else {}
        fact_dedup_cfg = (
            memory_cfg.get("atomic_fact_dedup", {})
            if isinstance(memory_cfg.get("atomic_fact_dedup"), dict)
            else {}
        )
        exact_cfg = (
            fact_dedup_cfg.get("exact_text_time", {})
            if isinstance(fact_dedup_cfg.get("exact_text_time"), dict)
            else {}
        )
        llm_cfg = (
            fact_dedup_cfg.get("llm_text_time", {})
            if isinstance(fact_dedup_cfg.get("llm_text_time"), dict)
            else {}
        )
        self.atomic_fact_dedup_enabled = bool(fact_dedup_cfg.get("enabled", False))
        self.atomic_fact_exact_dedup_enabled = bool(exact_cfg.get("enabled", self.atomic_fact_dedup_enabled))
        self.atomic_fact_llm_dedup_enabled = bool(llm_cfg.get("enabled", False))
        self.atomic_fact_llm_similarity_threshold = float(llm_cfg.get("similarity_threshold", 0.82))
        self.atomic_fact_llm_top_k_candidates = int(llm_cfg.get("top_k_candidates", 8))
        self.atomic_fact_llm_max_time_gap_seconds = float(llm_cfg.get("max_time_gap_seconds", 0.0))
        self.atomic_fact_llm_max_checks_per_fact = int(llm_cfg.get("max_llm_checks_per_fact", 4))
        self.atomic_fact_llm_prompt_key = str(llm_cfg.get("prompt_key", "atomic_fact_dedup_judge")).strip() or (
            "atomic_fact_dedup_judge"
        )

        self._fact_manager_lock = asyncio.Lock()
        self._root_index_lock = asyncio.Lock()
        self._pending_lock = asyncio.Lock()
        self._dirty_lock = asyncio.Lock()
        self._pending_jobs: Dict[str, _PendingWriteJob] = {}
        self._dirty_trees: Dict[str, Tuple[str, AsyncBPlusTree]] = {}

        self._session_seq: Dict[str, int] = defaultdict(int)
        self._session_last_ts: Dict[str, float] = {}
        self._session_seq_locks: Dict[str, asyncio.Lock] = {}
        self._session_seq_map_lock = asyncio.Lock()

        self._entity_seq: Dict[str, Dict[str, int]] = {
            "actor": defaultdict(int),
            "scene": defaultdict(int),
        }
        self._entity_seq_locks: Dict[str, Dict[str, asyncio.Lock]] = {"actor": {}, "scene": {}}
        self._entity_seq_map_lock = asyncio.Lock()
        entity_cfg = cfg.get("entity_resolution", {}) if isinstance(cfg.get("entity_resolution"), dict) else {}
        self._entity_resolution_cache_max_size = max(1, int(entity_cfg.get("llm_cache_size", 20000)))
        self._entity_resolution_cache: "OrderedDict[str, Tuple[List[str], str]]" = OrderedDict()
        self._entity_resolution_cache_lock = asyncio.Lock()
        self.tree_graph = TreeGraph()

    def close(self) -> None:
        extraction_pool = getattr(self, "_extraction_pool", None)
        if extraction_pool is not None:
            try:
                extraction_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._extraction_pool = None

        summary_pool = getattr(self, "_summary_pool", None)
        if summary_pool is not None:
            try:
                summary_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._summary_pool = None

    def __del__(self) -> None:
        self.close()

    async def append_dialogue(
        self,
        session_id: str,
        dialogue: Dict[str, Any],
        timestamp: float,
    ) -> PendingWriteReceipt:
        req = self._coerce_request(
            {
                "session_id": session_id,
                "dialogue": dialogue,
                "timestamp": timestamp,
            }
        )

        ts, seq = await self._allocate_session_slot(req.session_id, float(req.timestamp))
        key = (ts, seq)
        content_id = f"{req.session_id}#turn_{seq:08d}"
        session_tree = await self.sessions.get_tree(req.session_id)
        leaf = ContentLeaf(
            content_id=content_id,
            session_id=req.session_id,
            turn_index=int(seq),
            speaker=req.dialogue.speaker_tag,
            timestamp=ts,
            text=req.dialogue.content,
            listener_name=req.dialogue.listener_name,
            fact_ids=set(),
        )
        t_insert = perf_counter()
        await session_tree.insert(key, leaf, defer_meta_update=True)
        exec_sec = perf_counter() - t_insert
        self._emit_timing(
            stage="tree_batch_insert_queue_wait",
            tree_type="session",
            tree_id=_session_tree_id(req.session_id),
            duration_sec=0.0,
            extra={"item_count": 1, "sem_wait_sec": 0.0, "lock_wait_sec": 0.0},
        )
        self._emit_timing(
            stage="tree_batch_insert_exec",
            tree_type="session",
            tree_id=_session_tree_id(req.session_id),
            duration_sec=exec_sec,
            extra={"item_count": 1},
        )
        self._emit_timing(
            stage="tree_batch_insert",
            tree_type="session",
            tree_id=_session_tree_id(req.session_id),
            duration_sec=exec_sec,
            extra={"item_count": 1, "queue_wait_sec": 0.0, "exec_sec": exec_sec},
        )
        await self._mark_dirty_tree("session", _session_tree_id(req.session_id), session_tree)

        receipt = PendingWriteReceipt(
            write_id=uuid.uuid4().hex,
            session_id=req.session_id,
            content_id=content_id,
            inserted_key=key,
            queued_at=perf_counter(),
            state="queued",
            error=None,
        )
        task = asyncio.create_task(
            self._run_pending_dialogue_job(
                receipt=receipt,
                req=req,
                key=key,
                content_id=content_id,
                leaf=leaf,
            )
        )
        async with self._pending_lock:
            self._pending_jobs[receipt.write_id] = _PendingWriteJob(receipt=receipt, task=task)
        return receipt

    async def append_many(
        self,
        items: List[DialogueInsertRequest],
        *,
        auto_update: bool = True,
    ) -> BatchInsertResult:
        total = len(items)
        parsed: List[Tuple[int, DialogueInsertRequest]] = []
        errors: List[str] = []
        for idx, item in enumerate(items):
            try:
                parsed.append((idx, self._coerce_request(item)))
            except Exception as exc:
                errors.append(f"item[{idx}] invalid: {exc}")

        grouped: Dict[str, List[Tuple[int, DialogueInsertRequest]]] = defaultdict(list)
        for idx, req in parsed:
            grouped[req.session_id].append((idx, req))

        prepared_turns: List[_PreparedTurn] = []
        session_items_by_id: Dict[str, List[Tuple[Tuple[float, int], ContentLeaf]]] = defaultdict(list)
        session_tree_by_id: Dict[str, AsyncBPlusTree] = {}
        for session_id, reqs in grouped.items():
            ordered = sorted(reqs, key=lambda x: (float(x[1].timestamp), x[0]))
            session_tree = await self.sessions.get_tree(session_id)
            session_tree_by_id[session_id] = session_tree
            pending_history: List[Tuple[Tuple[float, int], Dict[str, str]]] = []
            for _, req in ordered:
                ts, seq = await self._allocate_session_slot(session_id, float(req.timestamp))
                key = (ts, seq)
                content_id = f"{session_id}#turn_{seq:08d}"
                history = await self._session_history_before(
                    session_id=session_id,
                    end_key=(ts, seq - 1),
                    pending=pending_history,
                )
                raw = RawContent(
                    speaker_id=req.dialogue.speaker_id,
                    speaker_tag=req.dialogue.speaker_tag,
                    speaker_name=req.dialogue.speaker_name,
                    content=req.dialogue.content,
                    timestamp=unix_to_time_text(ts, "UTC"),
                    listener_name=req.dialogue.listener_name,
                    session_id=session_id,
                    content_id=content_id,
                    history=history,
                )
                prepared_turns.append(
                    _PreparedTurn(
                        session_id=session_id,
                        req=req,
                        key=key,
                        content_id=content_id,
                        raw=raw,
                    )
                )
                pending_history.append(
                    (
                        key,
                        {"speaker": req.dialogue.speaker_tag, "text": req.dialogue.content},
                    )
                )

        extraction_sem = asyncio.Semaphore(max(1, self.max_extraction_tasks))
        extraction_tasks = [self._extract_for_raw(raw=turn.raw, semaphore=extraction_sem) for turn in prepared_turns]
        t_extract = perf_counter()
        extraction_results = await asyncio.gather(*extraction_tasks)
        self._emit_timing(
            stage="extraction",
            tree_type=None,
            tree_id=None,
            duration_sec=perf_counter() - t_extract,
            extra={"turn_count": len(prepared_turns)},
        )

        fact_context_rows: List[Tuple[AtomicFact, str, str, str]] = []
        raw_facts_by_content: Dict[str, List[AtomicFact]] = {}
        extract_error_by_content: Dict[str, Optional[str]] = {}
        for turn, (facts, labels, extract_error) in zip(prepared_turns, extraction_results):
            raw_facts_by_content[turn.content_id] = list(facts)
            extract_error_by_content[turn.content_id] = extract_error
            if extract_error:
                continue
            for idx, fact in enumerate(facts):
                source_key = f"{turn.session_id}|{turn.content_id}|{idx}|{fact.fact_text}"
                fact_context_rows.append((fact, turn.session_id, turn.content_id, source_key))

        accepted_rows = await self._dedupe_fact_context_rows(fact_context_rows)
        all_facts = [row[0] for row in accepted_rows]
        if all_facts:
            self._harmonize_entities_in_facts(all_facts)
        if all_facts:
            async with self._fact_manager_lock:
                source_keys = [row[3] for row in accepted_rows]
                self.fact_manager.add_facts(all_facts, source_keys=source_keys, persist=False)

        accepted_fact_ids_by_content: Dict[str, List[str]] = defaultdict(list)
        for fact, _, cid, _ in accepted_rows:
            accepted_fact_ids_by_content[cid].append(fact.fact_id)
        duplicate_fact_count_by_content: Dict[str, int] = defaultdict(int)
        for turn in prepared_turns:
            raw_count = len(raw_facts_by_content.get(turn.content_id, []))
            accepted_count = len(accepted_fact_ids_by_content.get(turn.content_id, []))
            duplicate_fact_count_by_content[turn.content_id] = max(0, raw_count - accepted_count)

        results: List[DialogueInsertResult] = []
        for turn in prepared_turns:
            fact_ids = list(accepted_fact_ids_by_content.get(turn.content_id, []))
            results.append(
                DialogueInsertResult(
                    session_id=turn.session_id,
                    content_id=turn.content_id,
                    inserted_key=turn.key,
                    fact_ids=fact_ids,
                    extracted_fact_count=len(raw_facts_by_content.get(turn.content_id, [])),
                    accepted_fact_count=len(fact_ids),
                    duplicate_fact_count=int(duplicate_fact_count_by_content.get(turn.content_id, 0)),
                    extraction_error=extract_error_by_content.get(turn.content_id),
                )
            )
            session_items_by_id[turn.session_id].append(
                (
                    turn.key,
                    ContentLeaf(
                        content_id=turn.content_id,
                        session_id=turn.session_id,
                        turn_index=int(turn.key[1]),
                        speaker=turn.req.dialogue.speaker_tag,
                        timestamp=float(turn.key[0]),
                        text=turn.req.dialogue.content,
                        listener_name=turn.req.dialogue.listener_name,
                        fact_ids=set(fact_ids),
                    ),
                )
            )

        facts_by_id: Dict[str, AtomicFact] = {}
        content_id_by_fact_id: Dict[str, str] = {}
        session_id_by_fact_id: Dict[str, str] = {}
        for fact, sid, cid, _ in fact_context_rows:
            facts_by_id[fact.fact_id] = fact
            content_id_by_fact_id[fact.fact_id] = cid
            session_id_by_fact_id[fact.fact_id] = sid
        all_facts = list(facts_by_id.values())

        insert_requests: List[Tuple[str, str, AsyncBPlusTree, List[Tuple[Tuple[float, int], Any]]]] = []
        for sid, session_items in session_items_by_id.items():
            if session_items:
                insert_requests.append(("session", _session_tree_id(sid), session_tree_by_id[sid], session_items))

        per_fact_actor_tree_ids: Dict[str, Set[str]] = defaultdict(set)
        per_fact_scene_tree_ids: Dict[str, Set[str]] = defaultdict(set)
        if all_facts:
            actor_groups = self._group_facts_by_actor_norm(all_facts)
            for actor_key, facts in actor_groups.items():
                items_by_tree = await self._build_entity_items("actor", actor_key, facts)
                if items_by_tree:
                    actor_tree = await self.actors.get_tree(actor_key)
                    actor_tree_id = _actor_tree_id(actor_key)
                    insert_requests.append(("actor", actor_tree_id, actor_tree, items_by_tree))
                for fact in facts:
                    per_fact_actor_tree_ids[fact.fact_id].add(_actor_tree_id(actor_key))

            scene_groups = self._group_facts_by_scene_norm(all_facts)
            for scene_key, facts in scene_groups.items():
                items_by_tree = await self._build_entity_items("scene", scene_key, facts)
                if items_by_tree:
                    scene_tree = await self.scenes.get_tree(scene_key)
                    scene_tree_id = _scene_tree_id(scene_key)
                    insert_requests.append(("scene", scene_tree_id, scene_tree, items_by_tree))
                for fact in facts:
                    per_fact_scene_tree_ids[fact.fact_id].add(_scene_tree_id(scene_key))

        inserted_trees, insert_errors = await self._run_tree_inserts(insert_requests, defer_meta_update=True)
        errors.extend(insert_errors)
        await self._mark_dirty_trees(inserted_trees)

        if all_facts:
            async with self._fact_manager_lock:
                for fact in all_facts:
                    sid = session_id_by_fact_id.get(fact.fact_id, "")
                    cid = content_id_by_fact_id.get(fact.fact_id, "")
                    if not sid or not cid:
                        continue
                    self.fact_manager.upsert_fact_links(
                        fact.fact_id,
                        session_id=sid,
                        session_tree_id=_session_tree_id(sid),
                        actor_tree_ids=sorted(per_fact_actor_tree_ids.get(fact.fact_id, set())),
                        scene_tree_ids=sorted(per_fact_scene_tree_ids.get(fact.fact_id, set())),
                        content_id=cid,
                        persist=False,
                    )

        if auto_update:
            update_report = await self.update(include_pending_jobs=False)
            errors.extend(update_report.failed_trees)

        results.sort(key=lambda x: (x.session_id, x.inserted_key))
        succeeded = len(results)
        failed = max(0, total - succeeded)
        return BatchInsertResult(total=total, succeeded=succeeded, failed=failed, results=results, errors=errors)

    async def wait_pending_jobs(self) -> int:
        async with self._pending_lock:
            jobs = list(self._pending_jobs.values())
        if not jobs:
            return 0
        await asyncio.gather(*(job.task for job in jobs), return_exceptions=True)
        return len(jobs)

    async def update(self, *, include_pending_jobs: bool = True) -> ForestUpdateReport:
        t0 = perf_counter()
        drained_jobs = 0
        failed_jobs: List[str] = []
        if include_pending_jobs:
            async with self._pending_lock:
                job_snapshot = list(self._pending_jobs.values())
            drained_jobs = len(job_snapshot)
            if job_snapshot:
                outcomes = await asyncio.gather(*(job.task for job in job_snapshot), return_exceptions=True)
                for job, outcome in zip(job_snapshot, outcomes):
                    if isinstance(outcome, Exception) or job.receipt.state == "failed":
                        failed_jobs.append(job.receipt.write_id)

        dirty_snapshot = await self._take_dirty_tree_snapshot()
        if not dirty_snapshot:
            return ForestUpdateReport(
                drained_jobs=drained_jobs,
                flushed_tree_count=0,
                flushed_tree_ids=[],
                failed_jobs=failed_jobs,
                failed_trees=[],
                duration_sec=perf_counter() - t0,
            )

        sem = asyncio.Semaphore(max(1, self.max_parallel_tree_updates))
        flush_tasks = [
            asyncio.create_task(self._flush_tree(tree_type, tree_id, tree, sem))
            for tree_id, (tree_type, tree) in dirty_snapshot.items()
        ]
        flushed_trees: Dict[str, Tuple[str, AsyncBPlusTree]] = {}
        failed_trees: List[str] = []
        if flush_tasks:
            flush_results = await asyncio.gather(*flush_tasks, return_exceptions=True)
            for out in flush_results:
                if isinstance(out, Exception):
                    failed_trees.append(f"tree flush failed: {out}")
                    continue
                tree_type, tree_id, tree_obj, err = out
                if err:
                    failed_trees.append(err)
                    continue
                if tree_obj is not None:
                    flushed_trees[tree_id] = (tree_type, tree_obj)

        if failed_trees:
            retry_dirty: Dict[str, Tuple[str, AsyncBPlusTree]] = {}
            for msg in failed_trees:
                for tree_id, (tree_type, tree) in dirty_snapshot.items():
                    if tree_id in msg:
                        retry_dirty[tree_id] = (tree_type, tree)
            await self._mark_dirty_trees(retry_dirty)

        await self._update_root_vectors(flushed_trees)
        self.refresh_tree_graph()
        return ForestUpdateReport(
            drained_jobs=drained_jobs,
            flushed_tree_count=len(flushed_trees),
            flushed_tree_ids=sorted(flushed_trees.keys()),
            failed_jobs=failed_jobs,
            failed_trees=failed_trees,
            duration_sec=perf_counter() - t0,
        )

    async def get_tree_by_tree_id(self, tree_id: str) -> Tuple[str, str, AsyncBPlusTree]:
        """
        Resolve tree id into concrete tree object.

        Returns:
            (tree_type, tree_key, tree)
        """

        raw = str(tree_id).strip()
        if ":" not in raw:
            raise ValueError("tree_id must be in format `session:*`, `actor:*`, or `scene:*`.")
        prefix, key_raw = raw.split(":", 1)
        tree_type = prefix.strip().lower()
        key = key_raw.strip()
        if not key:
            raise ValueError("tree_id suffix is empty.")

        if tree_type == "session":
            tree = self.sessions.trees.get(key)
            if tree is None:
                key2 = _strip_snapshot_digest_suffix(key)
                tree = self.sessions.trees.get(key2)
                if tree is not None:
                    key = key2
            if tree is None:
                raise KeyError(f"Unknown session tree: {raw}")
            return tree_type, key, tree
        if tree_type == "actor":
            norm_key = _normalize_entity_key(key)
            tree = self.actors.trees.get(norm_key)
            if tree is None:
                norm_key2 = _normalize_entity_key(_strip_snapshot_digest_suffix(norm_key))
                tree = self.actors.trees.get(norm_key2)
                if tree is not None:
                    norm_key = norm_key2
            if tree is None:
                raise KeyError(f"Unknown actor tree: {raw}")
            return tree_type, norm_key, tree
        if tree_type == "scene":
            norm_key = _normalize_entity_key(key)
            tree = self.scenes.trees.get(norm_key)
            if tree is None:
                norm_key2 = _normalize_entity_key(_strip_snapshot_digest_suffix(norm_key))
                tree = self.scenes.trees.get(norm_key2)
                if tree is not None:
                    norm_key = norm_key2
            if tree is None:
                raise KeyError(f"Unknown scene tree: {raw}")
            return tree_type, norm_key, tree
        raise ValueError(f"Unsupported tree type in tree_id: {raw}")

    def refresh_tree_graph(self) -> TreeGraph:
        raw_edges: Dict[Tuple[str, str], int] = defaultdict(int)
        outgoing_raw_weight: Dict[str, int] = defaultdict(int)
        known_tree_ids = self._all_known_tree_ids()

        for links in getattr(self.fact_manager, "fact_links", {}).values():
            tree_ids: List[str] = []
            tree_ids.extend([str(x).strip() for x in getattr(links, "session_tree_ids", []) if str(x).strip()])
            tree_ids.extend([str(x).strip() for x in getattr(links, "actor_tree_ids", []) if str(x).strip()])
            tree_ids.extend([str(x).strip() for x in getattr(links, "scene_tree_ids", []) if str(x).strip()])
            unique_tree_ids = sorted(set(tree_ids))
            if not unique_tree_ids:
                continue
            known_tree_ids.update(unique_tree_ids)
            for idx, left in enumerate(unique_tree_ids):
                for right in unique_tree_ids[idx + 1 :]:
                    raw_edges[(left, right)] += 1

        for (left, right), raw_weight in raw_edges.items():
            outgoing_raw_weight[left] += int(raw_weight)
            outgoing_raw_weight[right] += int(raw_weight)

        adjacency: Dict[str, List[TreeGraphEdge]] = {tree_id: [] for tree_id in sorted(known_tree_ids)}
        for (left, right), raw_weight in raw_edges.items():
            left_out = int(outgoing_raw_weight.get(left, 0))
            right_out = int(outgoing_raw_weight.get(right, 0))
            if left_out <= 0 or right_out <= 0:
                normalized_weight = 0.0
            else:
                normalized_weight = (float(raw_weight) / float(left_out)) * (float(raw_weight) / float(right_out))
            edge_lr = TreeGraphEdge(
                neighbor_tree_id=str(right),
                raw_weight=int(raw_weight),
                normalized_weight=float(normalized_weight),
            )
            edge_rl = TreeGraphEdge(
                neighbor_tree_id=str(left),
                raw_weight=int(raw_weight),
                normalized_weight=float(normalized_weight),
            )
            adjacency.setdefault(left, []).append(edge_lr)
            adjacency.setdefault(right, []).append(edge_rl)

        for tree_id in adjacency.keys():
            adjacency[tree_id].sort(
                key=lambda row: (
                    -float(row.normalized_weight),
                    -int(row.raw_weight),
                    str(row.neighbor_tree_id),
                )
            )

        node_stats = {
            tree_id: TreeGraphNodeStats(total_outgoing_raw_weight=int(outgoing_raw_weight.get(tree_id, 0)))
            for tree_id in adjacency.keys()
        }
        self.tree_graph = TreeGraph(
            adjacency=adjacency,
            node_stats=node_stats,
            last_built_from_fact_count=len(getattr(self.fact_manager, "fact_links", {})),
        )
        return self.tree_graph

    async def _run_pending_dialogue_job(
        self,
        *,
        receipt: PendingWriteReceipt,
        req: DialogueInsertRequest,
        key: Tuple[float, int],
        content_id: str,
        leaf: ContentLeaf,
    ) -> None:
        receipt.state = "running"
        errors: List[str] = []
        try:
            history = await self._session_history_before(session_id=req.session_id, end_key=(key[0], key[1] - 1))
            raw = RawContent(
                speaker_id=req.dialogue.speaker_id,
                speaker_tag=req.dialogue.speaker_tag,
                speaker_name=req.dialogue.speaker_name,
                content=req.dialogue.content,
                timestamp=unix_to_time_text(float(req.timestamp), "UTC"),
                listener_name=req.dialogue.listener_name,
                session_id=req.session_id,
                content_id=content_id,
                history=history,
            )
            facts, labels, extract_error = await self._extract_for_raw(
                raw=raw,
                semaphore=asyncio.Semaphore(1),
            )
            if extract_error:
                receipt.state = "failed"
                receipt.error = extract_error
                return

            fact_context_rows = [
                (
                    fact,
                    req.session_id,
                    content_id,
                    f"{req.session_id}|{content_id}|{idx}|{fact.fact_text}",
                )
                for idx, fact in enumerate(facts)
            ]
            accepted_rows = await self._dedupe_fact_context_rows(fact_context_rows)
            facts = [row[0] for row in accepted_rows]
            if facts:
                self._harmonize_entities_in_facts(facts)
            leaf.fact_ids = {fact.fact_id for fact in facts}

            if facts:
                async with self._fact_manager_lock:
                    source_keys = [row[3] for row in accepted_rows]
                    self.fact_manager.add_facts(facts, source_keys=source_keys, persist=False)

            insert_requests: List[Tuple[str, str, AsyncBPlusTree, List[Tuple[Tuple[float, int], Any]]]] = []
            per_fact_actor_tree_ids: Dict[str, Set[str]] = defaultdict(set)
            per_fact_scene_tree_ids: Dict[str, Set[str]] = defaultdict(set)
            if facts:
                actor_groups = self._group_facts_by_actor_norm(facts)
                for actor_key, actor_facts in actor_groups.items():
                    actor_tree = await self.actors.get_tree(actor_key)
                    actor_items = await self._build_entity_items("actor", actor_key, actor_facts)
                    if actor_items:
                        insert_requests.append(("actor", _actor_tree_id(actor_key), actor_tree, actor_items))
                    for fact in actor_facts:
                        per_fact_actor_tree_ids[fact.fact_id].add(_actor_tree_id(actor_key))

                scene_groups = self._group_facts_by_scene_norm(facts)
                for scene_key, scene_facts in scene_groups.items():
                    scene_tree = await self.scenes.get_tree(scene_key)
                    scene_items = await self._build_entity_items("scene", scene_key, scene_facts)
                    if scene_items:
                        insert_requests.append(("scene", _scene_tree_id(scene_key), scene_tree, scene_items))
                    for fact in scene_facts:
                        per_fact_scene_tree_ids[fact.fact_id].add(_scene_tree_id(scene_key))

            inserted_trees, insert_errors = await self._run_tree_inserts(insert_requests, defer_meta_update=True)
            errors.extend(insert_errors)
            await self._mark_dirty_trees(inserted_trees)

            if facts:
                async with self._fact_manager_lock:
                    for fact in facts:
                        self.fact_manager.upsert_fact_links(
                            fact.fact_id,
                            session_id=req.session_id,
                            session_tree_id=_session_tree_id(req.session_id),
                            actor_tree_ids=sorted(per_fact_actor_tree_ids.get(fact.fact_id, set())),
                            scene_tree_ids=sorted(per_fact_scene_tree_ids.get(fact.fact_id, set())),
                            content_id=content_id,
                            persist=False,
                        )

            if errors:
                receipt.state = "failed"
                receipt.error = "; ".join(errors)
            else:
                receipt.state = "done"
                receipt.error = None
        except Exception as exc:  # pragma: no cover - safety net
            receipt.state = "failed"
            receipt.error = str(exc)
        finally:
            async with self._pending_lock:
                self._pending_jobs.pop(receipt.write_id, None)

    async def _session_history_before(
        self,
        *,
        session_id: str,
        end_key: Tuple[float, int],
        pending: Optional[List[Tuple[Tuple[float, int], Dict[str, str]]]] = None,
    ) -> List[Dict[str, str]]:
        tree = await self.sessions.get_tree(session_id)
        existing = await tree.range_query(
            start=(-float("inf"), -10**18),
            end=end_key,
            limit=None,
        )
        history: List[Dict[str, str]] = []
        for row in existing:
            speaker = getattr(row, "speaker", "")
            text = getattr(row, "text", "")
            if isinstance(speaker, str) and isinstance(text, str):
                if speaker.strip() and text.strip():
                    clipped = text
                    if self.history_text_max_chars > 0 and len(clipped) > self.history_text_max_chars:
                        clipped = clipped[: self.history_text_max_chars]
                    history.append({"speaker": speaker, "text": clipped})
        if pending:
            pending_rows = [entry for key, entry in pending if key <= end_key]
            if self.history_text_max_chars > 0:
                clipped_pending: List[Dict[str, str]] = []
                for row in pending_rows:
                    speaker = str(row.get("speaker", ""))
                    text = str(row.get("text", ""))
                    if len(text) > self.history_text_max_chars:
                        text = text[: self.history_text_max_chars]
                    clipped_pending.append({"speaker": speaker, "text": text})
                pending_rows = clipped_pending
            history.extend(pending_rows)
        return history[-self.history_slide_window :]

    async def _dedupe_fact_context_rows(
        self,
        rows: List[Tuple[AtomicFact, str, str, str]],
    ) -> List[Tuple[AtomicFact, str, str, str]]:
        if not rows or not self.atomic_fact_dedup_enabled:
            return list(rows)

        ordered = sorted(
            list(enumerate(rows)),
            key=lambda item: (float(item[1][0].timestamp), int(item[0])),
        )
        accepted_rows: List[Tuple[AtomicFact, str, str, str]] = []
        local_exact_index: Dict[str, str] = {}

        async with self._fact_manager_lock:
            for _, row in ordered:
                fact, session_id, content_id, source_key = row
                canonical_fact_id = ""
                reason = ""

                if self.atomic_fact_exact_dedup_enabled:
                    exact_key = self.fact_manager.build_exact_text_time_key(fact.fact_text, fact.timestamp)
                    canonical_fact_id = local_exact_index.get(exact_key) or (
                        self.fact_manager.lookup_exact_text_time_duplicate(
                            fact_text=fact.fact_text,
                            timestamp=fact.timestamp,
                        )
                        or ""
                    )
                    if canonical_fact_id:
                        reason = "exact_text_time"
                        self.fact_manager.record_duplicate(
                            duplicate_fact_id=fact.fact_id,
                            canonical_fact_id=canonical_fact_id,
                            reason=reason,
                            source_session_id=session_id,
                            source_content_id=content_id,
                            fact_text=fact.fact_text,
                            time_text=fact.time_text,
                            timestamp=fact.timestamp,
                        )
                        self._emit_fact_dedup_event(
                            stage="fact_dedup_exact_hit",
                            fact=fact,
                            session_id=session_id,
                            content_id=content_id,
                            canonical_fact_id=canonical_fact_id,
                            reason=reason,
                        )
                        continue

                if self.atomic_fact_llm_dedup_enabled:
                    candidates = self._collect_text_time_candidates(
                        fact=fact,
                        accepted_rows=accepted_rows,
                    )
                    for candidate, score in candidates[: self.atomic_fact_llm_max_checks_per_fact]:
                        self._emit_fact_dedup_event(
                            stage="fact_dedup_llm_candidate",
                            fact=fact,
                            session_id=session_id,
                            content_id=content_id,
                            canonical_fact_id=candidate.fact_id,
                            reason=f"candidate_score={score:.4f}",
                        )
                        if not self._llm_judge_same_fact(fact, candidate):
                            continue
                        canonical_fact_id = candidate.fact_id
                        reason = "llm_text_time_same_fact"
                        self.fact_manager.record_duplicate(
                            duplicate_fact_id=fact.fact_id,
                            canonical_fact_id=canonical_fact_id,
                            reason=reason,
                            source_session_id=session_id,
                            source_content_id=content_id,
                            fact_text=fact.fact_text,
                            time_text=fact.time_text,
                            timestamp=fact.timestamp,
                        )
                        self._emit_fact_dedup_event(
                            stage="fact_dedup_llm_hit",
                            fact=fact,
                            session_id=session_id,
                            content_id=content_id,
                            canonical_fact_id=canonical_fact_id,
                            reason=reason,
                        )
                        break
                    if canonical_fact_id:
                        continue

                accepted_rows.append(row)
                local_exact_index[self.fact_manager.build_exact_text_time_key(fact.fact_text, fact.timestamp)] = (
                    fact.fact_id
                )
                self._emit_fact_dedup_event(
                    stage="fact_registered",
                    fact=fact,
                    session_id=session_id,
                    content_id=content_id,
                    canonical_fact_id=fact.fact_id,
                    reason="accepted",
                )

        return accepted_rows

    def _collect_text_time_candidates(
        self,
        *,
        fact: AtomicFact,
        accepted_rows: List[Tuple[AtomicFact, str, str, str]],
    ) -> List[Tuple[AtomicFact, float]]:
        candidates: List[Tuple[AtomicFact, float]] = []
        seen: Set[str] = set()
        max_gap = float(self.atomic_fact_llm_max_time_gap_seconds)
        threshold = float(self.atomic_fact_llm_similarity_threshold)

        for accepted, _, _, _ in accepted_rows:
            if abs(float(accepted.timestamp) - float(fact.timestamp)) > max_gap:
                continue
            score = self.fact_manager.fact_text_similarity(fact.fact_text, accepted.fact_text)
            if score < threshold:
                continue
            seen.add(accepted.fact_id)
            candidates.append((accepted, float(score)))

        global_hits = self.fact_manager.search_text_time_candidates(
            fact_text=fact.fact_text,
            timestamp=fact.timestamp,
            top_k=self.atomic_fact_llm_top_k_candidates,
            similarity_threshold=threshold,
            max_time_gap_seconds=max_gap,
        )
        for candidate, score in global_hits:
            if candidate.fact_id in seen:
                continue
            seen.add(candidate.fact_id)
            candidates.append((candidate, float(score)))

        candidates.sort(
            key=lambda row: (-float(row[1]), float(row[0].timestamp), str(row[0].fact_id)),
        )
        return candidates

    def _llm_judge_same_fact(self, left: AtomicFact, right: AtomicFact) -> bool:
        payload = self.prompt_manager.generate(
            self.atomic_fact_llm_prompt_key,
            {
                "fact_a_text": str(left.fact_text),
                "fact_a_time_text": str(left.time_text),
                "fact_b_text": str(right.fact_text),
                "fact_b_time_text": str(right.time_text),
            },
        )
        raw = self.prompt_manager.call_llm(self.atomic_fact_llm_prompt_key, payload)
        try:
            parsed = json.loads(str(raw))
        except Exception:
            return False
        if not isinstance(parsed, dict):
            return False
        return bool(parsed.get("same_fact", False))

    def _emit_fact_dedup_event(
        self,
        *,
        stage: str,
        fact: AtomicFact,
        session_id: str,
        content_id: str,
        canonical_fact_id: str,
        reason: str,
    ) -> None:
        self._emit_timing(
            stage=stage,
            tree_type=None,
            tree_id=None,
            duration_sec=0.0,
            extra={
                "fact_id": str(fact.fact_id),
                "fact_text": str(fact.fact_text),
                "time_text": str(fact.time_text),
                "timestamp": float(fact.timestamp),
                "session_id": str(session_id),
                "content_id": str(content_id),
                "canonical_fact_id": str(canonical_fact_id),
                "reason": str(reason),
            },
        )

    async def _extract_for_raw(
        self,
        *,
        raw: RawContent,
        semaphore: asyncio.Semaphore,
    ) -> Tuple[List[AtomicFact], Dict[str, List[str]], Optional[str]]:
        t_wait = perf_counter()
        await semaphore.acquire()
        wait_sec = perf_counter() - t_wait
        self._emit_timing(
            stage="extract_queue_wait",
            tree_type=None,
            tree_id=None,
            duration_sec=wait_sec,
            extra={"session_id": raw.session_id, "content_id": raw.content_id},
        )
        try:
            t_exec = perf_counter()
            raw_facts = await run_blocking(
                extract_atomic_facts_raw,
                raw,
                self.prompt_manager,
                executor=self._extraction_pool,
            )
            facts = self.fact_canonicalizer.canonicalize(raw_facts)
            if facts:
                key_to_positions: Dict[str, List[int]] = defaultdict(list)
                unique_indices: List[int] = []
                for idx, fact in enumerate(facts):
                    key = _entity_resolution_cache_key(fact)
                    if key not in key_to_positions:
                        unique_indices.append(idx)
                    key_to_positions[key].append(idx)

                cached_results: Dict[str, Tuple[List[str], str]] = {}
                uncached_indices: List[int] = []
                async with self._entity_resolution_cache_lock:
                    for src_idx in unique_indices:
                        key = _entity_resolution_cache_key(facts[src_idx])
                        cached = self._entity_resolution_cache.get(key)
                        if cached is None:
                            uncached_indices.append(src_idx)
                            continue
                        cached_results[key] = (list(cached[0]), str(cached[1]))

                resolved_unique_uncached: List[Any] = []
                if uncached_indices:
                    resolved_unique_uncached = await asyncio.gather(
                        *(self.entity_resolver.resolve_fact(facts[idx]) for idx in uncached_indices),
                        return_exceptions=True,
                    )
                uncached_result_by_index: Dict[int, Any] = {
                    idx: out for idx, out in zip(uncached_indices, resolved_unique_uncached)
                }

                merged = list(facts)
                for src_idx in unique_indices:
                    src = facts[src_idx]
                    key = _entity_resolution_cache_key(src)
                    positions = key_to_positions.get(key, [])
                    if key in cached_results:
                        actors_norm, scene_norm = cached_results[key]
                        for pos in positions:
                            clone = facts[pos]
                            clone.actors_norm = list(actors_norm)
                            clone.scene_norm = str(scene_norm)
                            merged[pos] = clone
                        continue

                    out = uncached_result_by_index.get(src_idx)
                    if out is None:
                        for pos in positions:
                            merged[pos] = facts[pos]
                        continue
                    if isinstance(out, Exception):
                        for pos in positions:
                            merged[pos] = facts[pos]
                        continue

                    actors_norm = list(getattr(out, "actors_norm", []) or [])
                    scene_norm = str(getattr(out, "scene_norm", "") or "")
                    async with self._entity_resolution_cache_lock:
                        self._entity_resolution_cache[key] = (actors_norm, scene_norm)
                        self._entity_resolution_cache.move_to_end(key, last=True)
                        while len(self._entity_resolution_cache) > self._entity_resolution_cache_max_size:
                            self._entity_resolution_cache.popitem(last=False)

                    for pos in positions:
                        clone = facts[pos]
                        clone.actors_norm = list(actors_norm)
                        clone.scene_norm = scene_norm
                        merged[pos] = clone

                facts = merged
            labels = extract_vector_labels(facts)
            self._emit_timing(
                stage="extract_exec",
                tree_type=None,
                tree_id=None,
                duration_sec=perf_counter() - t_exec,
                extra={"session_id": raw.session_id, "content_id": raw.content_id},
            )
            return facts, labels, None
        except Exception as exc:  # extraction should not block session write
            return [], {}, str(exc)
        finally:
            semaphore.release()

    async def _run_tree_inserts(
        self,
        insert_requests: Sequence[Tuple[str, str, AsyncBPlusTree, List[Tuple[Tuple[float, int], Any]]]],
        *,
        defer_meta_update: bool,
    ) -> Tuple[Dict[str, Tuple[str, AsyncBPlusTree]], List[str]]:
        if not insert_requests:
            return {}, []
        sem = asyncio.Semaphore(max(1, self.max_parallel_tree_inserts))
        tasks = [
            asyncio.create_task(
                self._insert_tree_items(
                    tree_type=tree_type,
                    tree_id=tree_id,
                    tree=tree,
                    items=items,
                    semaphore=sem,
                    defer_meta_update=defer_meta_update,
                )
            )
            for tree_type, tree_id, tree, items in insert_requests
        ]
        out: Dict[str, Tuple[str, AsyncBPlusTree]] = {}
        errors: List[str] = []
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for item in results:
            if isinstance(item, Exception):
                errors.append(f"tree insert failed: {item}")
                continue
            tree_type, tree_id, tree_obj, err = item
            if err:
                errors.append(err)
                continue
            if tree_obj is not None:
                out[tree_id] = (tree_type, tree_obj)
        return out, errors

    async def _mark_dirty_tree(self, tree_type: str, tree_id: str, tree: AsyncBPlusTree) -> None:
        async with self._dirty_lock:
            self._dirty_trees[str(tree_id)] = (str(tree_type), tree)

    async def _mark_dirty_trees(self, trees: Mapping[str, Tuple[str, AsyncBPlusTree]]) -> None:
        if not trees:
            return
        async with self._dirty_lock:
            for tree_id, (tree_type, tree) in trees.items():
                self._dirty_trees[str(tree_id)] = (str(tree_type), tree)

    async def _take_dirty_tree_snapshot(self) -> Dict[str, Tuple[str, AsyncBPlusTree]]:
        async with self._dirty_lock:
            snapshot = dict(self._dirty_trees)
            self._dirty_trees.clear()
            return snapshot

    async def _flush_tree(
        self,
        tree_type: str,
        tree_id: str,
        tree: AsyncBPlusTree,
        semaphore: asyncio.Semaphore,
    ) -> Tuple[str, str, Optional[AsyncBPlusTree], Optional[str]]:
        t_wait = perf_counter()
        await semaphore.acquire()
        wait_sec = perf_counter() - t_wait
        self._emit_timing(
            stage="tree_flush_queue_wait",
            tree_type=tree_type,
            tree_id=tree_id,
            duration_sec=wait_sec,
            extra={},
        )
        try:
            t_exec = perf_counter()
            if tree_type in {"session", "actor", "scene"}:
                await tree.ensure_internal_root(defer_meta_update=True)
            await tree.flush_meta_updates()
            exec_sec = perf_counter() - t_exec
            self._emit_timing(
                stage="tree_flush_exec",
                tree_type=tree_type,
                tree_id=tree_id,
                duration_sec=exec_sec,
                extra={},
            )
            return tree_type, tree_id, tree, None
        except Exception as exc:
            return tree_type, tree_id, None, f"{tree_type} tree `{tree_id}` flush failed: {exc}"
        finally:
            semaphore.release()

    async def _insert_tree_items(
        self,
        *,
        tree_type: str,
        tree_id: str,
        tree: AsyncBPlusTree,
        items: List[Tuple[Tuple[float, int], Any]],
        semaphore: asyncio.Semaphore,
        defer_meta_update: bool,
    ) -> Tuple[str, str, Optional[AsyncBPlusTree], Optional[str]]:
        if not items:
            return tree_type, tree_id, None, None
        t_wait_sem = perf_counter()
        await semaphore.acquire()
        sem_wait_sec = perf_counter() - t_wait_sem
        try:
            t_insert_total = perf_counter()
            tree_timing: Dict[str, float] = {}
            await tree.batch_insert(
                items,
                optimize_meta=True,
                defer_meta_update=defer_meta_update,
                timing=tree_timing,
            )
            if tree_type in {"session", "actor", "scene"}:
                await tree.ensure_internal_root(defer_meta_update=defer_meta_update)
            total_sec = perf_counter() - t_insert_total
            lock_wait_sec = float(tree_timing.get("lock_wait_sec", 0.0))
            queue_wait_sec = sem_wait_sec + lock_wait_sec
            exec_sec = float(tree_timing.get("exec_sec", max(0.0, total_sec - lock_wait_sec)))
            self._emit_timing(
                stage="tree_batch_insert_queue_wait",
                tree_type=tree_type,
                tree_id=tree_id,
                duration_sec=queue_wait_sec,
                extra={
                    "item_count": len(items),
                    "sem_wait_sec": sem_wait_sec,
                    "lock_wait_sec": lock_wait_sec,
                },
            )
            self._emit_timing(
                stage="tree_batch_insert_exec",
                tree_type=tree_type,
                tree_id=tree_id,
                duration_sec=exec_sec,
                extra={"item_count": len(items)},
            )
            self._emit_timing(
                stage="tree_batch_insert",
                tree_type=tree_type,
                tree_id=tree_id,
                duration_sec=sem_wait_sec + total_sec,
                extra={
                    "item_count": len(items),
                    "queue_wait_sec": queue_wait_sec,
                    "exec_sec": exec_sec,
                    "sem_wait_sec": sem_wait_sec,
                    "lock_wait_sec": lock_wait_sec,
                    "defer_meta_update": bool(defer_meta_update),
                },
            )
            return tree_type, tree_id, tree, None
        except Exception as exc:
            return tree_type, tree_id, None, f"{tree_type} tree `{tree_id}` insert failed: {exc}"
        finally:
            semaphore.release()

    async def _build_entity_items(
        self,
        entity_type: str,
        entity_key: str,
        facts: Sequence[AtomicFact],
    ) -> List[Tuple[Tuple[float, int], Dict[str, Any]]]:
        if not facts:
            return []
        ordered = sorted(facts, key=lambda x: (float(x.timestamp), str(x.fact_id)))
        seqs = await self._reserve_entity_seqs(entity_type, entity_key, len(ordered))
        out: List[Tuple[Tuple[float, int], Dict[str, Any]]] = []
        for fact, seq in zip(ordered, seqs):
            out.append(
                (
                    (float(fact.timestamp), int(seq)),
                    {
                        "fact_id": fact.fact_id,
                        "fact_ids": [fact.fact_id],
                        "fact_text": fact.fact_text,
                        "timestamp": float(fact.timestamp),
                        "actors": list(fact.actors_norm),
                        "scene": fact.scene_norm,
                        "status": fact.status,
                        "session_id": "",
                        "content_id": "",
                    },
                )
            )
        return out

    async def _update_root_vectors(self, affected_trees: Mapping[str, Tuple[str, AsyncBPlusTree]]) -> None:
        if not affected_trees:
            return
        sem = asyncio.Semaphore(max(1, self.max_parallel_tree_updates))

        async def _upsert_one(tree_type: str, tree_id: str, tree: AsyncBPlusTree) -> None:
            t_wait = perf_counter()
            await sem.acquire()
            try:
                self._emit_timing(
                    stage="tree_root_vector_upsert_queue_wait",
                    tree_type=tree_type,
                    tree_id=tree_id,
                    duration_sec=perf_counter() - t_wait,
                    extra={},
                )
                root = await tree.get_root()
                if root is None:
                    return
                root_meta = await tree.get_node_summary(root)
                leaf_count = await tree.get_leaf_count()
                tags = self._root_tags(root_meta)
                metadata = {
                    "tree_id": tree_id,
                    "time_range": list(root_meta.time_range) if root_meta.time_range else None,
                    "roles": sorted(list(root_meta.roles)),
                    "fact_count": len(root_meta.fact_ids),
                    "leaf_count": int(leaf_count),
                }
                t_exec = perf_counter()
                self.root_index_manager.upsert_root(
                    tree_type,
                    tree_id,
                    root_meta.summary,
                    tags,
                    metadata,
                    persist=False,
                )
                self._emit_timing(
                    stage="tree_root_vector_upsert",
                    tree_type=tree_type,
                    tree_id=tree_id,
                    duration_sec=perf_counter() - t_exec,
                    extra={"fact_count": len(root_meta.fact_ids)},
                )
            finally:
                sem.release()

        tasks = [
            asyncio.create_task(_upsert_one(tree_type, tree_id, tree))
            for tree_id, (tree_type, tree) in affected_trees.items()
        ]
        await asyncio.gather(*tasks)

    def _root_tags(self, meta: BPlusNodeMeta) -> List[str]:
        tags: List[str] = []
        seen: Set[str] = set()
        for token in _TOKEN_RE.findall((meta.summary or "").lower()):
            if len(token) <= 2:
                continue
            tag = f"kw:{token}"
            if tag not in seen:
                seen.add(tag)
                tags.append(tag)
            if len(tags) >= 32:
                break
        for role in sorted(meta.roles):
            tag = f"entity:{_normalize_entity_key(role)}"
            if tag and tag not in seen:
                seen.add(tag)
                tags.append(tag)
        if meta.time_range is not None:
            start, end = meta.time_range
            for ts in (start, end):
                bucket = datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m")
                tag = f"time_bucket:{bucket}"
                if tag not in seen:
                    seen.add(tag)
                    tags.append(tag)
        return tags

    def _harmonize_actor_keys_in_facts(self, facts: Iterable[AtomicFact]) -> List[AtomicFact]:
        fact_list = list(facts)
        if not fact_list:
            return fact_list

        variant_groups: Dict[str, Set[str]] = defaultdict(set)
        variant_counts: Dict[str, int] = defaultdict(int)
        for fact in fact_list:
            for actor in list(getattr(fact, "actors_norm", []) or []):
                key = _normalize_entity_key(actor)
                if not key:
                    continue
                base = _singularize_entity_key(key)
                variant_groups[base].add(key)
                variant_counts[key] += 1

        alias_map: Dict[str, str] = {}
        for base, variants in variant_groups.items():
            if len(variants) <= 1:
                continue
            preferred = base if base in variants else min(
                variants,
                key=lambda key: (-int(variant_counts.get(key, 0)), len(key), key),
            )
            for variant in variants:
                alias_map[variant] = preferred

        alias_map = self._extend_actor_alias_map(fact_list, alias_map)
        return self._apply_actor_alias_map(fact_list, alias_map)

    def _extend_actor_alias_map(self, facts: Sequence[AtomicFact], alias_map: Dict[str, str]) -> Dict[str, str]:
        actor_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"timestamps": set(), "coactors": set(), "count": 0}
        )
        for fact in facts:
            actors = [_normalize_entity_key(x) for x in list(getattr(fact, "actors_norm", []) or [])]
            actors = [alias_map.get(x, x) for x in actors if x]
            actors = sorted(set(actors))
            ts = int(float(getattr(fact, "timestamp", 0.0) or 0.0))
            for actor in actors:
                actor_stats[actor]["timestamps"].add(ts)
                actor_stats[actor]["count"] += 1
                actor_stats[actor]["coactors"].update(x for x in actors if x != actor)

        keys = sorted(actor_stats.keys(), key=lambda key: (len(_entity_key_tokens(key)), len(key), key))
        pending = dict(alias_map)
        for short in keys:
            short_tokens = _entity_key_tokens(short)
            if len(short_tokens) > 2 or not short_tokens:
                continue
            candidates: List[Tuple[int, int, int, str]] = []
            for long in keys:
                if long == short:
                    continue
                long_tokens = _entity_key_tokens(long)
                if len(long_tokens) <= len(short_tokens):
                    continue
                if short_tokens[0] != long_tokens[0] and short_tokens[-1] != long_tokens[-1]:
                    continue
                if not set(short_tokens).issubset(set(long_tokens)):
                    continue
                ts_overlap = len(actor_stats[short]["timestamps"] & actor_stats[long]["timestamps"])
                if ts_overlap <= 0:
                    continue
                co_overlap = len(actor_stats[short]["coactors"] & actor_stats[long]["coactors"])
                candidates.append((ts_overlap, co_overlap, actor_stats[long]["count"], long))
            if not candidates:
                continue
            candidates.sort(reverse=True)
            best = candidates[0]
            if len(candidates) > 1 and candidates[1][:3] == best[:3]:
                continue
            pending[short] = best[3]
        return pending

    def _apply_actor_alias_map(self, facts: Sequence[AtomicFact], alias_map: Mapping[str, str]) -> List[AtomicFact]:
        fact_list = list(facts)
        if not alias_map:
            return fact_list
        for fact in fact_list:
            deduped: List[str] = []
            seen: Set[str] = set()
            for actor in list(getattr(fact, "actors_norm", []) or []):
                key = _normalize_entity_key(actor)
                if not key:
                    continue
                key = str(alias_map.get(key, key))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(key)
            fact.actors_norm = deduped
        return fact_list

    def _harmonize_scene_keys_in_facts(self, facts: Iterable[AtomicFact]) -> List[AtomicFact]:
        fact_list = list(facts)
        if not fact_list:
            return fact_list

        scene_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"timestamps": set(), "actor_signatures": set(), "count": 0}
        )
        for fact in fact_list:
            key = _normalize_entity_key(getattr(fact, "scene_norm", ""))
            if not key:
                continue
            scene_stats[key]["count"] += 1
            scene_stats[key]["timestamps"].add(int(float(getattr(fact, "timestamp", 0.0) or 0.0)))
            actors = sorted({_normalize_entity_key(x) for x in list(getattr(fact, "actors_norm", []) or []) if _normalize_entity_key(x)})
            scene_stats[key]["actor_signatures"].add("|".join(actors))

        alias_map: Dict[str, str] = {}
        keys = sorted(scene_stats.keys())
        for idx, left in enumerate(keys):
            left_tokens = set(_scene_key_tokens(left))
            if not left_tokens:
                continue
            for right in keys[idx + 1 :]:
                right_tokens = set(_scene_key_tokens(right))
                if not right_tokens:
                    continue
                if not (scene_stats[left]["timestamps"] & scene_stats[right]["timestamps"]):
                    continue
                if not (scene_stats[left]["actor_signatures"] & scene_stats[right]["actor_signatures"]):
                    continue
                overlap = left_tokens & right_tokens
                union = left_tokens | right_tokens
                similar = False
                if left in right or right in left:
                    similar = True
                elif overlap and (float(len(overlap)) / float(len(union) or 1)) >= 0.5:
                    similar = True
                elif _scene_key_tokens(left)[-1:] == _scene_key_tokens(right)[-1:] and overlap:
                    similar = True
                if not similar:
                    continue
                preferred = min(
                    (left, right),
                    key=lambda key: (-int(scene_stats[key]["count"]), len(key), key),
                )
                alias_map[left] = preferred
                alias_map[right] = preferred

        if not alias_map:
            return fact_list

        for fact in fact_list:
            key = _normalize_entity_key(getattr(fact, "scene_norm", ""))
            if not key:
                continue
            fact.scene_norm = alias_map.get(key, key)
        return fact_list

    def _harmonize_entities_in_facts(self, facts: Iterable[AtomicFact]) -> List[AtomicFact]:
        fact_list = list(facts)
        if not fact_list:
            return fact_list
        self._harmonize_actor_keys_in_facts(fact_list)
        self._harmonize_scene_keys_in_facts(fact_list)
        return fact_list

    def _group_facts_by_actor_norm(self, facts: Iterable[AtomicFact]) -> Dict[str, List[AtomicFact]]:
        grouped: Dict[str, List[AtomicFact]] = defaultdict(list)
        seen: Dict[str, Set[str]] = defaultdict(set)
        for fact in self._harmonize_entities_in_facts(facts):
            keys = {_normalize_entity_key(x) for x in fact.actors_norm if _normalize_entity_key(x)}
            for key in keys:
                if fact.fact_id in seen[key]:
                    continue
                seen[key].add(fact.fact_id)
                grouped[key].append(fact)
        return dict(grouped)

    def _group_facts_by_scene_norm(self, facts: Iterable[AtomicFact]) -> Dict[str, List[AtomicFact]]:
        grouped: Dict[str, List[AtomicFact]] = defaultdict(list)
        seen: Dict[str, Set[str]] = defaultdict(set)
        for fact in self._harmonize_entities_in_facts(facts):
            key = _normalize_entity_key(fact.scene_norm)
            if not key:
                continue
            if fact.fact_id in seen[key]:
                continue
            seen[key].add(fact.fact_id)
            grouped[key].append(fact)
        return dict(grouped)

    async def _next_session_seq(self, session_id: str) -> int:
        key = str(session_id)
        async with self._session_seq_map_lock:
            lock = self._session_seq_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._session_seq_locks[key] = lock
        async with lock:
            current = int(self._session_seq[key])
            self._session_seq[key] = current + 1
            return current

    async def _allocate_session_slot(self, session_id: str, requested_ts: float) -> Tuple[float, int]:
        key = str(session_id)
        async with self._session_seq_map_lock:
            lock = self._session_seq_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._session_seq_locks[key] = lock

        async with lock:
            seq = int(self._session_seq[key])
            self._session_seq[key] = seq + 1

            ts = float(requested_ts)
            last_ts = float(self._session_last_ts.get(key, float("-inf")))
            eps = max(0.0, float(self.timestamp_epsilon_sec))
            if eps > 0.0 and ts <= last_ts:
                ts = last_ts + eps
            self._session_last_ts[key] = ts
            return ts, seq

    async def _reserve_entity_seqs(self, entity_type: str, entity_key: str, n: int) -> List[int]:
        if n <= 0:
            return []
        e_type = str(entity_type).strip().lower()
        if e_type not in {"actor", "scene"}:
            raise ValueError(f"Unsupported entity_type: {entity_type}")
        key = _normalize_entity_key(entity_key)
        async with self._entity_seq_map_lock:
            lock = self._entity_seq_locks[e_type].get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._entity_seq_locks[e_type][key] = lock
        async with lock:
            start = int(self._entity_seq[e_type][key])
            self._entity_seq[e_type][key] = start + int(n)
            return list(range(start, start + int(n)))

    def _coerce_request(self, item: Any) -> DialogueInsertRequest:
        if isinstance(item, DialogueInsertRequest):
            session_id = str(item.session_id).strip()
            if not session_id:
                raise ValueError("`session_id` is required.")
            if not isinstance(item.timestamp, (int, float)):
                raise ValueError("`timestamp` must be numeric unix timestamp.")
            return DialogueInsertRequest(
                session_id=session_id,
                dialogue=self._coerce_dialogue(item.dialogue),
                timestamp=float(item.timestamp),
            )
        if not isinstance(item, Mapping):
            raise ValueError("DialogueInsertRequest must be dataclass or mapping.")

        session_id = str(item.get("session_id", "")).strip()
        if not session_id:
            raise ValueError("`session_id` is required.")

        dialogue_raw = item.get("dialogue")
        dialogue = self._coerce_dialogue(dialogue_raw)

        timestamp = item.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            raise ValueError("`timestamp` must be numeric unix timestamp.")
        return DialogueInsertRequest(
            session_id=session_id,
            dialogue=dialogue,
            timestamp=float(timestamp),
        )

    def _coerce_dialogue(self, dialogue: Any) -> DialogueInput:
        if isinstance(dialogue, DialogueInput):
            if not str(dialogue.speaker_name).strip():
                raise ValueError("`dialogue.speaker_name` is required.")
            if not str(dialogue.listener_name).strip():
                raise ValueError("`dialogue.listener_name` is required.")
            return dialogue
        if not isinstance(dialogue, Mapping):
            raise ValueError("`dialogue` must be DialogueInput or mapping.")

        speaker_id = str(dialogue.get("speaker_id", "")).strip()
        speaker_name = str(dialogue.get("speaker_name", "")).strip()
        listener_name = str(dialogue.get("listener_name", "")).strip()
        content = str(dialogue.get("content", "")).strip()
        speaker_tag = str(dialogue.get("speaker_tag", "user")).strip() or "user"
        if not speaker_id:
            raise ValueError("`dialogue.speaker_id` is required.")
        if not speaker_name:
            raise ValueError("`dialogue.speaker_name` is required.")
        if not listener_name:
            raise ValueError("`dialogue.listener_name` is required.")
        if not content:
            raise ValueError("`dialogue.content` is required.")

        metadata: Dict[str, Any] = {}
        for k, v in dict(dialogue).items():
            if k in {"speaker_id", "speaker_name", "listener_name", "speaker_tag", "content"}:
                continue
            metadata[str(k)] = v

        return DialogueInput(
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            listener_name=listener_name,
            speaker_tag=speaker_tag,
            content=content,
            metadata=metadata,
        )

    def _emit_timing(
        self,
        *,
        stage: str,
        tree_type: Optional[str],
        tree_id: Optional[str],
        duration_sec: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        cb = self._timing_callback
        if cb is None:
            return
        event = ForestTimingEvent(
            stage=str(stage),
            tree_type=tree_type if isinstance(tree_type, str) else None,
            tree_id=tree_id if isinstance(tree_id, str) else None,
            duration_sec=float(duration_sec),
            extra=dict(extra or {}),
        )
        try:
            cb(event)
        except Exception:
            return

    def _all_known_tree_ids(self) -> Set[str]:
        tree_ids: Set[str] = set()
        tree_ids.update({_session_tree_id(key) for key in self.sessions.trees.keys()})
        tree_ids.update({_actor_tree_id(key) for key in self.actors.trees.keys()})
        tree_ids.update({_scene_tree_id(key) for key in self.scenes.trees.keys()})
        records = getattr(self.root_index_manager, "_records", {})
        if isinstance(records, Mapping):
            for tree_type in ("session", "actor", "scene"):
                table = records.get(tree_type, {})
                if not isinstance(table, Mapping):
                    continue
                tree_ids.update({str(tree_id).strip() for tree_id in table.keys() if str(tree_id).strip()})
        return tree_ids

    async def save(self, snapshot_dir: str) -> ForestSnapshotReport:
        target = Path(snapshot_dir)
        tmp = target.parent / f"{target.name}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)

        state_dir = tmp / "state"
        trees_dir = tmp / "trees"
        facts_dir = tmp / "facts"
        roots_dir = tmp / "roots"
        state_dir.mkdir(parents=True, exist_ok=True)
        (trees_dir / "session").mkdir(parents=True, exist_ok=True)
        (trees_dir / "actor").mkdir(parents=True, exist_ok=True)
        (trees_dir / "scene").mkdir(parents=True, exist_ok=True)
        facts_dir.mkdir(parents=True, exist_ok=True)
        roots_dir.mkdir(parents=True, exist_ok=True)

        warnings: List[str] = []

        counters_payload = {
            "session_seq": {str(k): int(v) for k, v in self._session_seq.items()},
            "entity_seq": {
                "actor": {str(k): int(v) for k, v in self._entity_seq["actor"].items()},
                "scene": {str(k): int(v) for k, v in self._entity_seq["scene"].items()},
            },
        }
        (state_dir / "counters.json").write_text(
            json.dumps(counters_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        tree_manifest: Dict[str, List[Dict[str, str]]] = {"session": [], "actor": [], "scene": []}
        saved_tree_counts = {"session": 0, "actor": 0, "scene": 0}

        for tree_type, trees in (
            ("session", self.sessions.trees),
            ("actor", self.actors.trees),
            ("scene", self.scenes.trees),
        ):
            for key, tree in sorted(trees.items(), key=lambda x: x[0]):
                filename = self._snapshot_tree_filename(key)
                rel_path = f"{tree_type}/{filename}"
                abs_path = trees_dir / rel_path
                await tree.save(str(abs_path))
                tree_manifest[tree_type].append({"key": str(key), "file": rel_path})
                saved_tree_counts[tree_type] += 1

        facts_files = self.fact_manager.export_snapshot(str(facts_dir))
        roots_files = self.root_index_manager.export_snapshot(str(roots_dir))

        summary_parallel_limit = int(self.config.get("summarization", {}).get("max_parallel_llm", 8))
        tree_order = int(self.sessions.order)
        manifest = {
            "snapshot_version": 1,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "tree_order": tree_order,
            "summary_parallel_limit": summary_parallel_limit,
            "trees": tree_manifest,
            "facts": facts_files,
            "roots": roots_files,
            "strict_default": False,
        }
        (tmp / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if target.exists():
            shutil.rmtree(target)
        tmp.rename(target)
        return ForestSnapshotReport(
            snapshot_dir=str(target),
            saved_tree_counts=saved_tree_counts,
            loaded_tree_counts={"session": 0, "actor": 0, "scene": 0},
            warnings=warnings,
        )

    @classmethod
    async def load(
        cls,
        *,
        snapshot_dir: str,
        prompt_manager: PromptManager,
        config: Optional[Dict[str, Any]] = None,
        tree_order: int = 4,
        strict: bool = False,
    ) -> Tuple["Forest", ForestSnapshotReport]:
        root = Path(snapshot_dir)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Snapshot manifest not found: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            raise ValueError("Snapshot manifest must be a JSON object.")

        cfg = config or {}
        vector_dim = int(cfg.get("model", {}).get("embedding", {}).get("global", {}).get("dimension", 1024))

        fact_manager = FactManager.from_snapshot(str(root / "facts"), vector_dim=vector_dim)
        root_index_manager = RootSummaryIndexManager.from_snapshot(
            str(root / "roots"),
            vector_dim=vector_dim,
        )

        resolved_tree_order = int(manifest.get("tree_order", tree_order))
        forest = cls(
            prompt_manager=prompt_manager,
            fact_manager=fact_manager,
            config=cfg,
            root_index_manager=root_index_manager,
            tree_order=resolved_tree_order,
        )

        counters_path = root / "state" / "counters.json"
        warnings: List[str] = []
        if counters_path.exists():
            try:
                counters = json.loads(counters_path.read_text(encoding="utf-8"))
                session_seq = counters.get("session_seq", {})
                entity_seq = counters.get("entity_seq", {})
                actor_seq = entity_seq.get("actor", {})
                scene_seq = entity_seq.get("scene", {})

                forest._session_seq = defaultdict(
                    int,
                    {str(k): int(v) for k, v in dict(session_seq).items()},
                )
                forest._entity_seq = {
                    "actor": defaultdict(
                        int,
                        {str(k): int(v) for k, v in dict(actor_seq).items()},
                    ),
                    "scene": defaultdict(
                        int,
                        {str(k): int(v) for k, v in dict(scene_seq).items()},
                    ),
                }
            except Exception as exc:
                msg = f"Failed to load counters: {exc}"
                if strict:
                    raise RuntimeError(msg) from exc
                warnings.append(msg)

        loaded_tree_counts = {"session": 0, "actor": 0, "scene": 0}
        trees_manifest = manifest.get("trees", {})
        if not isinstance(trees_manifest, Mapping):
            raise ValueError("Manifest `trees` must be a mapping.")

        for tree_type, manager in (
            ("session", forest.sessions),
            ("actor", forest.actors),
            ("scene", forest.scenes),
        ):
            entries = trees_manifest.get(tree_type, [])
            if not isinstance(entries, list):
                msg = f"Manifest trees[{tree_type}] must be a list."
                if strict:
                    raise ValueError(msg)
                warnings.append(msg)
                continue
            for item in entries:
                try:
                    if not isinstance(item, Mapping):
                        raise ValueError("tree entry must be a mapping.")
                    key = str(item.get("key", ""))
                    rel_file = str(item.get("file", ""))
                    if not key or not rel_file:
                        raise ValueError("tree entry requires non-empty key/file.")
                    tree_path = root / "trees" / rel_file
                    tree = await AsyncBPlusTree.load(
                        str(tree_path),
                        prompt_manager=prompt_manager,
                        summary_executor=forest._summary_pool,
                    )
                    manager.trees[key] = tree
                    loaded_tree_counts[tree_type] += 1
                except Exception as exc:
                    msg = f"Failed loading {tree_type} tree entry {item}: {exc}"
                    if strict:
                        raise RuntimeError(msg) from exc
                    warnings.append(msg)

        report = ForestSnapshotReport(
            snapshot_dir=str(root),
            saved_tree_counts={
                "session": len(trees_manifest.get("session", []))
                if isinstance(trees_manifest.get("session", []), list)
                else 0,
                "actor": len(trees_manifest.get("actor", []))
                if isinstance(trees_manifest.get("actor", []), list)
                else 0,
                "scene": len(trees_manifest.get("scene", []))
                if isinstance(trees_manifest.get("scene", []), list)
                else 0,
            },
            loaded_tree_counts=loaded_tree_counts,
            warnings=warnings,
        )
        forest.refresh_tree_graph()
        return forest, report

    def _snapshot_tree_filename(self, key: str) -> str:
        normalized = str(key)
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in normalized)
        digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()[:10]
        return f"{safe}_{digest}.json"


class SessionForest(SessionForestManager):
    """Backward-compatible session forest facade."""

    async def append_turn(self, session_id: str, content: ContentLeaf) -> None:
        tree = await self.get_tree(session_id)
        await tree.insert((float(content.timestamp), int(content.turn_index)), content, defer_meta_update=False)


class RoleForest(ActorForestManager):
    """Backward-compatible role forest facade (maps role to actor tree)."""

    async def add_fact(self, role_id: str, fact_id: str, timestamp: float) -> None:
        tree = await self.get_tree(role_id)
        value = {
            "fact_id": str(fact_id),
            "fact_ids": [str(fact_id)],
            "fact_text": str(fact_id),
            "timestamp": float(timestamp),
            "actors": [str(role_id)],
            "scene": "",
            "status": "active",
            "session_id": "",
            "content_id": "",
        }
        await tree.insert((float(timestamp), 0), value, defer_meta_update=False)
