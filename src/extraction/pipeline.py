"""Single-pass MemCell extraction pipeline with tree-friendly memory items."""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

from src.extraction.chunker import chunk_session
from src.prompt import (
    PromptManager,
    UNIFIED_MEMORY_EXTRACTION_PROMPT_NAME,
    build_extraction_prompt_manager,
)
from src.utils.text import normalize_entity_key, split_sentences
from src.utils.time import render_time_text
from src.utils.types import (
    CellExtractionRequest,
    CellExtractionResult,
    ChunkingConfig,
    MemCell,
    MemoryItem,
    SessionExtractionResult,
)


class JsonExtractionBackend(Protocol):
    """Minimal backend contract for single-pass extraction."""

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a parsed JSON object for the prompt pair."""


class ChunkExtractionPipeline:
    """Extract tree-friendly memory items from MemCells."""

    # Turn-based scaling constants for per-cell item budgets.
    # base items / base turns gives the per-turn budget; actual cap =
    # max(base, base * n_turns / base_turns).
    _ITEMS_BASE = 15
    _ITEMS_BASE_TURNS = 2
    _ASST_ITEMS_BASE = 2
    _ASST_ITEMS_BASE_TURNS = 2

    def __init__(
        self,
        *,
        backend: JsonExtractionBackend | None = None,
        chunking: ChunkingConfig = ChunkingConfig(),
        prompt_manager: PromptManager | None = None,
        max_items_per_cell: int = 15,
        max_assistant_items_per_cell: int = 2,
        max_topics_per_item: int = 3,
        max_attribute_keys_per_item: int = 2,
        max_domain_keys_per_item: int = 2,
        max_collection_keys_per_item: int = 2,
    ) -> None:
        self.backend = backend
        self.chunking = chunking
        self.prompt_manager = prompt_manager or build_extraction_prompt_manager()
        self.max_items_per_cell = max(1, int(max_items_per_cell))
        self.max_assistant_items_per_cell = max(0, int(max_assistant_items_per_cell))
        self.max_topics_per_item = max(1, int(max_topics_per_item))
        self.max_attribute_keys_per_item = max(0, int(max_attribute_keys_per_item))
        self.max_domain_keys_per_item = max(0, int(max_domain_keys_per_item))
        self.max_collection_keys_per_item = max(0, int(max_collection_keys_per_item))

    def _scaled_items_cap(self, n_turns: int) -> int:
        """Return max_items_per_cell scaled by actual turn count in cell."""
        return max(self.max_items_per_cell,
                   self._ITEMS_BASE * n_turns // self._ITEMS_BASE_TURNS)

    def _scaled_assistant_cap(self, n_turns: int) -> int:
        """Return max_assistant_items_per_cell scaled by actual turn count."""
        return max(self.max_assistant_items_per_cell,
                   self._ASST_ITEMS_BASE * n_turns // self._ASST_ITEMS_BASE_TURNS)

    def extract_session(self, session_id: str, turns: list[dict[str, Any]]) -> SessionExtractionResult:
        cells = self.build_cells(session_id, turns)
        memory_items: list[MemoryItem] = []
        for cell in cells:
            extracted = self.extract_cell(cell)
            memory_items.extend(extracted.memory_items)
        return SessionExtractionResult(
            session_id=session_id,
            cells=cells,
            memory_items=memory_items,
        )

    def build_cells(self, session_id: str, turns: list[dict[str, Any]]) -> list[MemCell]:
        return chunk_session(session_id, turns, config=self.chunking)

    def extract_cell(self, cell: MemCell) -> CellExtractionResult:
        return self._extract_cell_impl(cell, trace=None)

    def extract_cell_request(self, request: CellExtractionRequest) -> CellExtractionResult:
        trace = {
            "request_id": request.request_id,
            "session_id": request.session_id,
            "cell_id": request.cell.cell_id,
            "cell_index": request.cell.cell_index,
        }
        return self._extract_cell_impl(request.cell, trace=trace)

    def _extract_cell_impl(
        self,
        cell: MemCell,
        *,
        trace: dict[str, Any] | None,
    ) -> CellExtractionResult:
        if self.backend is None:
            return _fallback_extract_cell(cell)

        n_turns = len(cell.turns) if cell.turns else 1
        eff_max_items = self._scaled_items_cap(n_turns)
        eff_max_asst = self._scaled_assistant_cap(n_turns)

        system_prompt, user_prompt = self.prompt_manager.render(
            UNIFIED_MEMORY_EXTRACTION_PROMPT_NAME,
            {
                "session_id": cell.session_id,
                "cell_id": cell.cell_id,
                "cell_index": cell.cell_index,
                "time_start": render_time_text(cell.time_start),
                "time_end": render_time_text(cell.time_end),
                "max_items_per_cell": eff_max_items,
                "max_assistant_items_per_cell": eff_max_asst,
                "max_topics_per_item": self.max_topics_per_item,
                "max_attribute_keys_per_item": self.max_attribute_keys_per_item,
                "max_domain_keys_per_item": self.max_domain_keys_per_item,
                "max_collection_keys_per_item": self.max_collection_keys_per_item,
                "cell_text": cell.text,
            },
        )
        try:
            payload = self.backend.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                trace=trace,
            )
        except TypeError:
            payload = self.backend.generate_json(system_prompt=system_prompt, user_prompt=user_prompt)
        return _parse_cell_response(
            cell,
            payload,
            max_items_per_cell=eff_max_items,
            max_assistant_items_per_cell=eff_max_asst,
            max_topics_per_item=self.max_topics_per_item,
            max_attribute_keys_per_item=self.max_attribute_keys_per_item,
            max_domain_keys_per_item=self.max_domain_keys_per_item,
            max_collection_keys_per_item=self.max_collection_keys_per_item,
        )


def _parse_cell_response(
    cell: MemCell,
    payload: dict[str, Any],
    *,
    max_items_per_cell: int,
    max_assistant_items_per_cell: int,
    max_topics_per_item: int,
    max_attribute_keys_per_item: int,
    max_domain_keys_per_item: int,
    max_collection_keys_per_item: int,
) -> CellExtractionResult:
    summary = str(payload.get("cell_summary", "")).strip()
    raw_items = payload.get("memory_items", [])
    if not isinstance(raw_items, list):
        raw_items = []
    items: list[MemoryItem] = []
    assistant_item_count = 0
    for idx, row in enumerate(raw_items[: max_items_per_cell]):
        if not isinstance(row, dict):
            continue
        fact_text = str(row.get("fact_text", "")).strip()
        if not fact_text:
            continue
        origin = _normalize_origin(row.get("origin"))
        # Enforce a per-cell cap on assistant-origin items. The LLM prompt already
        # restricts to ≤2 items per assistant turn, but across multiple turns in a
        # long cell the total can still be high. User facts (P1/P2) always take
        # priority; assistant items beyond the cap are silently dropped.
        if origin == "assistant":
            if assistant_item_count >= max_assistant_items_per_cell:
                continue
            assistant_item_count += 1
        participants = _normalize_list(row.get("participants", []))
        semantic_role = _normalize_semantic_role(row.get("semantic_role"))
        entities = _normalize_list(row.get("entities", []))
        topics = _normalize_topics(row.get("topics", []), limit=max_topics_per_item)
        attribute_keys = _normalize_keys(row.get("attribute_keys", []), limit=max_attribute_keys_per_item)
        domain_keys = _normalize_keys(row.get("domain_keys", []), limit=max_domain_keys_per_item)
        collection_keys = _normalize_keys(row.get("collection_keys", []), limit=max_collection_keys_per_item)
        if semantic_role in {"reference", "detail"}:
            attribute_keys = []
        if semantic_role == "reference":
            collection_keys = []
        time_text = str(row.get("time_text", "")).strip()
        time_start = _maybe_float(row.get("time_start"))
        time_end = _maybe_float(row.get("time_end"))
        item_id = _stable_id("item", cell.cell_id, idx, fact_text)
        items.append(
            MemoryItem(
                item_id=item_id,
                session_id=cell.session_id,
                cell_id=cell.cell_id,
                fact_text=fact_text,
                source_turn_ids=list(cell.turn_ids),
                source_spans=[],
                participants=participants,
                origin=origin,
                semantic_role=semantic_role,
                entities=entities,
                topics=topics,
                time_text=time_text,
                time_start=time_start,
                time_end=time_end,
                attribute_keys=attribute_keys,
                domain_keys=domain_keys,
                collection_keys=collection_keys,
                detail_level="specific",
                confidence=0.8,
                metadata={},
            )
        )
    return CellExtractionResult(
        cell_id=cell.cell_id,
        session_id=cell.session_id,
        cell_summary=summary,
        memory_items=items,
    )


def _fallback_extract_cell(cell: MemCell) -> CellExtractionResult:
    items: list[MemoryItem] = []
    for idx, turn in enumerate(cell.turns):
        for sentence_idx, sentence in enumerate(split_sentences(turn.text)):
            fact_text = sentence.strip()
            if not fact_text:
                continue
            item_id = _stable_id("item", cell.cell_id, idx * 100 + sentence_idx, fact_text)
            origin = _origin_from_turn(turn)
            semantic_role = "event" if origin == "user" else "reference" if origin == "assistant" else "event"
            items.append(
                MemoryItem(
                    item_id=item_id,
                    session_id=cell.session_id,
                    cell_id=cell.cell_id,
                    fact_text=fact_text,
                    source_turn_ids=[turn.turn_id],
                    source_spans=[fact_text],
                    participants=_normalize_list([turn.speaker_name]),
                    origin=origin,
                    semantic_role=semantic_role,
                    entities=[],
                    topics=[],
                    time_text=turn.timestamp_text,
                    time_start=float(turn.timestamp_unix),
                    time_end=float(turn.timestamp_unix),
                    attribute_keys=[],
                    domain_keys=[],
                    collection_keys=[],
                    detail_level="specific",
                    confidence=0.6,
                    metadata={},
                )
            )
    summary = str(items[0].fact_text) if items else ""
    return CellExtractionResult(
        cell_id=cell.cell_id,
        session_id=cell.session_id,
        cell_summary=summary,
        memory_items=items,
    )


def _normalize_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = normalize_entity_key(text) or text.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return normalized


def _normalize_topics(values: Any, *, limit: int) -> list[str]:
    topics = _normalize_list(values)
    return topics[: max(0, int(limit))]


def _normalize_keys(values: Any, *, limit: int) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value).strip().lower().split())
        if not text:
            continue
        text = text.replace(" ", "_")
        if text in seen:
            continue
        seen.add(text)
        normalized.append(text)
        if len(normalized) >= max(0, int(limit)):
            break
    return normalized


def _normalize_origin(value: Any) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    if text in {"user", "assistant", "mixed", "unknown"}:
        return text
    return "unknown"


def _normalize_semantic_role(value: Any) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    if text in {"event", "state", "preference", "constraint", "plan", "detail", "reference"}:
        return text
    return "event"


def _origin_from_turn(turn: Any) -> str:
    text = str(getattr(turn, "speaker_tag", "") or "").strip().lower()
    if text == "assistant":
        return "assistant"
    if text == "user":
        return "user"
    speaker_name = str(getattr(turn, "speaker_name", "") or "").strip().lower()
    if speaker_name == "assistant":
        return "assistant"
    if speaker_name == "user":
        return "user"
    return "unknown"


def _maybe_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _stable_id(prefix: str, seed: str, index: int, value: Any) -> str:
    raw = f"{prefix}|{seed}|{index}|{value}"
    return f"{prefix}_{hashlib.md5(raw.encode('utf-8')).hexdigest()[:16]}"
