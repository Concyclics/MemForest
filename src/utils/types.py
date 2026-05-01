"""Core data structures for chunking, extraction, and tree-friendly routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChunkingConfig:
    """Controls lightweight MemCell chunking before extraction."""

    max_turns: int = 2       # ablation: turn2 optimal (99% support, 9.8x tok efficiency vs whole)
    max_chars: int = 8000    # safety net; turn count is the primary chunking axis
    max_time_gap_seconds: float = 6 * 3600
    hard_boundary_markers: tuple[str, ...] = (
        "--- topic",
        "--- session",
        "new topic:",
        "switching gears",
    )


@dataclass(frozen=True)
class NormalizedTurn:
    """Normalized dialogue turn used by the extraction pipeline."""

    turn_id: str
    session_id: str
    turn_index: int
    speaker_tag: str
    speaker_name: str
    listener_name: str
    text: str
    timestamp_text: str
    timestamp_unix: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemCell:
    """A contiguous local memory unit inside one session."""

    cell_id: str
    session_id: str
    cell_index: int
    start_turn_index: int
    end_turn_index: int
    turn_ids: list[str]
    content_ids: list[str]
    speaker_names: list[str]
    time_start: float
    time_end: float
    text: str
    turns: list[NormalizedTurn] = field(default_factory=list)


# Short compatibility alias while the rest of vNext code is migrating.
SessionChunk = MemCell


@dataclass(frozen=True)
class MemoryItem:
    """Tree-friendly memory item emitted from a MemCell."""

    item_id: str
    session_id: str
    cell_id: str
    fact_text: str
    source_turn_ids: list[str]
    source_spans: list[str]
    participants: list[str]
    origin: str
    semantic_role: str
    entities: list[str]
    topics: list[str]
    time_text: str
    time_start: float | None
    time_end: float | None
    attribute_keys: list[str]
    domain_keys: list[str]
    collection_keys: list[str]
    detail_level: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FactOccurrence:
    """One original extraction item occurrence linked to a managed canonical fact."""

    session_id: str
    cell_id: str
    item_id: str
    source_turn_ids: list[str]
    source_spans: list[str]
    participants: list[str]
    time_text: str
    time_start: float | None
    time_end: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManagedFact:
    """Persisted canonical fact managed by the vNext fact store."""

    fact_id: str
    fact_text: str
    embedding_id: str
    origin: str
    semantic_role: str
    entities: list[str]
    topics: list[str]
    attribute_keys: list[str]
    domain_keys: list[str]
    collection_keys: list[str]
    detail_level: str
    confidence: float
    first_session_id: str
    first_cell_id: str
    occurrences: list[FactOccurrence] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Temporal range derived from first occurrence (optional; used by tree layer)
    time_start: float | None = None
    time_end: float | None = None


@dataclass(frozen=True)
class DuplicateFactRecord:
    """One duplicate input item merged into an existing canonical fact."""

    duplicate_item_id: str
    canonical_fact_id: str
    similarity: float
    reason: str
    source_session_id: str
    source_cell_id: str
    fact_text: str


@dataclass(frozen=True)
class FactWriteResult:
    """Result summary for one batch write into the fact manager."""

    input_count: int
    inserted_count: int
    merged_count: int
    skipped_count: int
    inserted_facts: list[ManagedFact] = field(default_factory=list)
    canonical_fact_ids: list[str] = field(default_factory=list)
    duplicate_records: list[DuplicateFactRecord] = field(default_factory=list)


@dataclass(frozen=True)
class CellExtractionResult:
    """Extraction output for a single MemCell."""

    cell_id: str
    session_id: str
    cell_summary: str
    memory_items: list[MemoryItem] = field(default_factory=list)


@dataclass(frozen=True)
class SessionExtractionResult:
    """End-to-end extraction output for one session."""

    session_id: str
    cells: list[MemCell] = field(default_factory=list)
    memory_items: list[MemoryItem] = field(default_factory=list)


@dataclass(frozen=True)
class ExtractionRequest:
    """One top-level extraction request handled by the extraction manager."""

    session_id: str
    turns: list[dict[str, Any]]
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CellExtractionRequest:
    """Internal request unit for one MemCell."""

    session_id: str
    cell: MemCell
    request_id: str | None = None
