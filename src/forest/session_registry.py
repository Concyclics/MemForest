"""SessionRegistry: per-user session/turn/fact tracking for MemForest.

Tracks sessions, cells, turns, and the fact→cell provenance mapping.
Provides deletion bookkeeping and the orphan-detection + rebuild-state
helpers needed by UserForest.

No API calls; pure in-memory data management with JSON persistence.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.types import MemCell


# ── record types ──────────────────────────────────────────────────────────────

@dataclass
class TurnRecord:
    turn_id: str
    session_id: str
    cell_id: str
    deleted: bool = False


@dataclass
class CellRecord:
    cell_id: str
    session_id: str
    turn_ids: list[str]
    fact_ids: list[str]


@dataclass
class SessionRecord:
    session_id: str
    cell_ids: list[str]
    deleted: bool = False


@dataclass
class DeleteResult:
    deleted_session_ids: list[str]
    deleted_turn_ids: list[str]
    affected_cell_ids: list[str]


# ── registry ──────────────────────────────────────────────────────────────────

class SessionRegistry:
    """Tracks sessions, cells, turns, and fact→cell provenance for one user.

    Thread-safe via an internal RLock. All mutating operations hold the lock
    for their entire duration.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionRecord] = {}
        self._cells: dict[str, CellRecord] = {}
        self._turns: dict[str, TurnRecord] = {}
        # fact_id → set of cell_ids that produced it (for orphan detection)
        self._fact_to_cells: dict[str, set[str]] = {}
        self._lock = threading.RLock()

    # ── ingest ────────────────────────────────────────────────────────────────

    def register_session(
        self,
        session_id: str,
        cells: "list[MemCell]",
        cell_to_facts: dict[str, list[str]],
    ) -> None:
        """Record a newly ingested session.

        Args:
            session_id: The raw session identifier (internal, not aliased).
            cells: MemCell objects produced by ChunkExtractionPipeline.
            cell_to_facts: {cell_id: [canonical_fact_id, ...]} from post-ingest
                FactOccurrence data.
        """
        with self._lock:
            if session_id in self._sessions:
                # Re-ingest: merge cell lists without overwriting deleted flag
                existing = self._sessions[session_id]
                merged_cell_ids = list(
                    dict.fromkeys(existing.cell_ids + [c.cell_id for c in cells])
                )
                self._sessions[session_id] = SessionRecord(
                    session_id=session_id,
                    cell_ids=merged_cell_ids,
                    deleted=existing.deleted,
                )
            else:
                self._sessions[session_id] = SessionRecord(
                    session_id=session_id,
                    cell_ids=[c.cell_id for c in cells],
                    deleted=False,
                )

            for cell in cells:
                fact_ids = list(cell_to_facts.get(cell.cell_id, []))
                self._cells[cell.cell_id] = CellRecord(
                    cell_id=cell.cell_id,
                    session_id=session_id,
                    turn_ids=list(cell.turn_ids),
                    fact_ids=fact_ids,
                )
                for turn_id in cell.turn_ids:
                    self._turns[turn_id] = TurnRecord(
                        turn_id=turn_id,
                        session_id=session_id,
                        cell_id=cell.cell_id,
                        deleted=False,
                    )
                for fact_id in fact_ids:
                    self._fact_to_cells.setdefault(fact_id, set()).add(cell.cell_id)

    def register_synthetic_session(
        self,
        session_id: str,
        cell_ids: list[str],
        cell_fact_ids: dict[str, list[str]],
    ) -> None:
        """Register a session imported from legacy snapshot (no turn data).

        Used by MemForest.import_user_from_legacy. CellRecords are created
        with empty turn_ids; TurnRecords are NOT created (turn-level deletion
        is unavailable for imported sessions).
        """
        with self._lock:
            self._sessions[session_id] = SessionRecord(
                session_id=session_id,
                cell_ids=list(cell_ids),
                deleted=False,
            )
            for cell_id in cell_ids:
                fact_ids = list(cell_fact_ids.get(cell_id, []))
                self._cells[cell_id] = CellRecord(
                    cell_id=cell_id,
                    session_id=session_id,
                    turn_ids=[],   # no turn data in legacy snapshots
                    fact_ids=fact_ids,
                )
                for fact_id in fact_ids:
                    self._fact_to_cells.setdefault(fact_id, set()).add(cell_id)

    # ── deletion ──────────────────────────────────────────────────────────────

    def delete_session(self, session_id: str) -> DeleteResult:
        """Mark an entire session and all its turns as deleted.

        Raises:
            KeyError: If session_id is not registered.
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session {session_id!r} not registered")
            rec = self._sessions[session_id]
            self._sessions[session_id] = SessionRecord(
                session_id=session_id,
                cell_ids=rec.cell_ids,
                deleted=True,
            )
            for cell_id in rec.cell_ids:
                cell_rec = self._cells.get(cell_id)
                if cell_rec is None:
                    continue
                for turn_id in cell_rec.turn_ids:
                    if turn_id in self._turns:
                        t = self._turns[turn_id]
                        self._turns[turn_id] = TurnRecord(
                            turn_id=turn_id,
                            session_id=session_id,
                            cell_id=t.cell_id,
                            deleted=True,
                        )
            return DeleteResult(
                deleted_session_ids=[session_id],
                deleted_turn_ids=[],
                affected_cell_ids=list(rec.cell_ids),
            )

    def delete_turn(self, session_id: str, turn_id: str) -> DeleteResult:
        """Mark one turn as deleted.

        If all turns in the containing cell are now deleted, that cell_id is
        included in affected_cell_ids (for orphan detection).

        Raises:
            KeyError: If turn_id is not registered.
            ValueError: If turn belongs to a different session.
        """
        with self._lock:
            if turn_id not in self._turns:
                raise KeyError(f"Turn {turn_id!r} not found")
            t = self._turns[turn_id]
            if t.session_id != session_id:
                raise ValueError(
                    f"Turn {turn_id!r} belongs to session {t.session_id!r}, "
                    f"not {session_id!r}"
                )
            self._turns[turn_id] = TurnRecord(
                turn_id=turn_id,
                session_id=session_id,
                cell_id=t.cell_id,
                deleted=True,
            )
            # Check if the containing cell is now fully deleted
            affected_cells: list[str] = []
            cell_rec = self._cells.get(t.cell_id)
            if cell_rec and cell_rec.turn_ids:
                all_deleted = all(
                    self._turns.get(tid, TurnRecord("", "", "", True)).deleted
                    for tid in cell_rec.turn_ids
                )
                if all_deleted:
                    affected_cells.append(t.cell_id)

            return DeleteResult(
                deleted_session_ids=[],
                deleted_turn_ids=[turn_id],
                affected_cell_ids=affected_cells,
            )

    # ── orphan detection ──────────────────────────────────────────────────────

    def compute_orphaned_facts(self, all_fact_ids: set[str]) -> set[str]:
        """Return fact_ids that have no surviving producing cells.

        A fact is orphaned iff ALL cells that produced it are fully dead.
        A cell is dead if:
          (a) its session is marked deleted, OR
          (b) all its turn records are marked deleted
              (for legacy sessions with no turn records, this condition
               is NOT satisfied — cells survive unless the session is deleted)
        """
        with self._lock:
            orphaned: set[str] = set()
            for fact_id in all_fact_ids:
                producing_cells = self._fact_to_cells.get(fact_id, set())
                if not producing_cells:
                    # No provenance — conservatively keep
                    continue
                all_dead = True
                for cell_id in producing_cells:
                    cell_rec = self._cells.get(cell_id)
                    if cell_rec is None:
                        # Unknown cell — conservatively treat as alive
                        all_dead = False
                        break
                    session_rec = self._sessions.get(cell_rec.session_id)
                    if session_rec and session_rec.deleted:
                        continue  # session deleted → cell dead
                    # Check individual turn deletion
                    if not cell_rec.turn_ids:
                        # Legacy cell with no turn data — alive unless session deleted
                        all_dead = False
                        break
                    cell_fully_deleted = all(
                        self._turns.get(tid, TurnRecord("", "", "", False)).deleted
                        for tid in cell_rec.turn_ids
                    )
                    if not cell_fully_deleted:
                        all_dead = False
                        break
                if all_dead:
                    orphaned.add(fact_id)
            return orphaned

    # ── rebuild helpers ───────────────────────────────────────────────────────

    def build_surviving_session_id_to_cells(
        self,
        cell_store: "dict[str, MemCell]",
    ) -> "dict[str, list[MemCell]]":
        """Build {session_id: [surviving MemCell, ...]} for tree rebuild.

        A cell survives if its session is not deleted AND at least one of its
        turns is alive (or it is a legacy cell with no turn data).
        """
        with self._lock:
            result: dict[str, list[MemCell]] = {}
            for session_id, session_rec in self._sessions.items():
                if session_rec.deleted:
                    continue
                surviving: list["MemCell"] = []
                for cell_id in session_rec.cell_ids:
                    cell_rec = self._cells.get(cell_id)
                    if cell_rec is None:
                        continue
                    if cell_rec.turn_ids:
                        # Normal cell: alive if any turn survives
                        if not any(
                            not self._turns.get(tid, TurnRecord("", "", "", True)).deleted
                            for tid in cell_rec.turn_ids
                        ):
                            continue
                    # Legacy cell (no turns): always alive in surviving session
                    mc = cell_store.get(cell_id)
                    if mc is not None:
                        surviving.append(mc)
                if surviving:
                    result[session_id] = surviving
            return result

    def build_surviving_cell_to_facts(
        self,
        orphaned_fact_ids: set[str],
    ) -> dict[str, list[str]]:
        """Build {cell_id: [fact_id, ...]} for surviving cells, excluding orphaned facts."""
        with self._lock:
            result: dict[str, list[str]] = {}
            for cell_id, cell_rec in self._cells.items():
                session_rec = self._sessions.get(cell_rec.session_id)
                if session_rec and session_rec.deleted:
                    continue
                surviving_facts = [
                    fid for fid in cell_rec.fact_ids
                    if fid not in orphaned_fact_ids
                ]
                if surviving_facts:
                    result[cell_id] = surviving_facts
            return result

    # ── query helpers ─────────────────────────────────────────────────────────

    def list_sessions(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())

    def list_active_sessions(self) -> list[str]:
        with self._lock:
            return [sid for sid, rec in self._sessions.items() if not rec.deleted]

    def has_session(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def get_cell_fact_ids(self, cell_id: str) -> list[str]:
        """Return canonical fact IDs registered for one cell."""
        with self._lock:
            rec = self._cells.get(cell_id)
            if rec is None:
                return []
            return list(rec.fact_ids)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        with self._lock:
            data = {
                "sessions": {
                    sid: asdict(rec)
                    for sid, rec in self._sessions.items()
                },
                "cells": {
                    cid: asdict(rec)
                    for cid, rec in self._cells.items()
                },
                "turns": {
                    tid: asdict(rec)
                    for tid, rec in self._turns.items()
                },
                "fact_to_cells": {
                    fid: list(cell_ids)
                    for fid, cell_ids in self._fact_to_cells.items()
                },
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        with self._lock:
            self._sessions = {
                sid: SessionRecord(**rec)
                for sid, rec in (data.get("sessions") or {}).items()
            }
            self._cells = {
                cid: CellRecord(**rec)
                for cid, rec in (data.get("cells") or {}).items()
            }
            self._turns = {
                tid: TurnRecord(**rec)
                for tid, rec in (data.get("turns") or {}).items()
            }
            self._fact_to_cells = {
                fid: set(cells)
                for fid, cells in (data.get("fact_to_cells") or {}).items()
            }
