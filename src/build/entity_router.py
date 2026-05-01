"""Entity routing with lifecycle-aware activation.

The router keeps every non-noise entity as a candidate, accumulates support
across incremental writes, and only exposes active entity trees to the build
pipeline. Low-support entities remain lazy instead of being dropped.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from src.build.tree_types import EntityCandidate, EntityManifestRecord
from src.config.tree_config import EntityTreeConfig

if TYPE_CHECKING:
    from src.utils.types import ManagedFact


ENTITY_USER_TREE_ID = "entity:user"

_INVALID_ENTITY_RE = re.compile(r"^[\W_]+$")
_TIME_LIKE_RE = re.compile(r"^\d{1,4}([:/-]\d{1,4}){1,3}$")


class EntityRouter:
    """Routes facts into the user tree and lifecycle-managed entity trees."""

    def __init__(self, *, config: EntityTreeConfig) -> None:
        self._config = config
        self._candidates: dict[str, EntityCandidate] = {}
        self._lock = threading.RLock()

    def assign(self, fact: "ManagedFact") -> list[str]:
        """Observe a fact and return active entity tree IDs for it."""
        tree_ids: list[str] = []
        with self._lock:
            if fact.origin == "user":
                tree_ids.append(ENTITY_USER_TREE_ID)
            for entity in fact.entities:
                entity_key = _normalise_entity(entity)
                if not entity_key:
                    continue
                candidate = self._upsert_candidate(entity_key, entity, fact)
                if candidate.state == "active":
                    tree_id = f"entity:{entity_key}"
                    if tree_id not in tree_ids:
                        tree_ids.append(tree_id)
        return tree_ids

    def all_active_entity_keys(self) -> list[str]:
        with self._lock:
            return sorted(
                key for key, candidate in self._candidates.items() if candidate.state == "active"
            )

    def all_lazy_entity_keys(self) -> list[str]:
        with self._lock:
            return sorted(
                key for key, candidate in self._candidates.items() if candidate.state == "lazy"
            )

    def candidate(self, entity_key: str) -> EntityCandidate | None:
        with self._lock:
            return self._candidates.get(entity_key)

    def iter_candidates(self) -> list[EntityCandidate]:
        with self._lock:
            return [self._copy_candidate(candidate) for candidate in self._candidates.values()]

    def active_manifests(self) -> list[EntityManifestRecord]:
        with self._lock:
            return [
                _manifest(candidate)
                for candidate in self._candidates.values()
                if candidate.state == "active"
            ]

    def lazy_manifests(self) -> list[EntityManifestRecord]:
        with self._lock:
            return [
                _manifest(candidate)
                for candidate in self._candidates.values()
                if candidate.state == "lazy"
            ]

    def suppressed_manifests(self) -> list[EntityManifestRecord]:
        with self._lock:
            return [
                _manifest(candidate)
                for candidate in self._candidates.values()
                if candidate.state == "suppressed"
            ]

    def apply_suppression(self, suppressed_keys: set[str]) -> None:
        with self._lock:
            for entity_key, candidate in list(self._candidates.items()):
                if entity_key == "user":
                    continue
                next_state = "suppressed" if entity_key in suppressed_keys else _activation_state(candidate, self._config)
                self._candidates[entity_key] = self._copy_candidate(candidate, state=next_state)

    def merge_from(
        self,
        other: "EntityRouter | str | Path",
    ) -> None:
        """Merge all entity candidates from *other* into this router.

        For each source candidate, if the entity key already exists in this
        router, fact_ids and session_ids are unioned and counts recalculated.
        Otherwise the candidate is added directly.  Activation state is
        recomputed after merging.

        ``other`` can be an :class:`EntityRouter` instance or a path to a
        saved router JSON file.
        """
        if isinstance(other, (str, Path)):
            source_candidates = self._load_candidates_from_path(Path(other))
        else:
            with other._lock:
                source_candidates = {
                    k: self._copy_candidate(v) for k, v in other._candidates.items()
                }

        with self._lock:
            for entity_key, src in source_candidates.items():
                existing = self._candidates.get(entity_key)
                if existing is None:
                    # New entity — add directly and recompute state.
                    merged = self._copy_candidate(
                        src,
                        state=_activation_state(src, self._config),
                    )
                    self._candidates[entity_key] = merged
                else:
                    # Merge: union fact_ids and session_ids.
                    merged_fact_ids = list(existing.fact_ids)
                    for fid in src.fact_ids:
                        if fid not in merged_fact_ids:
                            merged_fact_ids.append(fid)
                    merged_session_ids = list(existing.session_ids)
                    for sid in src.session_ids:
                        if sid and sid not in merged_session_ids:
                            merged_session_ids.append(sid)
                    merged = self._copy_candidate(
                        existing,
                        display_label=_preferred_display_label(
                            existing.display_label, src.display_label, entity_key
                        ),
                        support_count=existing.support_count + src.support_count,
                        distinct_fact_count=len(merged_fact_ids),
                        distinct_session_count=len(merged_session_ids),
                        first_seen_at=_min_time(existing.first_seen_at, src.first_seen_at),
                        last_seen_at=_max_time(existing.last_seen_at, src.last_seen_at),
                        fact_ids=merged_fact_ids,
                        session_ids=merged_session_ids,
                    )
                    merged = self._copy_candidate(
                        merged,
                        state=_activation_state(merged, self._config),
                    )
                    self._candidates[entity_key] = merged

    @staticmethod
    def _load_candidates_from_path(path: Path) -> dict[str, EntityCandidate]:
        """Load candidates from a saved router JSON file."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        candidates: dict[str, EntityCandidate] = {}
        for key, value in (raw.get("candidates") or {}).items():
            candidates[key] = EntityCandidate(
                entity_key=str(value.get("entity_key", key)),
                display_label=str(value.get("display_label", key)),
                support_count=int(value.get("support_count", 0)),
                distinct_fact_count=int(value.get("distinct_fact_count", 0)),
                distinct_session_count=int(value.get("distinct_session_count", 0)),
                first_seen_at=_maybe_float(value.get("first_seen_at")),
                last_seen_at=_maybe_float(value.get("last_seen_at")),
                fact_ids=list(value.get("fact_ids") or []),
                session_ids=list(value.get("session_ids") or []),
                state=str(value.get("state", "lazy")),
            )
        return candidates

    def remove_fact_ids(
        self,
        fact_ids_to_remove: set[str],
        remaining_fact_lookup: dict[str, "ManagedFact"],
    ) -> None:
        if not fact_ids_to_remove:
            return
        with self._lock:
            updated: dict[str, EntityCandidate] = {}
            for entity_key, candidate in self._candidates.items():
                kept_fact_ids = [fid for fid in candidate.fact_ids if fid not in fact_ids_to_remove]
                if entity_key != "user" and not kept_fact_ids:
                    continue
                kept_facts = [remaining_fact_lookup[fid] for fid in kept_fact_ids if fid in remaining_fact_lookup]
                if entity_key == "user":
                    display_label = candidate.display_label or "user"
                else:
                    display_label = candidate.display_label or entity_key
                first_seen_values = [fact.time_start for fact in kept_facts if fact.time_start is not None]
                last_seen_values = [
                    fact.time_end if fact.time_end is not None else fact.time_start
                    for fact in kept_facts
                    if (fact.time_end if fact.time_end is not None else fact.time_start) is not None
                ]
                session_ids = list(dict.fromkeys(
                    fact.first_session_id
                    for fact in kept_facts
                    if fact.first_session_id
                ))
                refreshed = self._copy_candidate(
                    candidate,
                    display_label=display_label,
                    support_count=len(kept_fact_ids),
                    distinct_fact_count=len(kept_fact_ids),
                    distinct_session_count=len(session_ids),
                    first_seen_at=min(first_seen_values) if first_seen_values else None,
                    last_seen_at=max(last_seen_values) if last_seen_values else None,
                    fact_ids=kept_fact_ids,
                    session_ids=session_ids,
                )
                refreshed = self._copy_candidate(
                    refreshed,
                    state=_activation_state(refreshed, self._config),
                )
                updated[entity_key] = refreshed
            self._candidates = updated

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                "candidates": {
                    key: {
                        "entity_key": candidate.entity_key,
                        "display_label": candidate.display_label,
                        "support_count": candidate.support_count,
                        "distinct_fact_count": candidate.distinct_fact_count,
                        "distinct_session_count": candidate.distinct_session_count,
                        "first_seen_at": candidate.first_seen_at,
                        "last_seen_at": candidate.last_seen_at,
                        "fact_ids": list(candidate.fact_ids),
                        "session_ids": list(candidate.session_ids),
                        "state": candidate.state,
                    }
                    for key, candidate in self._candidates.items()
                }
            }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        candidates_raw = raw.get("candidates") or {}
        restored: dict[str, EntityCandidate] = {}
        for key, value in candidates_raw.items():
            restored[key] = EntityCandidate(
                entity_key=str(value.get("entity_key", key)),
                display_label=str(value.get("display_label", key)),
                support_count=int(value.get("support_count", 0)),
                distinct_fact_count=int(value.get("distinct_fact_count", 0)),
                distinct_session_count=int(value.get("distinct_session_count", 0)),
                first_seen_at=_maybe_float(value.get("first_seen_at")),
                last_seen_at=_maybe_float(value.get("last_seen_at")),
                fact_ids=list(value.get("fact_ids") or []),
                session_ids=list(value.get("session_ids") or []),
                state=str(value.get("state", "lazy")),
            )
        with self._lock:
            self._candidates = restored

    def _upsert_candidate(
        self,
        entity_key: str,
        display_label: str,
        fact: "ManagedFact",
    ) -> EntityCandidate:
        candidate = self._candidates.get(entity_key)
        if candidate is None:
            candidate = EntityCandidate(
                entity_key=entity_key,
                display_label=display_label.strip() or entity_key,
                support_count=0,
                distinct_fact_count=0,
                distinct_session_count=0,
                first_seen_at=fact.time_start,
                last_seen_at=fact.time_end if fact.time_end is not None else fact.time_start,
                fact_ids=[],
                session_ids=[],
                state="lazy",
            )
        fact_ids = list(candidate.fact_ids)
        if fact.fact_id not in fact_ids:
            fact_ids.append(fact.fact_id)
        session_ids = list(candidate.session_ids)
        session_id = fact.first_session_id or ""
        if session_id and session_id not in session_ids:
            session_ids.append(session_id)
        updated = self._copy_candidate(
            candidate,
            display_label=_preferred_display_label(candidate.display_label, display_label, entity_key),
            support_count=candidate.support_count + 1,
            distinct_fact_count=len(fact_ids),
            distinct_session_count=len(session_ids),
            first_seen_at=_min_time(candidate.first_seen_at, fact.time_start),
            last_seen_at=_max_time(candidate.last_seen_at, fact.time_end if fact.time_end is not None else fact.time_start),
            fact_ids=fact_ids,
            session_ids=session_ids,
        )
        updated = self._copy_candidate(updated, state=_activation_state(updated, self._config))
        self._candidates[entity_key] = updated
        return updated

    @staticmethod
    def _copy_candidate(candidate: EntityCandidate, **overrides) -> EntityCandidate:
        return EntityCandidate(
            entity_key=str(overrides.get("entity_key", candidate.entity_key)),
            display_label=str(overrides.get("display_label", candidate.display_label)),
            support_count=int(overrides.get("support_count", candidate.support_count)),
            distinct_fact_count=int(overrides.get("distinct_fact_count", candidate.distinct_fact_count)),
            distinct_session_count=int(overrides.get("distinct_session_count", candidate.distinct_session_count)),
            first_seen_at=overrides.get("first_seen_at", candidate.first_seen_at),
            last_seen_at=overrides.get("last_seen_at", candidate.last_seen_at),
            fact_ids=list(overrides.get("fact_ids", candidate.fact_ids)),
            session_ids=list(overrides.get("session_ids", candidate.session_ids)),
            state=str(overrides.get("state", candidate.state)),
        )


def _activation_state(candidate: EntityCandidate, config: EntityTreeConfig) -> str:
    if candidate.entity_key == "user":
        return "active"
    if not _surface_is_specific(candidate.display_label):
        return "lazy"
    if (
        candidate.distinct_fact_count >= config.active_min_facts
        and candidate.distinct_session_count >= config.active_min_sessions
    ):
        return "active"
    return "lazy"


def _manifest(candidate: EntityCandidate) -> EntityManifestRecord:
    return EntityManifestRecord(
        entity_key=candidate.entity_key,
        display_label=candidate.display_label,
        state=candidate.state,
        support_count=candidate.support_count,
        distinct_fact_count=candidate.distinct_fact_count,
        distinct_session_count=candidate.distinct_session_count,
        fact_ids=list(candidate.fact_ids),
    )


def _normalise_entity(entity: str) -> str | None:
    text = str(entity).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"user", "assistant", "unknown"}:
        return None
    if _INVALID_ENTITY_RE.match(text):
        return None
    if _TIME_LIKE_RE.match(text):
        return None
    if text.isdigit():
        return None
    if len(text) > 64:
        return None
    if len(text.split()) > 8:
        return None
    normalized = lowered.replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        return None
    return normalized


def _surface_is_specific(label: str) -> bool:
    text = (label or "").strip()
    if not text:
        return False
    if len(text.split()) >= 2:
        return True
    if any(ch.isupper() for ch in text):
        return True
    if any(ch.isdigit() for ch in text):
        return True
    return False


def _preferred_display_label(current: str, incoming: str, fallback: str) -> str:
    current = (current or "").strip()
    incoming = (incoming or "").strip()
    if _surface_is_specific(current):
        return current
    if _surface_is_specific(incoming):
        return incoming
    return current or incoming or fallback


def _min_time(left: float | None, right: float | None) -> float | None:
    values = [v for v in (left, right) if v is not None]
    return min(values) if values else None


def _max_time(left: float | None, right: float | None) -> float | None:
    values = [v for v in (left, right) if v is not None]
    return max(values) if values else None


def _maybe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
