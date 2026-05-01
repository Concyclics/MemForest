"""Session normalization and deterministic chunking."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

from src.utils.text import normalize_entity_key
from src.utils.time import parse_timestamp_to_unix, render_time_text
from src.utils.types import ChunkingConfig, MemCell, NormalizedTurn


def normalize_turns(session_id: str, turns: Sequence[dict[str, Any]]) -> list[NormalizedTurn]:
    """Normalize raw turn dictionaries into stable typed turns."""

    out: list[NormalizedTurn] = []
    for idx, turn in enumerate(turns):
        speaker_tag = str(turn.get("role", turn.get("speaker_tag", "unknown"))).strip() or "unknown"
        speaker_name = str(turn.get("speaker_name", speaker_tag)).strip() or speaker_tag
        listener_name = str(turn.get("listener_name", "listener")).strip() or "listener"
        text = str(turn.get("content", turn.get("text", ""))).strip()
        if not text:
            continue
        timestamp_raw = turn.get("timestamp") or turn.get("time_stamp") or turn.get("session_time")
        if timestamp_raw is None:
            timestamp_unix = float(idx)
            timestamp_text = render_time_text(timestamp_unix)
        else:
            timestamp_unix = parse_timestamp_to_unix(timestamp_raw)
            timestamp_text = str(timestamp_raw)
        turn_id = str(turn.get("content_id", f"{session_id}#turn_{idx:04d}"))
        out.append(
            NormalizedTurn(
                turn_id=turn_id,
                session_id=session_id,
                turn_index=idx,
                speaker_tag=speaker_tag,
                speaker_name=speaker_name,
                listener_name=listener_name,
                text=text,
                timestamp_text=timestamp_text,
                timestamp_unix=timestamp_unix,
                metadata={k: v for k, v in turn.items() if k not in {"content", "text"}},
            )
        )
    return out


def chunk_session(
    session_id: str,
    turns: Sequence[NormalizedTurn] | Sequence[dict[str, Any]],
    *,
    config: ChunkingConfig = ChunkingConfig(),
) -> list[MemCell]:
    """Split one session into deterministic extraction chunks."""

    normalized = (
        list(turns)
        if turns and isinstance(turns[0], NormalizedTurn)
        else normalize_turns(session_id, turns)  # type: ignore[arg-type]
    )
    if not normalized:
        return []

    chunks: list[MemCell] = []
    current: list[NormalizedTurn] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current, current_chars
        if not current:
            return
        chunk_index = len(chunks)
        start = current[0]
        end = current[-1]
        chunk_text = _render_chunk_text(current)
        digest = hashlib.md5(
            f"{session_id}|{start.turn_index}|{end.turn_index}|{chunk_text}".encode("utf-8")
        ).hexdigest()[:12]
        chunks.append(
            MemCell(
                cell_id=f"{session_id}#cell_{chunk_index:04d}_{digest}",
                session_id=session_id,
                cell_index=chunk_index,
                start_turn_index=start.turn_index,
                end_turn_index=end.turn_index,
                turn_ids=[turn.turn_id for turn in current],
                content_ids=[turn.turn_id for turn in current],
                speaker_names=sorted({turn.speaker_name for turn in current}),
                time_start=float(start.timestamp_unix),
                time_end=float(end.timestamp_unix),
                text=chunk_text,
                turns=list(current),
            )
        )
        current = []
        current_chars = 0

    last_turn: NormalizedTurn | None = None
    for turn in normalized:
        should_split = False
        if current:
            if len(current) >= int(config.max_turns):
                should_split = True
            elif current_chars + len(turn.text) > int(config.max_chars):
                should_split = True
            elif last_turn is not None and abs(turn.timestamp_unix - last_turn.timestamp_unix) > float(config.max_time_gap_seconds):
                should_split = True
            elif any(marker in turn.text.lower() for marker in config.hard_boundary_markers):
                should_split = True
        if should_split:
            flush()
        current.append(turn)
        current_chars += len(turn.text)
        last_turn = turn
    flush()
    return chunks


def _render_chunk_text(turns: Sequence[NormalizedTurn]) -> str:
    lines: list[str] = []
    for turn in turns:
        speaker = normalize_entity_key(turn.speaker_name) or turn.speaker_name
        lines.append(f"[{turn.timestamp_text}] {speaker}: {turn.text}")
    return "\n".join(lines)
