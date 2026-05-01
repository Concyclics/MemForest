"""Per-request extraction logger.

Writes one JSON line per ExtractionRequest to a configurable JSONL file.

Each record contains:
- request_id:      unique request identifier
- session_id:      session being extracted
- cell_count:      number of MemCells produced by chunking
- item_count:      total MemoryItems extracted across all cells
- cell_ids:        list of cell_id strings for tracing
- cell_item_counts: per-cell item counts
- start_time:      unix timestamp when extraction started
- end_time:        unix timestamp when extraction completed
- elapsed_sec:     wall-clock duration for the full request
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class ExtractionLogger:
    """Thread-safe JSONL logger for extraction requests."""

    def __init__(self, path: str | Path, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._path = Path(path)
        self._lock = threading.Lock()
        if self.enabled:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        request_id: str,
        session_id: str,
        turn_count: int,
        cell_count: int,
        item_count: int,
        cell_ids: list[str],
        cell_item_counts: list[int],
        metadata: dict[str, Any],
        success: bool,
        error: str | None,
        start_time: float,
        end_time: float,
    ) -> None:
        """Append one record to the JSONL log file (no-op when disabled)."""
        if not self.enabled:
            return
        record: dict[str, Any] = {
            "request_id": request_id,
            "session_id": session_id,
            "turn_count": turn_count,
            "cell_count": cell_count,
            "item_count": item_count,
            "cell_ids": cell_ids,
            "cell_item_counts": cell_item_counts,
            "metadata": metadata,
            "success": success,
            "error": error,
            "start_time": start_time,
            "end_time": end_time,
            "elapsed_sec": end_time - start_time,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
