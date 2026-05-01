"""Per-call API request/response logger.

Writes one JSON line per LLM API call to a configurable JSONL file.

Each record contains:
- step_label:          which pipeline step triggered the call (e.g. "extraction")
- model_name:          the model used
- system_prompt:       full system prompt sent
- user_prompt:         full user prompt sent
- response_content:    raw response content from the model
- prompt_tokens:       input token count (from API usage field; -1 if unavailable)
- completion_tokens:   output token count (from API usage field; -1 if unavailable)
- start_time:          unix timestamp (float) when the call was initiated
- end_time:            unix timestamp (float) when the response was received
- elapsed_sec:         wall-clock duration of the call
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class ApiCallLogger:
    """Thread-safe JSONL logger for individual LLM API calls."""

    def __init__(self, path: str | Path, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._path = Path(path)
        self._lock = threading.Lock()
        if self.enabled:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        step_label: str,
        model_name: str,
        request_id: str | None,
        session_id: str | None,
        cell_id: str | None,
        cell_index: int | None,
        system_prompt: str,
        user_prompt: str,
        response_content: str,
        prompt_tokens: int,
        completion_tokens: int,
        start_time: float,
        end_time: float,
    ) -> None:
        """Append one record to the JSONL log file (no-op when disabled)."""
        if not self.enabled:
            return
        record: dict[str, Any] = {
            "step_label": step_label,
            "model_name": model_name,
            "request_id": request_id,
            "session_id": session_id,
            "cell_id": cell_id,
            "cell_index": cell_index,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response_content": response_content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "start_time": start_time,
            "end_time": end_time,
            "elapsed_sec": end_time - start_time,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
