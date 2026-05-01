"""Parallel LLM summary dispatcher for tree node generation.

Mirrors ExtractionManager's flat-job + ThreadPoolExecutor + as_completed pattern,
enabling cross-tree parallelism within and between summary levels.
"""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from src.build.tree_types import SummaryRequest, SummaryResult
from src.config.tree_config import SummaryManagerConfig
from src.prompt.tree_prompts import build_tree_prompt_manager
from src.utils.time import render_time_text

if TYPE_CHECKING:
    from src.api.client import OpenAIChatClient
    from src.prompt.prompt import PromptManager


class SummaryManager:
    """Dispatch SummaryRequests in parallel across trees and levels.

    Usage:
        manager = SummaryManager(chat_client=..., config=..., model_name=...)
        results = manager.generate_summaries(requests, show_progress=True)
    """

    def __init__(
        self,
        *,
        chat_client: "OpenAIChatClient",
        config: SummaryManagerConfig,
        model_name: str,
        prompt_manager: "PromptManager | None" = None,
    ) -> None:
        self._chat_client = chat_client
        self._config = config
        self._model_name = model_name
        self._prompt_manager = prompt_manager or build_tree_prompt_manager()
        self._calls_lock = threading.Lock()
        self._total_calls: int = 0

    @property
    def total_calls(self) -> int:
        return self._total_calls

    def generate_summaries(
        self,
        requests: list[SummaryRequest],
        *,
        show_progress: bool = False,
    ) -> list[SummaryResult]:
        """Execute all SummaryRequests in parallel (up to max_inflight at once).

        Returns results in the same order as the input list.
        """
        if not requests:
            return []

        results: list[SummaryResult | None] = [None] * len(requests)
        worker_count = min(self._config.max_inflight, len(requests))

        if worker_count <= 1:
            for idx, req in enumerate(requests):
                results[idx] = self._call_one(req)
                if show_progress:
                    _print_progress(idx + 1, len(requests))
            return [r for r in results if r is not None]

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_idx = {
                executor.submit(self._call_one, req): idx
                for idx, req in enumerate(requests)
            }
            done = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()  # propagates exceptions
                done += 1
                if show_progress:
                    _print_progress(done, len(requests))

        if show_progress:
            sys.stderr.write("\n")
            sys.stderr.flush()

        return [r for r in results if r is not None]

    def _call_one(self, req: SummaryRequest) -> SummaryResult:
        """Render prompt and call LLM for a single node summary.

        Retries up to ``config.max_retries`` times when the response comes back
        empty without an explicit error (silent JSON truncation path).
        """
        prompt_name = _prompt_name_for(req.tree_type, req.tree_label)
        context = {
            "session_id": req.tree_label if req.tree_type == "session" else req.tree_id,
            "tree_label": req.tree_label,
            "time_range_text": _render_time_range(req.time_start, req.time_end),
            "level": req.level,
            "input_text": req.input_text,
            "max_words": _max_words_for(req.tree_type),
        }
        system_prompt, user_prompt = self._prompt_manager.render(prompt_name, context)

        attempts = 1 + self._config.max_retries
        start = time.time()
        summary = ""
        error: str | None = None

        for attempt in range(attempts):
            try:
                response = self._chat_client.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model_name=self._model_name,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                    top_p=self._config.top_p,
                    timeout=self._config.timeout_sec,
                    step_label="tree_summary",
                    trace={
                        "request_id": req.request_id,
                        "tree_id": req.tree_id,
                        "node_id": req.node_id,
                        "attempt": attempt,
                    },
                )
                summary = str(response.get("summary") or "").strip()
                if not summary:
                    summary = " ".join(
                        str(v) for v in response.values() if isinstance(v, str)
                    ).strip()
                error = None
            except Exception as exc:
                summary = ""
                error = f"{type(exc).__name__}: {exc}"

            # Stop retrying if we have a result or a hard error
            if summary or error:
                break

        elapsed = time.time() - start
        with self._calls_lock:
            self._total_calls += 1

        return SummaryResult(
            request_id=req.request_id,
            tree_id=req.tree_id,
            node_id=req.node_id,
            summary=summary,
            elapsed_sec=elapsed,
            error=error,
        )


def _prompt_name_for(tree_type: str, tree_label: str) -> str:
    if tree_type == "session":
        return "tree_summary_session"
    if tree_type == "entity":
        return "tree_summary_entity_user" if tree_label == "user" else "tree_summary_entity_generic"
    return "tree_summary_scene"


def _max_words_for(tree_type: str) -> int:
    return 250 if tree_type == "entity" else 200


def _render_time_range(time_start: float, time_end: float) -> str:
    try:
        start_str = render_time_text(time_start)
        end_str = render_time_text(time_end)
        return f"{start_str} to {end_str}"
    except (OSError, OverflowError, ValueError):
        return f"{time_start:.0f} – {time_end:.0f}"


def _print_progress(done: int, total: int) -> None:
    pct = 100.0 * done / total if total else 100.0
    sys.stderr.write(f"\r[summary] {done}/{total} ({pct:.1f}%)")
    sys.stderr.flush()
