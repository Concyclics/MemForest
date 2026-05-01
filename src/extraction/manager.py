"""Request-based extraction manager with global request-level parallel scheduling."""

from __future__ import annotations

import hashlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from src.extraction.pipeline import ChunkExtractionPipeline
from src.utils.types import (
    CellExtractionRequest,
    CellExtractionResult,
    ExtractionRequest,
    SessionExtractionResult,
)

if TYPE_CHECKING:
    from src.logger.extraction_log import ExtractionLogger


class ExtractionManager:
    """Coordinate extraction requests without partitioning concurrency by session."""

    def __init__(
        self,
        *,
        pipeline: ChunkExtractionPipeline,
        max_inflight_requests: int = 1,
        extraction_logger: "ExtractionLogger | None" = None,
    ) -> None:
        self.pipeline = pipeline
        self.max_inflight_requests = max(1, int(max_inflight_requests))
        self._extraction_logger = extraction_logger

    def extract_request(self, request: ExtractionRequest) -> SessionExtractionResult:
        return self.extract_requests([request])[0]

    def extract_requests(
        self,
        requests: list[ExtractionRequest],
        *,
        show_progress: bool = False,
        progress_label: str = "extract",
    ) -> list[SessionExtractionResult]:
        if not requests:
            return []

        start_times: dict[int, float] = {}
        end_times: dict[int, float] = {}
        prepared: list[tuple[ExtractionRequest, list]] = []
        for idx, request in enumerate(requests):
            start_times[idx] = time.time()
            prepared.append((request, self.pipeline.build_cells(request.session_id, request.turns)))
        total_cells = sum(len(cells) for _, cells in prepared)
        progress = _ProgressReporter(
            enabled=show_progress,
            label=progress_label,
            total_requests=len(requests),
            total_cells=total_cells,
        )

        if self.max_inflight_requests <= 1:
            results_by_index: list[SessionExtractionResult | None] = [None] * len(prepared)
            for idx, (request, cells) in enumerate(prepared):
                cell_results = [
                    self.pipeline.extract_cell_request(_build_cell_request(request, cell))
                    for cell in cells
                ]
                results_by_index[idx] = _build_session_result(request, cells, cell_results)
                end_times[idx] = time.time()
                progress.update(completed_cells=len(cells), completed_requests=1)
            self._log_results(requests, prepared, results_by_index, start_times, end_times, errors={})
            progress.finish()
            return [result for result in results_by_index if result is not None]

        per_request_results: list[list[CellExtractionResult | None]] = [
            [None] * len(cells) for _, cells in prepared
        ]
        cell_jobs: list[tuple[int, int, CellExtractionRequest]] = []
        for request_index, (request, cells) in enumerate(prepared):
            request_id = request.request_id or _stable_request_id(request.session_id, request.turns)
            for cell_index, cell in enumerate(cells):
                cell_jobs.append(
                    (
                        request_index,
                        cell_index,
                        CellExtractionRequest(
                            session_id=request.session_id,
                            cell=cell,
                            request_id=f"{request_id}:cell:{cell.cell_index}",
                        ),
                    )
                )

        if not cell_jobs:
            results_by_index: list[SessionExtractionResult | None] = [
                SessionExtractionResult(session_id=request.session_id, cells=cells, memory_items=[])
                for request, cells in prepared
            ]
            now = time.time()
            for idx in range(len(prepared)):
                end_times[idx] = now
            progress.update(completed_requests=len(prepared))
            self._log_results(requests, prepared, results_by_index, start_times, end_times, errors={})
            progress.finish()
            return [result for result in results_by_index if result is not None]

        worker_count = min(self.max_inflight_requests, len(cell_jobs))
        errors: dict[int, str] = {}
        remaining_cells = {idx: len(cells) for idx, (_, cells) in enumerate(prepared)}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_job = {
                executor.submit(self.pipeline.extract_cell_request, cell_request): (request_index, cell_index)
                for request_index, cell_index, cell_request in cell_jobs
            }
            for future in as_completed(future_to_job):
                request_index, cell_index = future_to_job[future]
                try:
                    per_request_results[request_index][cell_index] = future.result()
                    remaining_cells[request_index] -= 1
                    progress.update(completed_cells=1)
                    if remaining_cells[request_index] == 0:
                        end_times[request_index] = time.time()
                        progress.update(completed_requests=1)
                except Exception as exc:
                    errors[request_index] = f"{type(exc).__name__}: {exc}"
                    end_times[request_index] = time.time()
                    self._log_results(
                        requests,
                        prepared,
                        [None] * len(prepared),
                        start_times,
                        end_times,
                        errors=errors,
                    )
                    progress.finish()
                    raise

        results_by_index: list[SessionExtractionResult | None] = [None] * len(prepared)
        for idx, ((request, cells), cell_results) in enumerate(zip(prepared, per_request_results)):
            ordered = [result for result in cell_results if result is not None]
            results_by_index[idx] = _build_session_result(request, cells, ordered)
            end_times.setdefault(idx, time.time())
        self._log_results(requests, prepared, results_by_index, start_times, end_times, errors=errors)
        progress.finish()
        return [result for result in results_by_index if result is not None]

    def _log_results(
        self,
        requests: list[ExtractionRequest],
        prepared: list[tuple[ExtractionRequest, list]],
        results: list[SessionExtractionResult | None],
        start_times: dict[int, float],
        end_times: dict[int, float],
        errors: dict[int, str],
    ) -> None:
        if self._extraction_logger is None:
            return
        for idx, request in enumerate(requests):
            req_id = request.request_id or _stable_request_id(request.session_id, request.turns)
            cells = prepared[idx][1]
            result = results[idx] if idx < len(results) else None
            cell_ids = [c.cell_id for c in cells]
            cell_item_counts = [
                0 if result is None else sum(1 for item in result.memory_items if item.cell_id == c.cell_id)
                for c in cells
            ]
            self._extraction_logger.log(
                request_id=req_id,
                session_id=request.session_id,
                turn_count=len(request.turns),
                cell_count=len(cells),
                item_count=0 if result is None else len(result.memory_items),
                cell_ids=cell_ids,
                cell_item_counts=cell_item_counts,
                metadata=dict(request.metadata),
                success=idx not in errors,
                error=errors.get(idx),
                start_time=start_times[idx],
                end_time=end_times.get(idx, time.time()),
            )


def _build_session_result(
    request: ExtractionRequest,
    cells: list,
    cell_results: list[CellExtractionResult],
) -> SessionExtractionResult:
    memory_items = [item for result in cell_results for item in result.memory_items]
    return SessionExtractionResult(
        session_id=request.session_id,
        cells=cells,
        memory_items=memory_items,
    )


def _stable_request_id(session_id: str, turns: list[dict]) -> str:
    h = hashlib.sha1()
    h.update(session_id.encode("utf-8"))
    for turn in turns:
        h.update(str(turn.get("content", "")).encode("utf-8"))
        h.update(str(turn.get("timestamp", "")).encode("utf-8"))
    return f"req_{h.hexdigest()[:12]}"


def _build_cell_request(request: ExtractionRequest, cell) -> CellExtractionRequest:
    request_id = request.request_id or _stable_request_id(request.session_id, request.turns)
    return CellExtractionRequest(
        session_id=request.session_id,
        cell=cell,
        request_id=f"{request_id}:cell:{cell.cell_index}",
    )


class _ProgressReporter:
    def __init__(
        self,
        *,
        enabled: bool,
        label: str,
        total_requests: int,
        total_cells: int,
    ) -> None:
        self.enabled = enabled
        self.label = label
        self.total_requests = max(0, int(total_requests))
        self.total_cells = max(0, int(total_cells))
        self.completed_requests = 0
        self.completed_cells = 0
        self.started = time.time()
        self._last_render = 0.0
        self._finished = False

    def update(self, *, completed_cells: int = 0, completed_requests: int = 0) -> None:
        if not self.enabled:
            return
        self.completed_cells += max(0, int(completed_cells))
        self.completed_requests += max(0, int(completed_requests))
        now = time.time()
        if now - self._last_render < 0.1 and self.completed_requests < self.total_requests:
            return
        self._last_render = now
        elapsed = now - self.started
        req_pct = 100.0 if self.total_requests == 0 else (self.completed_requests / self.total_requests) * 100.0
        msg = (
            f"\r[{self.label}] requests {self.completed_requests}/{self.total_requests} "
            f"({req_pct:5.1f}%) | cells {self.completed_cells}/{self.total_cells} "
            f"| elapsed {elapsed:6.1f}s"
        )
        sys.stderr.write(msg)
        sys.stderr.flush()

    def finish(self) -> None:
        if not self.enabled or self._finished:
            return
        if self.completed_requests < self.total_requests or self.completed_cells < self.total_cells:
            self.update()
        sys.stderr.write("\n")
        sys.stderr.flush()
        self._finished = True
