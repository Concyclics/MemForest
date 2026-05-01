"""Logger configuration for vNext pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ApiLogConfig:
    enabled: bool = True
    path: str = "data/logs/api_calls.jsonl"


@dataclass(frozen=True)
class ExtractionLogConfig:
    enabled: bool = True
    path: str = "data/logs/extraction_requests.jsonl"


@dataclass(frozen=True)
class LoggerConfig:
    api: ApiLogConfig = field(default_factory=ApiLogConfig)
    extraction: ExtractionLogConfig = field(default_factory=ExtractionLogConfig)
