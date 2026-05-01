"""API-related configuration models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatCompletionConfig:
    url: str
    model_name: str
    key: str
    topk: int | None = None
    top_p: float = 1.0
    max_token: int = 1024
    temperature: float = 0.0


@dataclass(frozen=True)
class EmbeddingConfig:
    url: str
    model_name: str
    key: str
    dimension: int


@dataclass(frozen=True)
class APISettings:
    llm: ChatCompletionConfig
    embedding: EmbeddingConfig

