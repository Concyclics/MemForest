"""Extraction-step configuration for vNext."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.utils.types import ChunkingConfig


@dataclass(frozen=True)
class FactManagerConfig:
    enabled: bool = False
    storage_dir: str = "data/index/vnext_fact_manager"
    persist_on_write: bool = False
    top_k: int = 8
    similarity_threshold: float = 0.93
    max_llm_pairs_per_item: int = 4
    normalize_embeddings: bool = True


@dataclass(frozen=True)
class ExtractionConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    use_llm_extraction: bool = True
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 0.8
    max_workers: int = 8
    max_inflight_requests: int = 8
    max_items_per_cell: int = 15          # base for auto-scaling: max(15, 15 * n_turns / 2)
    max_assistant_items_per_cell: int = 2
    max_topics_per_item: int = 3
    max_attribute_keys_per_item: int = 2
    max_domain_keys_per_item: int = 2
    max_collection_keys_per_item: int = 2
    fact_manager: FactManagerConfig = field(default_factory=FactManagerConfig)
