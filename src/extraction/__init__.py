"""Chunking and extraction pipeline for the vNext MemForest build path."""

from .chunker import chunk_session, normalize_turns
from .dedup import deduplicate_fact_texts, deduplicate_memory_items
from .manager import ExtractionManager
from .pipeline import ChunkExtractionPipeline, JsonExtractionBackend

__all__ = [
    "ChunkExtractionPipeline",
    "ExtractionManager",
    "ExtractionStep",
    "FactManager",
    "JsonExtractionBackend",
    "chunk_session",
    "deduplicate_fact_texts",
    "deduplicate_memory_items",
    "normalize_turns",
    "run_longmemeval_parallel",
]


def __getattr__(name: str):
    if name == "FactManager":
        from .fact_manager import FactManager

        return FactManager
    if name == "ExtractionStep":
        from .step import ExtractionStep

        return ExtractionStep
    if name == "run_longmemeval_parallel":
        from .runner import run_longmemeval_parallel

        return run_longmemeval_parallel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
