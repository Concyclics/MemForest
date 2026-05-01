"""Prompt registry and prompt templates for vNext."""

from .extraction_prompt import (
    UNIFIED_MEMORY_EXTRACTION_PROMPT_NAME,
    build_extraction_prompt_manager,
)
from .dedup_prompt import FACT_EQUIVALENCE_PROMPT_NAME, build_dedup_prompt_manager
from .prompt import PromptManager, PromptTemplate

__all__ = [
    "FACT_EQUIVALENCE_PROMPT_NAME",
    "PromptManager",
    "PromptTemplate",
    "UNIFIED_MEMORY_EXTRACTION_PROMPT_NAME",
    "build_dedup_prompt_manager",
    "build_extraction_prompt_manager",
]
