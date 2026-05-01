"""Top-level extraction step wired with config, prompts, and OpenAI-compatible APIs."""

from __future__ import annotations

from typing import Any

from src.api import OpenAIChatClient, OpenAIJsonBackend
from src.config import MemForestConfig, load_default_config
from src.extraction.fact_manager import FactManager
from src.extraction.manager import ExtractionManager
from src.extraction.pipeline import ChunkExtractionPipeline, JsonExtractionBackend
from src.logger import ApiCallLogger, ExtractionLogger
from src.prompt import build_extraction_prompt_manager
from src.utils.types import ExtractionRequest


class ExtractionStep:
    """End-to-end session extraction entrypoint for the vNext pipeline."""

    def __init__(
        self,
        *,
        config: MemForestConfig,
        backend: JsonExtractionBackend | None = None,
        fact_manager: FactManager | None = None,
    ) -> None:
        self.config = config
        self.prompt_manager = build_extraction_prompt_manager()
        self.api_logger = ApiCallLogger(
            config.logger.api.path,
            enabled=config.logger.api.enabled,
        )
        self.extraction_logger = ExtractionLogger(
            config.logger.extraction.path,
            enabled=config.logger.extraction.enabled,
        )
        chat_client: OpenAIChatClient | None = None
        if backend is None and config.extraction.use_llm_extraction:
            chat_client = OpenAIChatClient(config.api.llm, api_logger=self.api_logger)
            backend = OpenAIJsonBackend(
                chat_client,
                model_name=config.api.llm.model_name,
                temperature=config.extraction.temperature,
                max_tokens=config.extraction.max_tokens,
                top_p=config.extraction.top_p,
            )
        self.backend = backend
        self.pipeline = ChunkExtractionPipeline(
            backend=backend,
            chunking=config.extraction.chunking,
            prompt_manager=self.prompt_manager,
            max_items_per_cell=config.extraction.max_items_per_cell,
            max_assistant_items_per_cell=config.extraction.max_assistant_items_per_cell,
            max_topics_per_item=config.extraction.max_topics_per_item,
            max_attribute_keys_per_item=config.extraction.max_attribute_keys_per_item,
            max_domain_keys_per_item=config.extraction.max_domain_keys_per_item,
            max_collection_keys_per_item=config.extraction.max_collection_keys_per_item,
        )
        self.manager = ExtractionManager(
            pipeline=self.pipeline,
            max_inflight_requests=config.extraction.max_inflight_requests,
            extraction_logger=self.extraction_logger,
        )
        if fact_manager is None and config.extraction.fact_manager.enabled:
            fact_manager = FactManager.from_config(
                config,
                chat_client=chat_client,
            )
        self.fact_manager = fact_manager
        self.last_fact_write_result = None

    @classmethod
    def from_defaults(cls, default_yaml_path: str = "src/config/default.yaml") -> "ExtractionStep":
        return cls(config=load_default_config(default_yaml_path))

    def run(self, session_id: str, turns: list[dict[str, Any]], *, write_facts: bool = False):
        result = self.manager.extract_request(
            ExtractionRequest(session_id=session_id, turns=turns)
        )
        self.last_fact_write_result = None
        if write_facts and self.fact_manager is not None and self.config.extraction.fact_manager.enabled:
            self.last_fact_write_result = self.fact_manager.add_session_result(
                result,
                persist=self.config.extraction.fact_manager.persist_on_write,
            )
        return result
