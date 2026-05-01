"""Top-level config loader for the vNext pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.config.api import APISettings, ChatCompletionConfig, EmbeddingConfig
from src.config.extraction_config import ExtractionConfig, FactManagerConfig
from src.config.logger_config import ApiLogConfig, ExtractionLogConfig, LoggerConfig
from src.config.query_config import BrowseConfig, PlannerConfig, QueryConfig, RecallConfig
from src.config.tree_config import (
    EntityTreeConfig,
    SceneRouterConfig,
    SessionTreeConfig,
    SummaryManagerConfig,
    TreeConfig,
)
from src.utils.time import set_default_timezone
from src.utils.types import ChunkingConfig


@dataclass(frozen=True)
class MemForestConfig:
    api: APISettings
    extraction: ExtractionConfig
    tree: TreeConfig = field(default_factory=TreeConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    default_timezone: str = "UTC"


def load_default_config(default_yaml_path: str | Path = "src/config/default.yaml") -> MemForestConfig:
    raw = yaml.safe_load(Path(default_yaml_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Default config root must be a mapping.")

    default_timezone = str(raw.get("default_timezone", "UTC") or "UTC")
    set_default_timezone(default_timezone)

    llm_global = (((raw.get("model") or {}).get("llm") or {}).get("global") or {})
    llm_steps = (((raw.get("model") or {}).get("llm") or {}).get("steps") or {})
    llm_step = (llm_steps.get("extraction") or llm_steps.get("atomic_fact_extraction") or {})
    embedding_global = (((raw.get("model") or {}).get("embedding") or {}).get("global") or {})
    extraction_raw = raw.get("extraction") or {}

    llm_cfg = ChatCompletionConfig(
        url=str(llm_global.get("url", "http://127.0.0.1:8001/v1")),
        model_name=_resolve_model_name(llm_global, llm_step),
        key=str(llm_global.get("key", "")),
        topk=_maybe_int(llm_global.get("topk")),
        top_p=float(llm_global.get("top_p", 1.0)),
        max_token=int(llm_step.get("max_token", llm_global.get("max_token", 1024))),
        temperature=float(llm_step.get("temperature", llm_global.get("temperature", 0.0))),
    )
    embedding_cfg = EmbeddingConfig(
        url=str(embedding_global.get("url", "http://127.0.0.1:8002/v1")),
        model_name=str(embedding_global.get("model_name", "")),
        key=str(embedding_global.get("key", "")),
        dimension=int(embedding_global.get("dimension", 1024)),
    )
    extraction_cfg = ExtractionConfig(
        chunking=ChunkingConfig(
            max_turns=int(extraction_raw.get("memcell_max_turns", extraction_raw.get("history_slide_window", 10)) or 10),
            max_chars=int(extraction_raw.get("memcell_max_chars", 4200) or 4200),
            max_time_gap_seconds=float(extraction_raw.get("memcell_max_time_gap_seconds", 6 * 3600) or 6 * 3600),
            hard_boundary_markers=tuple(
                str(x).strip().lower()
                for x in (extraction_raw.get("memcell_hard_boundary_markers") or ChunkingConfig().hard_boundary_markers)
                if str(x).strip()
            ),
        ),
        use_llm_extraction=True,
        max_tokens=int(llm_step.get("max_token", 4096)),
        temperature=float(llm_step.get("temperature", 0.0)),
        top_p=float(llm_global.get("top_p", 0.8)),
        max_workers=int(extraction_raw.get("max_workers", min(32, max(1, os.cpu_count() or 8))) or min(32, max(1, os.cpu_count() or 8))),
        max_inflight_requests=int(
            extraction_raw.get(
                "max_inflight_requests",
                extraction_raw.get("max_cell_workers_per_session", 8),
            )
            or 8
        ),
        max_items_per_cell=int(extraction_raw.get("max_items_per_cell", 15) or 15),
        max_assistant_items_per_cell=int(extraction_raw.get("max_assistant_items_per_cell", 2) or 2),
        max_topics_per_item=int(extraction_raw.get("max_topics_per_item", 3) or 3),
        max_attribute_keys_per_item=int(extraction_raw.get("max_attribute_keys_per_item", 2) or 2),
        max_domain_keys_per_item=int(extraction_raw.get("max_domain_keys_per_item", 2) or 2),
        max_collection_keys_per_item=int(extraction_raw.get("max_collection_keys_per_item", 2) or 2),
        fact_manager=FactManagerConfig(
            enabled=bool((extraction_raw.get("fact_manager") or {}).get("enabled", False)),
            storage_dir=str((extraction_raw.get("fact_manager") or {}).get("storage_dir", "data/index/vnext_fact_manager")),
            persist_on_write=bool((extraction_raw.get("fact_manager") or {}).get("persist_on_write", False)),
            top_k=int((extraction_raw.get("fact_manager") or {}).get("top_k", 8) or 8),
            similarity_threshold=float((extraction_raw.get("fact_manager") or {}).get("similarity_threshold", 0.972) or 0.972),
            max_llm_pairs_per_item=int((extraction_raw.get("fact_manager") or {}).get("max_llm_pairs_per_item", 4) or 4),
            normalize_embeddings=bool((extraction_raw.get("fact_manager") or {}).get("normalize_embeddings", True)),
        ),
    )
    tree_raw = raw.get("tree") or {}
    session_raw = tree_raw.get("session") or {}
    entity_raw = tree_raw.get("entity") or {}
    scene_raw_t = tree_raw.get("scene") or {}
    sm_raw = tree_raw.get("summary_manager") or {}
    tree_cfg = TreeConfig(
        storage_dir=str(tree_raw.get("storage_dir", "data/index/trees")),
        root_index_dir=str(tree_raw.get("root_index_dir", "data/index/root_index")),
        session=SessionTreeConfig(
            k=int(session_raw.get("k", 3)),
            summary_max_words=int(session_raw.get("summary_max_words", 200)),
        ),
        entity=EntityTreeConfig(
            user_k=int(entity_raw.get("user_k", 10)),
            entity_k=int(entity_raw.get("entity_k", 8)),
            active_min_facts=int(entity_raw.get("active_min_facts", 3)),
            active_min_sessions=int(entity_raw.get("active_min_sessions", 2)),
            suppress_below_facts=int(entity_raw.get("suppress_below_facts", 2)),
            max_user_tree_size=int(entity_raw.get("max_user_tree_size", 500)),
            max_entity_tree_size=int(entity_raw.get("max_entity_tree_size", 100)),
            summary_max_words=int(entity_raw.get("summary_max_words", 250)),
        ),
        scene=SceneRouterConfig(
            k=int(scene_raw_t.get("k", 20)),
            max_cluster_size=int(scene_raw_t.get("max_cluster_size", 200)),
            bootstrap_cluster_count=int(scene_raw_t.get("bootstrap_cluster_count", 12)),
            bootstrap_min_facts=int(scene_raw_t.get("bootstrap_min_facts", 24)),
            bootstrap_max_facts=int(scene_raw_t.get("bootstrap_max_facts", 200)),
            theta_assign=float(scene_raw_t.get("theta_assign", 0.65)),
            theta_second=float(scene_raw_t.get("theta_second", 0.55)),
            theta_spawn=float(scene_raw_t.get("theta_spawn", 0.50)),
            theta_merge=float(scene_raw_t.get("theta_merge", 0.88)),
            merge_check_interval=int(scene_raw_t.get("merge_check_interval", 500)),
            max_time_gap_days=float(scene_raw_t.get("max_time_gap_days", 45.0)),
            summary_max_words=int(scene_raw_t.get("summary_max_words", 200)),
        ),
        summary_manager=SummaryManagerConfig(
            max_inflight=int(sm_raw.get("max_inflight", 16)),
            temperature=float(sm_raw.get("temperature", 0.0)),
            max_tokens=int(sm_raw.get("max_tokens", 4096)),
            top_p=float(sm_raw.get("top_p", 0.8)),
            max_retries=int(sm_raw.get("max_retries", 2)),
            timeout_sec=float(sm_raw.get("timeout_sec", 60.0)),
        ),
    )

    logger_raw = raw.get("logger") or {}
    api_log_raw = logger_raw.get("api") or {}
    extraction_log_raw = logger_raw.get("extraction") or {}
    logger_cfg = LoggerConfig(
        api=ApiLogConfig(
            enabled=bool(api_log_raw.get("enabled", True)),
            path=str(api_log_raw.get("path", "data/logs/src_api_calls.jsonl")),
        ),
        extraction=ExtractionLogConfig(
            enabled=bool(extraction_log_raw.get("enabled", True)),
            path=str(extraction_log_raw.get("path", "data/logs/src_extraction_requests.jsonl")),
        ),
    )

    query_raw = raw.get("query") or {}
    recall_raw = query_raw.get("recall") or {}
    browse_raw = query_raw.get("browse") or {}
    planner_raw = query_raw.get("planner") or {}
    query_cfg = QueryConfig(
        mode=str(query_raw.get("mode", "agentic")),
        node_index_vector_dim=int(query_raw.get("node_index_vector_dim", 1024)),
        recall=RecallConfig(
            top_k=int(recall_raw.get("top_k", 10)),
            direction=str(recall_raw.get("direction", "union")),
            bu_top_m=int(recall_raw.get("bu_top_m", 30)),
            force_entity_user=bool(recall_raw.get("force_entity_user", True)),
            llm_rerank=bool(recall_raw.get("llm_rerank", False)),
            llm_rerank_overfetch=int(recall_raw.get("llm_rerank_overfetch", 20)),
        ),
        browse=BrowseConfig(
            beam_width=int(browse_raw.get("beam_width", 10)),
            max_facts=int(browse_raw.get("max_facts", 0)),
            llm_guided=bool(browse_raw.get("llm_guided", True)),
            llm_guided_all_types=bool(browse_raw.get("llm_guided_all_types", True)),
            max_workers=int(browse_raw.get("max_workers", 4)),
        ),
        planner=PlannerConfig(
            enabled=bool(planner_raw.get("enabled", True)),
            model_name=planner_raw.get("model_name") or None,
        ),
    )

    return MemForestConfig(
        api=APISettings(llm=llm_cfg, embedding=embedding_cfg),
        extraction=extraction_cfg,
        tree=tree_cfg,
        query=query_cfg,
        logger=logger_cfg,
        default_timezone=default_timezone,
    )


def _resolve_model_name(llm_global: dict[str, Any], llm_step: dict[str, Any]) -> str:
    step_name = llm_step.get("model_name", "global")
    if step_name == "global":
        return str(llm_global.get("model_name", ""))
    return str(step_name or llm_global.get("model_name", ""))


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
