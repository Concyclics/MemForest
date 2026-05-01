"""Tree-layer configuration for the vNext MemoryForest build pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SessionTreeConfig:
    """Configuration for session tree construction."""

    k: int = 3
    summary_max_words: int = 200


@dataclass(frozen=True)
class EntityTreeConfig:
    """Configuration for lifecycle-based entity tree construction."""

    user_k: int = 10
    entity_k: int = 8
    active_min_facts: int = 3
    active_min_sessions: int = 2
    suppress_below_facts: int = 2
    max_user_tree_size: int = 500
    max_entity_tree_size: int = 100
    summary_max_words: int = 250


@dataclass(frozen=True)
class SceneRouterConfig:
    """Configuration for data-driven scene routing and cluster management."""

    k: int = 10
    max_cluster_size: int = 200
    bootstrap_cluster_count: int = 12
    bootstrap_min_facts: int = 24
    bootstrap_max_facts: int = 200
    theta_assign: float = 0.65       # primary assignment threshold
    theta_second: float = 0.55       # secondary (dual) assignment threshold
    theta_spawn: float = 0.50        # threshold below which a new cluster is created
    theta_merge: float = 0.88        # centroid similarity above which clusters merge
    merge_check_interval: int = 500  # run merge_check every N inserts
    max_time_gap_days: float = 45.0  # soft continuity horizon for routing bonus
    summary_max_words: int = 200


@dataclass(frozen=True)
class SummaryManagerConfig:
    """Configuration for the parallel SummaryManager."""

    max_inflight: int = 128
    temperature: float = 0.0
    max_tokens: int = 4096
    top_p: float = 0.8
    max_retries: int = 2          # retries on silent empty (error=None, summary="")
    timeout_sec: float = 60.0     # per-call HTTP timeout in seconds


@dataclass(frozen=True)
class TreeConfig:
    """Top-level tree configuration aggregating all sub-configs."""

    session: SessionTreeConfig = field(default_factory=SessionTreeConfig)
    entity: EntityTreeConfig = field(default_factory=EntityTreeConfig)
    scene: SceneRouterConfig = field(default_factory=SceneRouterConfig)
    summary_manager: SummaryManagerConfig = field(default_factory=SummaryManagerConfig)
    storage_dir: str = "data/index/trees"
    root_index_dir: str = "data/index/root_index"
