"""Tree-routing, semantic-overlay, and tree-index helpers for the vNext build path."""

from .routing import (
    OverlayUpdates,
    RoutedTreeBatch,
    RoutedTreeBatches,
    materialize_sparse_state_trees,
    route_memory_items,
    update_component_overlay,
    update_preference_overlay,
    update_state_overlay,
)
from .tree_types import (
    MemTree,
    MemTreeNode,
    SceneCluster,
    SessionLeaf,
    SummaryRequest,
    SummaryResult,
    TreeBuildResult,
    TreeCard,
)
from .tree import (
    build_tree_from_cells,
    build_tree_from_facts,
    clear_dirty_queues,
    collect_dirty_requests,
    delete_cell,
    delete_fact,
    fill_upper_level_inputs,
    insert_cell,
    insert_fact,
    rebuild_dirty_queues_from_flags,
    validate_tree_structure,
)
from .scene_router import SceneRouter
from .entity_router import EntityRouter
from .summary_manager import SummaryManager
from .tree_builder import TreeBuilder
from .tree_store import TreeStore
from .root_index import RootIndex
from .node_index import NodeIndex, NodeEntry

__all__ = [
    # routing
    "OverlayUpdates",
    "RoutedTreeBatch",
    "RoutedTreeBatches",
    "materialize_sparse_state_trees",
    "route_memory_items",
    "update_component_overlay",
    "update_preference_overlay",
    "update_state_overlay",
    # tree types
    "MemTree",
    "MemTreeNode",
    "SceneCluster",
    "SessionLeaf",
    "SummaryRequest",
    "SummaryResult",
    "TreeBuildResult",
    "TreeCard",
    # tree logic
    "build_tree_from_cells",
    "build_tree_from_facts",
    "clear_dirty_queues",
    "collect_dirty_requests",
    "delete_cell",
    "delete_fact",
    "fill_upper_level_inputs",
    "insert_cell",
    "insert_fact",
    "rebuild_dirty_queues_from_flags",
    "validate_tree_structure",
    # components
    "EntityRouter",
    "NodeEntry",
    "NodeIndex",
    "RootIndex",
    "SceneRouter",
    "SummaryManager",
    "TreeBuilder",
    "TreeStore",
]
