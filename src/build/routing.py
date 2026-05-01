"""Route extracted memory items into primary trees and semantic overlays."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict

from src.utils import normalize_entity_key
from src.utils.types import MemoryItem


@dataclass(frozen=True)
class RoutedTreeBatch:
    """One tree-specific batch of items ready for insertion."""

    tree_type: str
    tree_key: str
    items: list[MemoryItem]


@dataclass(frozen=True)
class RoutedTreeBatches:
    """Grouped primary-tree insert batches."""

    session: list[RoutedTreeBatch] = field(default_factory=list)
    actor: list[RoutedTreeBatch] = field(default_factory=list)
    scene: list[RoutedTreeBatch] = field(default_factory=list)


@dataclass(frozen=True)
class OverlayUpdates:
    """Semantic overlay posting lists derived from extracted items."""

    state: dict[str, list[str]] = field(default_factory=dict)
    preference: dict[str, list[str]] = field(default_factory=dict)
    component: dict[str, list[str]] = field(default_factory=dict)


def route_memory_items(items: list[MemoryItem]) -> tuple[RoutedTreeBatches, OverlayUpdates]:
    """Route items into primary trees plus overlay posting lists."""

    session_buckets: DefaultDict[str, list[MemoryItem]] = defaultdict(list)
    actor_buckets: DefaultDict[str, list[MemoryItem]] = defaultdict(list)
    scene_buckets: DefaultDict[str, list[MemoryItem]] = defaultdict(list)

    state_overlay: DefaultDict[str, list[str]] = defaultdict(list)
    preference_overlay: DefaultDict[str, list[str]] = defaultdict(list)
    component_overlay: DefaultDict[str, list[str]] = defaultdict(list)

    for item in items:
        session_buckets[str(item.session_id)].append(item)

        for entity in item.entities:
            entity_key = normalize_entity_key(entity)
            if entity_key:
                actor_buckets[entity_key].append(item)

        scene_keys = _ordered_unique([*item.topics, *item.domain_keys])
        for scene_key in scene_keys:
            scene_buckets[scene_key].append(item)

        update_state_overlay(state_overlay, item)
        update_preference_overlay(preference_overlay, item)
        update_component_overlay(component_overlay, item)

    batches = RoutedTreeBatches(
        session=_build_batches("session", session_buckets),
        actor=_build_batches("actor", actor_buckets),
        scene=_build_batches("scene", scene_buckets),
    )
    overlays = OverlayUpdates(
        state={key: list(value) for key, value in state_overlay.items()},
        preference={key: list(value) for key, value in preference_overlay.items()},
        component={key: list(value) for key, value in component_overlay.items()},
    )
    return batches, overlays


def update_state_overlay(state_overlay: DefaultDict[str, list[str]] | dict[str, list[str]], item: MemoryItem) -> None:
    """Add one item into the state overlay if it exposes state-like keys."""

    if not item.attribute_keys:
        return
    entity_keys = [normalize_entity_key(entity) for entity in item.entities]
    entity_keys = [key for key in entity_keys if key]
    if not entity_keys:
        return
    for entity_key in entity_keys:
        for attribute_key in item.attribute_keys:
            _append_overlay_item(state_overlay, f"{entity_key}|{attribute_key}", item.item_id)


def update_preference_overlay(
    preference_overlay: DefaultDict[str, list[str]] | dict[str, list[str]],
    item: MemoryItem,
) -> None:
    """Add one item into the preference overlay when it carries preference semantics."""

    if item.semantic_role not in {"preference", "constraint"} and not item.domain_keys:
        return
    entity_keys = [normalize_entity_key(entity) for entity in item.entities]
    entity_keys = [key for key in entity_keys if key]
    if not entity_keys:
        return
    for entity_key in entity_keys:
        for domain_key in item.domain_keys or ["general"]:
            _append_overlay_item(preference_overlay, f"{entity_key}|{domain_key}", item.item_id)


def update_component_overlay(
    component_overlay: DefaultDict[str, list[str]] | dict[str, list[str]],
    item: MemoryItem,
) -> None:
    """Add one item into the component overlay when it carries grouping keys."""

    for collection_key in item.collection_keys:
        _append_overlay_item(component_overlay, collection_key, item.item_id)


def materialize_sparse_state_trees(
    items: list[MemoryItem],
    state_overlay: dict[str, list[str]],
    *,
    min_items: int = 2,
    min_sessions: int = 2,
) -> dict[str, list[MemoryItem]]:
    """Return sparse state-tree candidates for dense cross-session state chains."""

    items_by_id = {item.item_id: item for item in items}
    sparse_trees: dict[str, list[MemoryItem]] = {}
    for state_key, item_ids in state_overlay.items():
        ordered_items = [items_by_id[item_id] for item_id in item_ids if item_id in items_by_id]
        if len(ordered_items) < int(min_items):
            continue
        session_count = len({item.session_id for item in ordered_items})
        if session_count < int(min_sessions):
            continue
        sparse_trees[state_key] = sorted(
            ordered_items,
            key=lambda item: (
                float(item.time_start) if item.time_start is not None else float("inf"),
                item.item_id,
            ),
        )
    return sparse_trees


def _append_overlay_item(
    overlay: DefaultDict[str, list[str]] | dict[str, list[str]],
    key: str,
    item_id: str,
) -> None:
    key = str(key).strip()
    item_id = str(item_id).strip()
    if not key or not item_id:
        return
    bucket = overlay.setdefault(key, [])
    if item_id not in bucket:
        bucket.append(item_id)


def _build_batches(tree_type: str, buckets: dict[str, list[MemoryItem]]) -> list[RoutedTreeBatch]:
    out: list[RoutedTreeBatch] = []
    for tree_key in sorted(buckets):
        out.append(RoutedTreeBatch(tree_type=tree_type, tree_key=tree_key, items=list(buckets[tree_key])))
    return out


def _ordered_unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value).strip().lower().split())
        if not text:
            continue
        text = text.replace(" ", "_")
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
