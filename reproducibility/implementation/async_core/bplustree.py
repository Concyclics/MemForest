"""Async B+ tree implementation for timeline and fact indexing."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import Executor
from bisect import bisect_left, bisect_right
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from memory_forest.core.models import BPlusNodeMeta, ContentLeaf, KeyType
from memory_forest.core.summarizer import fallback_summary_from_children, summarize_parent_with_llm
from memory_forest.core.utils import unix_to_time_text


class AsyncRWLock:
    """A small asyncio read-write lock."""

    def __init__(self) -> None:
        self._cond = asyncio.Condition()
        self._readers = 0
        self._writer = False

    @asynccontextmanager
    async def read_lock(self):
        async with self._cond:
            while self._writer:
                await self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            async with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @asynccontextmanager
    async def write_lock(self):
        async with self._cond:
            while self._writer or self._readers > 0:
                await self._cond.wait()
            self._writer = True
        try:
            yield
        finally:
            async with self._cond:
                self._writer = False
                self._cond.notify_all()


class AsyncBPlusTree:
    """
    Concurrent-safe async B+ tree.

    Notes:
    - Internal nodes store separator keys where child index = bisect_right(keys, key).
    - Leaf nodes are doubly linked with `prev`/`next`.
    - Every leaf/internal node keeps `meta` (summary, roles, time_range), and parent meta
      is regenerated from child meta only.
    """

    def __init__(
        self,
        order: int = 4,
        leaf_capacity: Optional[int] = None,
        prompt_manager: Optional[Any] = None,
        summary_parallel_limit: int = 8,
        summary_prompt_key: str = "node_summarizer",
        summary_context: Optional[Dict[str, Any]] = None,
        summary_executor: Optional[Executor] = None,
    ) -> None:
        if order < 3:
            raise ValueError("B+ tree order must be >= 3.")
        self.order = order
        self.TreeNodes: Dict[str, Dict[str, Any]] = {}
        self.LeafNodes: Dict[str, Dict[str, Any]] = {}
        self.RootNode: Optional[str] = None
        self.NodeCounter = 0
        self.LeafCounter = 0
        self._rwlock = AsyncRWLock()
        self.prompt_manager = prompt_manager
        self.summary_parallel_limit = max(1, int(summary_parallel_limit))
        self.summary_prompt_key = str(summary_prompt_key or "node_summarizer")
        self.summary_context = dict(summary_context or {})
        self.summary_executor = summary_executor
        if leaf_capacity is None:
            self.leaf_capacity = int(self.order)
        else:
            self.leaf_capacity = max(1, int(leaf_capacity))
        self._dirty_leaf_ids: Set[str] = set()
        self._dirty_internal_ids: Set[str] = set()

    async def insert(self, key: Tuple[float, int], value: Any, defer_meta_update: bool = False) -> None:
        async with self._rwlock.write_lock():
            if self.RootNode is None:
                leaf_id = self._new_leaf()
                self.RootNode = leaf_id
                leaf = self.LeafNodes[leaf_id]
                leaf["keys"].append(key)
                leaf["values"].append(value)
                self._mark_leaf_and_ancestors_dirty(leaf_id)
                if not defer_meta_update:
                    await self._flush_meta_updates_locked()
                return

            leaf_id = self._find_leaf_id(key)
            leaf = self.LeafNodes[leaf_id]
            idx = bisect_left(leaf["keys"], key)
            leaf["keys"].insert(idx, key)
            leaf["values"].insert(idx, value)
            self._mark_leaf_and_ancestors_dirty(leaf_id)

            if len(leaf["keys"]) > self.leaf_capacity:
                self._split_leaf(leaf_id)
            if not defer_meta_update:
                await self._flush_meta_updates_locked()

    async def batch_insert(
        self,
        items: List[Tuple[Tuple[float, int], Any]],
        optimize_meta: bool = True,
        defer_meta_update: bool = False,
        timing: Optional[Dict[str, float]] = None,
    ) -> None:
        if not items:
            if isinstance(timing, dict):
                timing["lock_wait_sec"] = 0.0
                timing["exec_sec"] = 0.0
                timing["total_sec"] = 0.0
            return
        t_lock_wait = perf_counter()
        async with self._rwlock.write_lock():
            t_exec = perf_counter()
            lock_wait_sec = t_exec - t_lock_wait
            for key, value in sorted(items, key=lambda x: x[0]):
                if self.RootNode is None:
                    leaf_id = self._new_leaf()
                    self.RootNode = leaf_id
                    leaf = self.LeafNodes[leaf_id]
                    leaf["keys"].append(key)
                    leaf["values"].append(value)
                    self._mark_leaf_and_ancestors_dirty(leaf_id)
                    continue
                leaf_id = self._find_leaf_id(key)
                leaf = self.LeafNodes[leaf_id]
                idx = bisect_left(leaf["keys"], key)
                leaf["keys"].insert(idx, key)
                leaf["values"].insert(idx, value)
                self._mark_leaf_and_ancestors_dirty(leaf_id)
                if len(leaf["keys"]) > self.leaf_capacity:
                    self._split_leaf(leaf_id)
            if not defer_meta_update:
                if optimize_meta:
                    await self._flush_meta_updates_locked()
                else:
                    await self._rebuild_all_metas_batched()
            exec_sec = perf_counter() - t_exec
        total_sec = perf_counter() - t_lock_wait
        if isinstance(timing, dict):
            timing["lock_wait_sec"] = float(lock_wait_sec)
            timing["exec_sec"] = float(exec_sec)
            timing["total_sec"] = float(total_sec)

    async def range_query(
        self,
        start: Tuple[float, int],
        end: Tuple[float, int],
        limit: Optional[int] = None,
    ) -> List[Any]:
        if self.RootNode is None:
            return []
        async with self._rwlock.read_lock():
            results: List[Any] = []
            leaf_id = self._find_leaf_id(start)
            while leaf_id is not None:
                leaf = self.LeafNodes[leaf_id]
                for k, v in zip(leaf["keys"], leaf["values"]):
                    if k < start:
                        continue
                    if k > end:
                        return results
                    results.append(v)
                    if limit is not None and len(results) >= limit:
                        return results
                leaf_id = leaf["next"]
            return results

    async def find_nearest(self, key: Tuple[float, int]) -> Tuple[Any, int]:
        """
        Return nearest value and signed offset in leaf index:
        - 0 for exact match
        - negative if predecessor picked
        - positive if successor picked
        """

        if self.RootNode is None:
            raise ValueError("Tree is empty.")

        async with self._rwlock.read_lock():
            leaf_id = self._find_leaf_id(key)
            leaf = self.LeafNodes[leaf_id]
            idx = bisect_left(leaf["keys"], key)

            if idx < len(leaf["keys"]) and leaf["keys"][idx] == key:
                return leaf["values"][idx], 0

            left_candidate = self._left_candidate(leaf_id, idx)
            right_candidate = self._right_candidate(leaf_id, idx)
            if left_candidate is None and right_candidate is None:
                raise ValueError("Tree is empty.")
            if left_candidate is None:
                return right_candidate[1], 1
            if right_candidate is None:
                return left_candidate[1], -1

            left_dist = self._key_distance(left_candidate[0], key)
            right_dist = self._key_distance(right_candidate[0], key)
            if left_dist <= right_dist:
                return left_candidate[1], -1
            return right_candidate[1], 1

    async def move(self, leaf_ref: Any, steps: int) -> List[Any]:
        """
        Move from a leaf position and collect traversed values.

        Supported refs:
        - {"leaf_id": "...", "index": int}
        - ("leaf_id", index)
        - "leaf_id" (index defaults to 0)
        """

        if self.RootNode is None:
            return []

        async with self._rwlock.read_lock():
            leaf_id, idx = self._resolve_leaf_ref(leaf_ref)
            if leaf_id not in self.LeafNodes:
                raise KeyError(f"Unknown leaf: {leaf_id}")

            out: List[Any] = []
            direction = 1 if steps >= 0 else -1
            remaining = abs(steps)
            curr_leaf_id = leaf_id
            curr_idx = idx

            while remaining > 0 and curr_leaf_id is not None:
                leaf = self.LeafNodes[curr_leaf_id]
                curr_idx += direction
                while curr_leaf_id is not None and (curr_idx < 0 or curr_idx >= len(leaf["values"])):
                    if curr_idx < 0:
                        curr_leaf_id = leaf["prev"]
                        if curr_leaf_id is None:
                            break
                        leaf = self.LeafNodes[curr_leaf_id]
                        curr_idx = len(leaf["values"]) - 1
                    else:
                        curr_leaf_id = leaf["next"]
                        if curr_leaf_id is None:
                            break
                        leaf = self.LeafNodes[curr_leaf_id]
                        curr_idx = 0
                if curr_leaf_id is None:
                    break
                out.append(leaf["values"][curr_idx])
                remaining -= 1
            return out

    async def get_node_summary(self, node_ref: Any) -> BPlusNodeMeta:
        async with self._rwlock.read_lock():
            node_id = self._normalize_node_ref(node_ref)
            node = self._get_node(node_id)
            return node["meta"]

    async def get_node_time_range(self, node_ref: Any) -> Optional[Tuple[float, float]]:
        async with self._rwlock.read_lock():
            node_id = self._normalize_node_ref(node_ref)
            node = self._get_node(node_id)
            return node["meta"].time_range

    async def get_parent(self, node_ref: Any) -> Optional[str]:
        async with self._rwlock.read_lock():
            node_id = self._normalize_node_ref(node_ref)
            node = self._get_node(node_id)
            return node["parent"]

    async def get_children(self, node_ref: Any) -> List[str]:
        async with self._rwlock.read_lock():
            node_id = self._normalize_node_ref(node_ref)
            node = self._get_node(node_id)
            if node["is_leaf"]:
                return []
            return list(node["children"])

    async def get_leaf_neighbors(self, leaf_ref: Any) -> Tuple[Optional[str], Optional[str]]:
        async with self._rwlock.read_lock():
            leaf_id, _ = self._resolve_leaf_ref(leaf_ref)
            leaf = self.LeafNodes[leaf_id]
            return leaf["prev"], leaf["next"]

    async def get_root(self) -> Optional[str]:
        async with self._rwlock.read_lock():
            return self.RootNode

    async def get_leaf_count(self) -> int:
        async with self._rwlock.read_lock():
            return int(len(self.LeafNodes))

    async def expand(self, node_ref: Any) -> List[Any]:
        async with self._rwlock.read_lock():
            node_id = self._normalize_node_ref(node_ref)
            node = self._get_node(node_id)
            if node["is_leaf"]:
                return list(node["values"])
            return list(node["children"])

    async def update_meta_upwards(self, leaf_ref: Any) -> None:
        async with self._rwlock.write_lock():
            leaf_id, _ = self._resolve_leaf_ref(leaf_ref)
            if leaf_id not in self.LeafNodes:
                raise KeyError(f"Unknown leaf: {leaf_id}")
            self._mark_leaf_and_ancestors_dirty(leaf_id)
            await self._flush_meta_updates_locked()

    async def flush_meta_updates(self) -> None:
        async with self._rwlock.write_lock():
            await self._flush_meta_updates_locked()

    async def ensure_internal_root(self, *, defer_meta_update: bool = False) -> bool:
        """
        Ensure the tree root is an internal node (root -> leaf), even when the
        tree currently has only one leaf node.

        Returns True if tree structure was changed.
        """

        async with self._rwlock.write_lock():
            if self.RootNode is None or (not self._is_leaf_id(self.RootNode)):
                return False
            leaf_id = self.RootNode
            leaf = self.LeafNodes.get(leaf_id)
            if leaf is None:
                return False

            root_id = self._new_internal(level=max(int(leaf["meta"].level) + 1, 1))
            root = self.TreeNodes[root_id]
            root["keys"] = []
            root["children"] = [leaf_id]
            leaf["parent"] = root_id
            self.RootNode = root_id

            self._mark_leaf_and_ancestors_dirty(leaf_id)
            self._mark_internal_and_ancestors_dirty(root_id)
            if not defer_meta_update:
                await self._flush_meta_updates_locked()
            return True

    async def dump_structure(self) -> List[Dict[str, Any]]:
        async with self._rwlock.read_lock():
            if self.RootNode is None:
                return []
            out: List[Dict[str, Any]] = []
            stack = [self.RootNode]
            while stack:
                node_id = stack.pop(0)
                node = self._get_node(node_id)
                meta: BPlusNodeMeta = node["meta"]
                item_count = len(node["values"]) if node["is_leaf"] else len(node["children"])
                out.append(
                    {
                        "node_id": node_id,
                        "level": meta.level,
                        "is_leaf": bool(node["is_leaf"]),
                        "parent_id": node["parent"],
                        "child_ids": [] if node["is_leaf"] else list(node["children"]),
                        "time_range": meta.time_range,
                        "summary": meta.summary,
                        "item_count": item_count,
                    }
                )
                if not node["is_leaf"]:
                    stack.extend(list(node["children"]))
            return out

    async def dump_structure_text(self) -> str:
        async with self._rwlock.read_lock():
            if self.RootNode is None:
                return "<empty tree>"
            lines: List[str] = []
            queue: List[Tuple[str, int]] = [(self.RootNode, 0)]
            while queue:
                node_id, depth = queue.pop(0)
                node = self._get_node(node_id)
                meta: BPlusNodeMeta = node["meta"]
                prefix = "  " * depth
                lines.append(
                    f"{prefix}- {node_id} level={meta.level} time_range={meta.time_range} "
                    f"summary={meta.summary}"
                )
                if not node["is_leaf"]:
                    for child_id in node["children"]:
                        queue.append((child_id, depth + 1))
            return "\n".join(lines)

    async def export_state(self) -> Dict[str, Any]:
        async with self._rwlock.read_lock():
            return {
                "order": int(self.order),
                "RootNode": self.RootNode,
                "NodeCounter": int(self.NodeCounter),
                "LeafCounter": int(self.LeafCounter),
                "leaf_capacity": int(self.leaf_capacity),
                "summary_parallel_limit": int(self.summary_parallel_limit),
                "summary_prompt_key": self.summary_prompt_key,
                "summary_context": dict(self.summary_context),
                "TreeNodes": {
                    node_id: self._encode_internal_node(node)
                    for node_id, node in self.TreeNodes.items()
                },
                "LeafNodes": {
                    leaf_id: self._encode_leaf_node(node)
                    for leaf_id, node in self.LeafNodes.items()
                },
            }

    async def import_state(self, state: Mapping[str, Any], *, validate: bool = True) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("B+ tree state must be a mapping.")

        async with self._rwlock.write_lock():
            order = int(state.get("order", self.order))
            if order < 3:
                raise ValueError("B+ tree order must be >= 3.")

            tree_nodes_raw = state.get("TreeNodes", {})
            leaf_nodes_raw = state.get("LeafNodes", {})
            if not isinstance(tree_nodes_raw, Mapping) or not isinstance(leaf_nodes_raw, Mapping):
                raise ValueError("`TreeNodes` and `LeafNodes` must be mappings.")

            tree_nodes: Dict[str, Dict[str, Any]] = {}
            for node_id, raw in tree_nodes_raw.items():
                node_key = str(node_id)
                if not isinstance(raw, Mapping):
                    raise ValueError(f"Internal node `{node_key}` must be a mapping.")
                tree_nodes[node_key] = self._decode_internal_node(node_key, raw)

            leaf_nodes: Dict[str, Dict[str, Any]] = {}
            for leaf_id, raw in leaf_nodes_raw.items():
                leaf_key = str(leaf_id)
                if not isinstance(raw, Mapping):
                    raise ValueError(f"Leaf node `{leaf_key}` must be a mapping.")
                leaf_nodes[leaf_key] = self._decode_leaf_node(leaf_key, raw)

            root_node = state.get("RootNode")
            root_node = str(root_node) if isinstance(root_node, str) else None
            node_counter = int(state.get("NodeCounter", len(tree_nodes)))
            leaf_counter = int(state.get("LeafCounter", len(leaf_nodes)))
            leaf_capacity = int(state.get("leaf_capacity", self.leaf_capacity))
            summary_parallel_limit = int(
                state.get("summary_parallel_limit", self.summary_parallel_limit)
            )
            summary_prompt_key = str(state.get("summary_prompt_key", self.summary_prompt_key))
            summary_context = state.get("summary_context", self.summary_context)
            if not isinstance(summary_context, Mapping):
                raise ValueError("`summary_context` must be a mapping.")

            if validate:
                self._validate_imported_state(
                    root_node=root_node,
                    tree_nodes=tree_nodes,
                    leaf_nodes=leaf_nodes,
                )

            self.order = order
            self.RootNode = root_node
            self.NodeCounter = node_counter
            self.LeafCounter = leaf_counter
            self.leaf_capacity = max(1, leaf_capacity)
            self.summary_parallel_limit = max(1, summary_parallel_limit)
            self.summary_prompt_key = summary_prompt_key or "node_summarizer"
            self.summary_context = dict(summary_context)
            self.TreeNodes = tree_nodes
            self.LeafNodes = leaf_nodes
            self._dirty_leaf_ids.clear()
            self._dirty_internal_ids.clear()

    async def save(self, path: str) -> None:
        target = Path(path)
        state = await self.export_state()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    async def load(
        cls,
        path: str,
        *,
        prompt_manager: Optional[Any] = None,
        summary_parallel_limit: int = 8,
        summary_prompt_key: str = "node_summarizer",
        summary_context: Optional[Dict[str, Any]] = None,
        summary_executor: Optional[Executor] = None,
    ) -> "AsyncBPlusTree":
        source = Path(path)
        state = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(state, Mapping):
            raise ValueError("Tree snapshot must be a JSON object.")
        order = int(state.get("order", 4))
        leaf_capacity = int(state.get("leaf_capacity", order))
        tree = cls(
            order=order,
            leaf_capacity=leaf_capacity,
            prompt_manager=prompt_manager,
            summary_parallel_limit=summary_parallel_limit,
            summary_prompt_key=summary_prompt_key,
            summary_context=summary_context,
            summary_executor=summary_executor,
        )
        await tree.import_state(state, validate=True)
        return tree

    def _new_leaf(self) -> str:
        self.LeafCounter += 1
        leaf_id = f"leaf_{self.LeafCounter}"
        self.LeafNodes[leaf_id] = {
            "id": leaf_id,
            "is_leaf": True,
            "keys": [],
            "values": [],
            "parent": None,
            "prev": None,
            "next": None,
            "meta": BPlusNodeMeta(node_id=leaf_id, level=0, time_range=None, roles=set(), summary=""),
        }
        return leaf_id

    def _new_internal(self, level: int) -> str:
        self.NodeCounter += 1
        node_id = f"node_{self.NodeCounter}"
        self.TreeNodes[node_id] = {
            "id": node_id,
            "is_leaf": False,
            "keys": [],
            "children": [],
            "parent": None,
            "meta": BPlusNodeMeta(node_id=node_id, level=level, time_range=None, roles=set(), summary=""),
        }
        return node_id

    def _is_leaf_id(self, node_id: str) -> bool:
        return node_id.startswith("leaf_")

    def _get_node(self, node_id: str) -> Dict[str, Any]:
        if self._is_leaf_id(node_id):
            if node_id not in self.LeafNodes:
                raise KeyError(f"Unknown node id: {node_id}")
            return self.LeafNodes[node_id]
        if node_id not in self.TreeNodes:
            raise KeyError(f"Unknown node id: {node_id}")
        return self.TreeNodes[node_id]

    def _normalize_node_ref(self, node_ref: Any) -> str:
        if isinstance(node_ref, str):
            return node_ref
        if isinstance(node_ref, dict):
            node_id = node_ref.get("node_id") or node_ref.get("id")
            if isinstance(node_id, str):
                return node_id
        raise ValueError("node_ref must be node_id string or dict containing `node_id`/`id`.")

    def _resolve_leaf_ref(self, leaf_ref: Any) -> Tuple[str, int]:
        if isinstance(leaf_ref, str):
            return leaf_ref, 0
        if isinstance(leaf_ref, tuple) and len(leaf_ref) == 2:
            leaf_id, idx = leaf_ref
            return str(leaf_id), int(idx)
        if isinstance(leaf_ref, dict):
            leaf_id = leaf_ref.get("leaf_id") or leaf_ref.get("id")
            idx = leaf_ref.get("index", 0)
            if isinstance(leaf_id, str):
                return leaf_id, int(idx)
        raise ValueError("leaf_ref must be str, tuple(leaf_id,index), or dict with leaf_id.")

    def _find_leaf_id(self, key: KeyType) -> str:
        if self.RootNode is None:
            raise ValueError("Tree is empty.")
        node_id = self.RootNode
        while not self._is_leaf_id(node_id):
            node = self.TreeNodes[node_id]
            idx = bisect_right(node["keys"], key)
            node_id = node["children"][idx]
        return node_id

    def _split_leaf(self, leaf_id: str) -> None:
        leaf = self.LeafNodes[leaf_id]
        mid = len(leaf["keys"]) // 2
        right_id = self._new_leaf()
        right_leaf = self.LeafNodes[right_id]

        right_leaf["keys"] = leaf["keys"][mid:]
        right_leaf["values"] = leaf["values"][mid:]
        leaf["keys"] = leaf["keys"][:mid]
        leaf["values"] = leaf["values"][:mid]

        right_leaf["parent"] = leaf["parent"]
        right_leaf["next"] = leaf["next"]
        right_leaf["prev"] = leaf_id
        if leaf["next"] is not None:
            self.LeafNodes[leaf["next"]]["prev"] = right_id
        leaf["next"] = right_id

        separator = right_leaf["keys"][0]
        self._insert_in_parent(left_id=leaf_id, separator=separator, right_id=right_id)
        self._mark_leaf_and_ancestors_dirty(leaf_id)
        self._mark_leaf_and_ancestors_dirty(right_id)

    def _insert_in_parent(self, left_id: str, separator: KeyType, right_id: str) -> None:
        left = self._get_node(left_id)
        parent_id = left["parent"]

        if parent_id is None:
            root_id = self._new_internal(level=max(left["meta"].level + 1, 1))
            root = self.TreeNodes[root_id]
            root["keys"] = [separator]
            root["children"] = [left_id, right_id]
            left["parent"] = root_id
            self._get_node(right_id)["parent"] = root_id
            self.RootNode = root_id
            self._mark_internal_and_ancestors_dirty(root_id)
            return

        parent = self.TreeNodes[parent_id]
        child_idx = parent["children"].index(left_id)
        parent["keys"].insert(child_idx, separator)
        parent["children"].insert(child_idx + 1, right_id)
        self._get_node(right_id)["parent"] = parent_id
        self._mark_internal_and_ancestors_dirty(parent_id)

        if len(parent["keys"]) > self.order:
            self._split_internal(parent_id)

    def _split_internal(self, node_id: str) -> None:
        node = self.TreeNodes[node_id]
        mid = len(node["keys"]) // 2
        promote_key = node["keys"][mid]

        right_id = self._new_internal(level=node["meta"].level)
        right = self.TreeNodes[right_id]

        right["keys"] = node["keys"][mid + 1 :]
        right["children"] = node["children"][mid + 1 :]
        for child_id in right["children"]:
            self._get_node(child_id)["parent"] = right_id

        node["keys"] = node["keys"][:mid]
        node["children"] = node["children"][: mid + 1]

        right["parent"] = node["parent"]
        self._insert_in_parent(left_id=node_id, separator=promote_key, right_id=right_id)
        self._mark_internal_and_ancestors_dirty(node_id)
        self._mark_internal_and_ancestors_dirty(right_id)

    def _rebuild_leaf_meta(self, leaf_id: str) -> None:
        leaf = self.LeafNodes[leaf_id]
        time_range = self._keys_time_range(leaf["keys"])
        roles = self._roles_from_values(leaf["values"])
        summary = self._summary_from_leaf_values(leaf["values"])
        fact_ids = self._fact_ids_from_values(leaf["values"])
        numeric_markers = self._numeric_markers_from_values(leaf["values"])
        actor_markers = self._actor_markers_from_values(leaf["values"])
        scene_markers = self._scene_markers_from_values(leaf["values"])
        leaf["meta"] = BPlusNodeMeta(
            node_id=leaf_id,
            level=0,
            time_range=time_range,
            roles=roles,
            fact_ids=fact_ids,
            summary=summary,
            parent_id=leaf["parent"],
            child_ids=[],
            numeric_markers=numeric_markers,
            actor_markers=actor_markers,
            scene_markers=scene_markers,
        )

    async def _rebuild_internal_meta(self, node_id: str, semaphore: Optional[asyncio.Semaphore]) -> None:
        node = self.TreeNodes[node_id]
        child_metas = [self._get_node(child_id)["meta"] for child_id in node["children"]]

        ranges = [m.time_range for m in child_metas if m.time_range is not None]
        if ranges:
            low = min(r[0] for r in ranges)
            high = max(r[1] for r in ranges)
            time_range = (low, high)
        else:
            time_range = None

        roles: Set[str] = set()
        fact_ids: Set[str] = set()
        actor_markers: Set[str] = set()
        scene_markers: Set[str] = set()
        numeric_markers: List[float] = []
        for m in child_metas:
            roles.update(m.roles)
            fact_ids.update(m.fact_ids)
            actor_markers.update(set(getattr(m, "actor_markers", set()) or set()))
            scene_markers.update(set(getattr(m, "scene_markers", set()) or set()))
            for v in list(getattr(m, "numeric_markers", []) or []):
                try:
                    numeric_markers.append(float(v))
                except Exception:
                    continue
        numeric_markers = self._normalize_numeric_markers(numeric_markers)

        # For trees with a single leaf child, directly project one leaf fact
        # into root summary using explicit time+text format.
        if len(child_metas) == 1:
            only_child_id = node["children"][0] if node.get("children") else None
            summary = self._single_leaf_root_summary(only_child_id)
            if not summary:
                summary = str(child_metas[0].summary or "").strip() or fallback_summary_from_children(child_metas)
        else:
            summary = fallback_summary_from_children(child_metas)
            if self.prompt_manager is not None and child_metas:
                try:
                    summary = await summarize_parent_with_llm(
                        children_meta=child_metas,
                        prompt_manager=self.prompt_manager,
                        semaphore=semaphore,
                        prompt_key=self.summary_prompt_key,
                        extra_context=self.summary_context,
                        executor=self.summary_executor,
                    )
                except Exception:
                    summary = fallback_summary_from_children(child_metas)

        node["meta"] = BPlusNodeMeta(
            node_id=node_id,
            level=node["meta"].level,
            time_range=time_range,
            roles=roles,
            fact_ids=fact_ids,
            summary=summary,
            parent_id=node["parent"],
            child_ids=list(node["children"]),
            numeric_markers=numeric_markers,
            actor_markers=actor_markers,
            scene_markers=scene_markers,
        )

    async def _rebuild_all_metas_batched(self) -> None:
        for leaf_id in self.LeafNodes:
            self._rebuild_leaf_meta(leaf_id)

        if not self.TreeNodes:
            return

        levels: Dict[int, Set[str]] = defaultdict(set)
        for node_id, node in self.TreeNodes.items():
            levels[int(node["meta"].level)].add(node_id)

        semaphore = asyncio.Semaphore(self.summary_parallel_limit)
        for level in sorted(levels.keys()):
            tasks = [
                self._rebuild_internal_meta(node_id, semaphore)
                for node_id in sorted(levels[level])
            ]
            await asyncio.gather(*tasks)
        self._dirty_leaf_ids.clear()
        self._dirty_internal_ids.clear()

    async def _flush_meta_updates_locked(self) -> None:
        if not self._dirty_leaf_ids and not self._dirty_internal_ids:
            return

        dirty_leaf_ids = [leaf_id for leaf_id in self._dirty_leaf_ids if leaf_id in self.LeafNodes]
        for leaf_id in dirty_leaf_ids:
            self._rebuild_leaf_meta(leaf_id)

        dirty_internal_ids = [node_id for node_id in self._dirty_internal_ids if node_id in self.TreeNodes]
        if dirty_internal_ids:
            levels: Dict[int, Set[str]] = defaultdict(set)
            for node_id in dirty_internal_ids:
                level = int(self.TreeNodes[node_id]["meta"].level)
                levels[level].add(node_id)
            semaphore = asyncio.Semaphore(self.summary_parallel_limit)
            for level in sorted(levels.keys()):
                tasks = [
                    self._rebuild_internal_meta(node_id, semaphore)
                    for node_id in sorted(levels[level])
                ]
                await asyncio.gather(*tasks)

        self._dirty_leaf_ids.clear()
        self._dirty_internal_ids.clear()

    def _mark_leaf_and_ancestors_dirty(self, leaf_id: str) -> None:
        if leaf_id in self.LeafNodes:
            self._dirty_leaf_ids.add(leaf_id)
            parent_id = self.LeafNodes[leaf_id]["parent"]
            while parent_id is not None:
                self._dirty_internal_ids.add(parent_id)
                parent = self.TreeNodes.get(parent_id)
                if parent is None:
                    break
                parent_id = parent["parent"]

    def _mark_internal_and_ancestors_dirty(self, node_id: str) -> None:
        if node_id in self.TreeNodes:
            self._dirty_internal_ids.add(node_id)
            parent_id = self.TreeNodes[node_id]["parent"]
            while parent_id is not None:
                self._dirty_internal_ids.add(parent_id)
                parent = self.TreeNodes.get(parent_id)
                if parent is None:
                    break
                parent_id = parent["parent"]

    def _summary_from_leaf_values(self, values: List[Any]) -> str:
        if not values:
            return "Empty leaf"
        text = self._value_text(values[0])
        if not text:
            return "No preview text."
        return text

    def _single_leaf_root_summary(self, child_id: Optional[str]) -> str:
        if not child_id or child_id not in self.LeafNodes:
            return ""
        values = list(self.LeafNodes[child_id].get("values", []) or [])
        if not values:
            return ""

        value = values[0]
        text = self._value_root_text(value)
        if not text:
            return ""

        timestamp = self._value_timestamp(value)
        if timestamp is None:
            return text
        return f"on {unix_to_time_text(timestamp, 'UTC')}, {text}"

    def _value_root_text(self, value: Any) -> str:
        text = ""
        if hasattr(value, "fact_text"):
            text = str(getattr(value, "fact_text") or "").strip()
        elif hasattr(value, "text"):
            text = str(getattr(value, "text") or "").strip()
        elif isinstance(value, dict):
            if "fact_text" in value:
                text = str(value.get("fact_text") or "").strip()
            elif "text" in value:
                text = str(value.get("text") or "").strip()
        else:
            text = str(value).strip()

        if not text:
            return ""
        return self._sanitize_summary_text(text)

    def _value_timestamp(self, value: Any) -> Optional[float]:
        raw: Any = None
        if hasattr(value, "timestamp"):
            raw = getattr(value, "timestamp")
        elif isinstance(value, dict):
            raw = value.get("timestamp")

        if raw is None:
            return None
        try:
            return float(raw)
        except Exception:
            return None

    def _value_text(self, value: Any) -> str:
        if hasattr(value, "speaker") and hasattr(value, "text"):
            speaker = str(getattr(value, "speaker") or "").strip() or "speaker"
            listener = str(getattr(value, "listener_name", "") or "").strip() or self._infer_listener_name(speaker)
            txt = str(getattr(value, "text") or "").strip()
            return self._sanitize_summary_text(f"{speaker} to {listener}: {txt}")
        if hasattr(value, "fact_text"):
            fact_text = str(getattr(value, "fact_text") or "").strip()
            ts = self._value_timestamp(value)
            if ts is not None:
                return self._sanitize_summary_text(f"On {unix_to_time_text(ts, 'UTC')}, {fact_text}")
            return self._sanitize_summary_text(fact_text)
        if isinstance(value, dict):
            if "speaker" in value and "text" in value:
                speaker = str(value.get("speaker") or "").strip() or "speaker"
                listener = str(value.get("listener_name") or "").strip() or self._infer_listener_name(speaker)
                txt = str(value.get("text") or "").strip()
                return self._sanitize_summary_text(f"{speaker} to {listener}: {txt}")
            if "fact_text" in value:
                fact_text = str(value.get("fact_text") or "").strip()
                ts = self._value_timestamp(value)
                if ts is not None:
                    return self._sanitize_summary_text(f"On {unix_to_time_text(ts, 'UTC')}, {fact_text}")
                return self._sanitize_summary_text(fact_text)
            if "text" in value:
                return self._sanitize_summary_text(str(value["text"]))
        return self._sanitize_summary_text(str(value))

    def _infer_listener_name(self, speaker: str) -> str:
        sp = str(speaker or "").strip().lower()
        if sp == "user":
            return "assistant"
        if sp == "assistant":
            return "user"
        return "listener"

    def _sanitize_summary_text(self, text: str) -> str:
        return str(text or "").replace("|", " / ")

    def _roles_from_values(self, values: List[Any]) -> Set[str]:
        roles: Set[str] = set()
        for value in values:
            if hasattr(value, "speaker") and isinstance(getattr(value, "speaker"), str):
                roles.add(getattr(value, "speaker"))
            elif hasattr(value, "actors"):
                actors = getattr(value, "actors")
                if isinstance(actors, list):
                    for actor in actors:
                        if isinstance(actor, str):
                            roles.add(actor)
            elif isinstance(value, dict):
                speaker = value.get("speaker")
                if isinstance(speaker, str):
                    roles.add(speaker)
                actors = value.get("actors")
                if isinstance(actors, list):
                    for actor in actors:
                        if isinstance(actor, str):
                            roles.add(actor)
        return roles

    def _fact_ids_from_values(self, values: List[Any]) -> Set[str]:
        fact_ids: Set[str] = set()
        for value in values:
            if hasattr(value, "fact_ids"):
                raw = getattr(value, "fact_ids")
                if isinstance(raw, set):
                    for x in raw:
                        if isinstance(x, str):
                            fact_ids.add(x)
            elif isinstance(value, dict):
                raw = value.get("fact_ids")
                if isinstance(raw, (list, set, tuple)):
                    for x in raw:
                        if isinstance(x, str):
                            fact_ids.add(x)
        return fact_ids

    def _numeric_markers_from_values(self, values: List[Any]) -> List[float]:
        markers: List[float] = []
        for value in values:
            slots = None
            if hasattr(value, "slots_numeric"):
                slots = getattr(value, "slots_numeric")
            elif isinstance(value, dict):
                slots = value.get("slots_numeric")
            if not isinstance(slots, Mapping):
                continue
            for key in ("quantity", "delta", "value"):
                raw = slots.get(key)
                try:
                    if raw is None:
                        continue
                    markers.append(float(raw))
                except Exception:
                    continue
        return self._normalize_numeric_markers(markers)

    def _normalize_numeric_markers(self, values: List[float], max_items: int = 16) -> List[float]:
        out: List[float] = []
        seen = set()
        for val in values:
            try:
                key = round(float(val), 6)
            except Exception:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(float(val))
        out.sort(key=lambda x: (abs(float(x)), float(x)), reverse=True)
        return out[: max(1, int(max_items))]

    def _actor_markers_from_values(self, values: List[Any]) -> Set[str]:
        actors: Set[str] = set()
        for value in values:
            if hasattr(value, "actors_norm"):
                raw = getattr(value, "actors_norm")
                if isinstance(raw, list):
                    for item in raw:
                        key = str(item).strip().lower()
                        if key:
                            actors.add(key)
            elif hasattr(value, "actors"):
                raw = getattr(value, "actors")
                if isinstance(raw, list):
                    for item in raw:
                        key = str(item).strip().lower()
                        if key:
                            actors.add(key)
            elif isinstance(value, dict):
                for field in ("actors_norm", "actors"):
                    raw = value.get(field)
                    if not isinstance(raw, list):
                        continue
                    for item in raw:
                        key = str(item).strip().lower()
                        if key:
                            actors.add(key)
        return actors

    def _scene_markers_from_values(self, values: List[Any]) -> Set[str]:
        scenes: Set[str] = set()
        for value in values:
            row = ""
            if hasattr(value, "scene_norm"):
                row = str(getattr(value, "scene_norm", "")).strip().lower()
            elif hasattr(value, "scene"):
                row = str(getattr(value, "scene", "")).strip().lower()
            elif isinstance(value, dict):
                row = str(value.get("scene_norm", value.get("scene", ""))).strip().lower()
            if row:
                scenes.add(row)
        return scenes

    def _keys_time_range(self, keys: List[KeyType]) -> Optional[Tuple[float, float]]:
        if not keys:
            return None
        return (float(keys[0][0]), float(keys[-1][0]))

    def _left_candidate(self, leaf_id: str, idx: int) -> Optional[Tuple[KeyType, Any]]:
        leaf = self.LeafNodes[leaf_id]
        if idx - 1 >= 0:
            return leaf["keys"][idx - 1], leaf["values"][idx - 1]
        prev_id = leaf["prev"]
        if prev_id is None:
            return None
        prev_leaf = self.LeafNodes[prev_id]
        if not prev_leaf["keys"]:
            return None
        return prev_leaf["keys"][-1], prev_leaf["values"][-1]

    def _right_candidate(self, leaf_id: str, idx: int) -> Optional[Tuple[KeyType, Any]]:
        leaf = self.LeafNodes[leaf_id]
        if idx < len(leaf["keys"]):
            return leaf["keys"][idx], leaf["values"][idx]
        next_id = leaf["next"]
        if next_id is None:
            return None
        next_leaf = self.LeafNodes[next_id]
        if not next_leaf["keys"]:
            return None
        return next_leaf["keys"][0], next_leaf["values"][0]

    def _key_distance(self, a: KeyType, b: KeyType) -> int:
        # Timestamp dominates; turn-index acts as tie-breaker.
        ts_delta = abs(a[0] - b[0])
        if ts_delta > 0:
            return int(ts_delta * 1_000_000)
        return abs(a[1] - b[1])

    def _encode_key(self, key: KeyType) -> List[float]:
        return [float(key[0]), int(key[1])]

    def _decode_key(self, raw: Any) -> KeyType:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(f"Invalid key format: {raw!r}")
        return (float(raw[0]), int(raw[1]))

    def _encode_meta(self, meta: BPlusNodeMeta) -> Dict[str, Any]:
        return {
            "node_id": meta.node_id,
            "level": int(meta.level),
            "time_range": None
            if meta.time_range is None
            else [float(meta.time_range[0]), float(meta.time_range[1])],
            "roles": sorted(list(meta.roles)),
            "fact_ids": sorted(list(meta.fact_ids)),
            "summary": str(meta.summary),
            "parent_id": meta.parent_id,
            "child_ids": list(meta.child_ids),
            "numeric_markers": [float(x) for x in list(meta.numeric_markers or [])],
            "actor_markers": sorted(list(meta.actor_markers or set())),
            "scene_markers": sorted(list(meta.scene_markers or set())),
        }

    def _decode_meta(self, raw: Mapping[str, Any]) -> BPlusNodeMeta:
        time_range_raw = raw.get("time_range")
        time_range: Optional[Tuple[float, float]]
        if time_range_raw is None:
            time_range = None
        elif isinstance(time_range_raw, (list, tuple)) and len(time_range_raw) == 2:
            time_range = (float(time_range_raw[0]), float(time_range_raw[1]))
        else:
            raise ValueError(f"Invalid meta.time_range: {time_range_raw!r}")

        roles_raw = raw.get("roles", [])
        fact_ids_raw = raw.get("fact_ids", [])
        child_ids_raw = raw.get("child_ids", [])
        numeric_raw = raw.get("numeric_markers", [])
        actor_raw = raw.get("actor_markers", [])
        scene_raw = raw.get("scene_markers", [])
        numeric_markers: List[float] = []
        if isinstance(numeric_raw, list):
            for x in numeric_raw:
                try:
                    numeric_markers.append(float(x))
                except Exception:
                    continue
        return BPlusNodeMeta(
            node_id=str(raw.get("node_id", "")),
            level=int(raw.get("level", 0)),
            time_range=time_range,
            roles={str(x) for x in roles_raw if str(x).strip()},
            fact_ids={str(x) for x in fact_ids_raw if str(x).strip()},
            summary=str(raw.get("summary", "")),
            parent_id=(
                str(raw["parent_id"])
                if raw.get("parent_id") is not None
                else None
            ),
            child_ids=[str(x) for x in child_ids_raw],
            numeric_markers=self._normalize_numeric_markers(numeric_markers),
            actor_markers={str(x).strip().lower() for x in actor_raw if str(x).strip()},
            scene_markers={str(x).strip().lower() for x in scene_raw if str(x).strip()},
        )

    def _encode_value(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, ContentLeaf):
            data = asdict(value)
            data["fact_ids"] = sorted(list(value.fact_ids))
            return {"__type__": "ContentLeaf", "data": data}
        if isinstance(value, dict):
            return {"__type__": "dict", "data": value}
        raise TypeError(f"Unsupported leaf value type for snapshot: {type(value)}")

    def _decode_value(self, raw: Mapping[str, Any]) -> Any:
        kind = str(raw.get("__type__", "")).strip()
        data = raw.get("data")
        if kind == "ContentLeaf":
            if not isinstance(data, Mapping):
                raise ValueError("Invalid ContentLeaf payload.")
            return ContentLeaf(
                content_id=str(data.get("content_id", "")),
                session_id=str(data.get("session_id", "")),
                turn_index=int(data.get("turn_index", 0)),
                speaker=str(data.get("speaker", "")),
                timestamp=float(data.get("timestamp", 0.0)),
                text=str(data.get("text", "")),
                listener_name=str(data.get("listener_name", "")),
                parent_id=(
                    str(data["parent_id"])
                    if data.get("parent_id") is not None
                    else None
                ),
                fact_ids={str(x) for x in data.get("fact_ids", []) if str(x).strip()},
            )
        if kind == "dict":
            if not isinstance(data, Mapping):
                raise ValueError("Invalid dict leaf payload.")
            return dict(data)
        raise ValueError(f"Unsupported snapshot value type: {kind!r}")

    def _encode_internal_node(self, node: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(node["id"]),
            "is_leaf": False,
            "keys": [self._encode_key(k) for k in node["keys"]],
            "children": [str(x) for x in node["children"]],
            "parent": node["parent"],
            "meta": self._encode_meta(node["meta"]),
        }

    def _encode_leaf_node(self, node: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(node["id"]),
            "is_leaf": True,
            "keys": [self._encode_key(k) for k in node["keys"]],
            "values": [self._encode_value(v) for v in node["values"]],
            "parent": node["parent"],
            "prev": node["prev"],
            "next": node["next"],
            "meta": self._encode_meta(node["meta"]),
        }

    def _decode_internal_node(self, node_id: str, raw: Mapping[str, Any]) -> Dict[str, Any]:
        keys_raw = raw.get("keys", [])
        children_raw = raw.get("children", [])
        meta_raw = raw.get("meta", {})
        if not isinstance(keys_raw, list) or not isinstance(children_raw, list):
            raise ValueError(f"Invalid internal node `{node_id}` keys/children.")
        if not isinstance(meta_raw, Mapping):
            raise ValueError(f"Invalid internal node `{node_id}` meta.")
        return {
            "id": node_id,
            "is_leaf": False,
            "keys": [self._decode_key(x) for x in keys_raw],
            "children": [str(x) for x in children_raw],
            "parent": str(raw["parent"]) if raw.get("parent") is not None else None,
            "meta": self._decode_meta(meta_raw),
        }

    def _decode_leaf_node(self, leaf_id: str, raw: Mapping[str, Any]) -> Dict[str, Any]:
        keys_raw = raw.get("keys", [])
        values_raw = raw.get("values", [])
        meta_raw = raw.get("meta", {})
        if not isinstance(keys_raw, list) or not isinstance(values_raw, list):
            raise ValueError(f"Invalid leaf node `{leaf_id}` keys/values.")
        if len(keys_raw) != len(values_raw):
            raise ValueError(f"Leaf `{leaf_id}` keys/values length mismatch.")
        if not isinstance(meta_raw, Mapping):
            raise ValueError(f"Invalid leaf node `{leaf_id}` meta.")
        return {
            "id": leaf_id,
            "is_leaf": True,
            "keys": [self._decode_key(x) for x in keys_raw],
            "values": [self._decode_value(x) for x in values_raw],
            "parent": str(raw["parent"]) if raw.get("parent") is not None else None,
            "prev": str(raw["prev"]) if raw.get("prev") is not None else None,
            "next": str(raw["next"]) if raw.get("next") is not None else None,
            "meta": self._decode_meta(meta_raw),
        }

    def _validate_imported_state(
        self,
        *,
        root_node: Optional[str],
        tree_nodes: Mapping[str, Mapping[str, Any]],
        leaf_nodes: Mapping[str, Mapping[str, Any]],
    ) -> None:
        all_nodes: Dict[str, Mapping[str, Any]] = {}
        all_nodes.update(tree_nodes)
        all_nodes.update(leaf_nodes)

        if root_node is not None and root_node not in all_nodes:
            raise ValueError(f"Root node `{root_node}` not found in snapshot.")

        for node_id, node in tree_nodes.items():
            keys = node["keys"]
            if any(keys[i] > keys[i + 1] for i in range(len(keys) - 1)):
                raise ValueError(f"Internal node `{node_id}` keys are not sorted.")
            children = node["children"]
            for child_id in children:
                if child_id not in all_nodes:
                    raise ValueError(f"Node `{node_id}` references missing child `{child_id}`.")
                child = all_nodes[child_id]
                if child.get("parent") != node_id:
                    raise ValueError(
                        f"Child `{child_id}` parent mismatch: "
                        f"{child.get('parent')} != {node_id}"
                    )
            parent_id = node.get("parent")
            if parent_id is not None and parent_id not in tree_nodes:
                raise ValueError(f"Internal node `{node_id}` has invalid parent `{parent_id}`.")

        for leaf_id, leaf in leaf_nodes.items():
            keys = leaf["keys"]
            if any(keys[i] > keys[i + 1] for i in range(len(keys) - 1)):
                raise ValueError(f"Leaf node `{leaf_id}` keys are not sorted.")

            parent_id = leaf.get("parent")
            if parent_id is not None:
                if parent_id not in tree_nodes:
                    raise ValueError(f"Leaf `{leaf_id}` has invalid parent `{parent_id}`.")
                parent = tree_nodes[parent_id]
                if leaf_id not in parent["children"]:
                    raise ValueError(f"Leaf `{leaf_id}` missing from parent `{parent_id}` children.")

            prev_id = leaf.get("prev")
            next_id = leaf.get("next")
            if prev_id is not None:
                if prev_id not in leaf_nodes:
                    raise ValueError(f"Leaf `{leaf_id}` has invalid prev `{prev_id}`.")
                if leaf_nodes[prev_id].get("next") != leaf_id:
                    raise ValueError(f"Leaf link mismatch: `{prev_id}` next is not `{leaf_id}`.")
            if next_id is not None:
                if next_id not in leaf_nodes:
                    raise ValueError(f"Leaf `{leaf_id}` has invalid next `{next_id}`.")
                if leaf_nodes[next_id].get("prev") != leaf_id:
                    raise ValueError(f"Leaf link mismatch: `{next_id}` prev is not `{leaf_id}`.")

        if root_node is not None:
            root = all_nodes[root_node]
            if root.get("parent") is not None:
                raise ValueError(f"Root node `{root_node}` parent must be None.")
