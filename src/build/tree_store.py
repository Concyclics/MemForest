"""TreeStore: persist, load, and browse MemTree objects.

Provides structured access to trees stored on disk, including
time-filtered browsing and session cell context windows.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.build.tree import validate_tree_structure
from src.build.tree_types import MemTree, MemTreeNode, SessionLeaf, TreeCard

if TYPE_CHECKING:
    from src.utils.types import ManagedFact


class TreeStore:
    """Load, save, and browse MemTree objects.

    Storage layout:
        {storage_dir}/
            {tree_type}/
                {tree_id}.json
    """

    def __init__(self, storage_dir: Path) -> None:
        self._dir = Path(storage_dir)
        self._trees: dict[str, MemTree] = {}
        self._lock = threading.RLock()

    # ── persistence ───────────────────────────────────────────────────────────

    def save_tree(self, tree: MemTree) -> None:
        """Write tree to {storage_dir}/{tree_type}/{tree_id}.json."""
        tree_dir = self._dir / tree.tree_type
        tree_dir.mkdir(parents=True, exist_ok=True)
        safe_id = tree.tree_id.replace(":", "_").replace("/", "_")
        path = tree_dir / f"{safe_id}.json"
        data = _tree_to_dict(tree)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_tree(self, tree_id: str) -> MemTree | None:
        """Load tree from disk. Cache in _trees. Returns None if not found."""
        with self._lock:
            if tree_id in self._trees:
                return self._trees[tree_id]
        # Search all type subdirs
        safe_id = tree_id.replace(":", "_").replace("/", "_")
        for subdir in self._dir.iterdir() if self._dir.exists() else []:
            if not subdir.is_dir():
                continue
            path = subdir / f"{safe_id}.json"
            if path.exists():
                tree = _tree_from_dict(json.loads(path.read_text(encoding="utf-8")))
                with self._lock:
                    self._trees[tree_id] = tree
                return tree
        return None

    def load_all(self) -> None:
        """Load all trees from disk into the in-memory cache."""
        if not self._dir.exists():
            return
        for subdir in self._dir.iterdir():
            if not subdir.is_dir():
                continue
            for json_path in subdir.glob("*.json"):
                try:
                    data = json.loads(json_path.read_text(encoding="utf-8"))
                    tree = _tree_from_dict(data)
                except Exception as exc:
                    raise ValueError(f"failed to load tree file {json_path}: {exc}") from exc
                with self._lock:
                    self._trees[tree.tree_id] = tree

    def register(self, tree: MemTree) -> None:
        """Register an in-memory tree (built but not yet persisted)."""
        with self._lock:
            self._trees[tree.tree_id] = tree

    def get(self, tree_id: str) -> MemTree | None:
        with self._lock:
            return self._trees.get(tree_id)

    def all_tree_ids(self) -> list[str]:
        with self._lock:
            return list(self._trees.keys())

    # ── tree card ─────────────────────────────────────────────────────────────

    def get_tree_card(self, tree_id: str) -> TreeCard | None:
        """Return a lightweight descriptor from the loaded tree."""
        with self._lock:
            tree = self._trees.get(tree_id)
        if tree is None:
            tree = self.load_tree(tree_id)
        if tree is None:
            return None
        root = tree.nodes.get(tree.root_node_id)
        return TreeCard(
            tree_id=tree.tree_id,
            tree_type=tree.tree_type,
            label=tree.label,
            time_start=root.time_start if root else 0.0,
            time_end=root.time_end if root else 0.0,
            root_summary=root.summary if root else "",
            item_count=root.item_count if root else 0,
        )

    # ── browsing ──────────────────────────────────────────────────────────────

    def browse_tree(
        self,
        tree_id: str,
        *,
        time_filter: tuple[float, float] | None = None,
        max_items: int = 20,
        fact_loader: "dict[str, ManagedFact] | None" = None,
    ) -> list[SessionLeaf | Any]:
        """Top-down traversal returning leaves (SessionLeaf or fact objects).

        For session trees: returns SessionLeaf objects.
        For entity/scene trees: returns ManagedFact objects via fact_loader,
          or fact_id strings if fact_loader is None.

        Respects tree.deactivated — redirects to replacement_tree_ids.
        """
        with self._lock:
            tree = self._trees.get(tree_id)
        if tree is None:
            tree = self.load_tree(tree_id)
        if tree is None:
            return []

        if tree.deactivated:
            results = []
            for replacement_id in tree.replacement_tree_ids:
                results.extend(
                    self.browse_tree(
                        replacement_id,
                        time_filter=time_filter,
                        max_items=max_items,
                        fact_loader=fact_loader,
                    )
                )
            return results[:max_items]

        root = tree.nodes.get(tree.root_node_id)
        if root is None:
            return []

        collected: list[str] = []
        _traverse(tree, root, time_filter=time_filter, collected=collected, max_items=max_items)

        if tree.tree_type == "session":
            return [
                tree.session_leaves[cid]
                for cid in collected
                if cid in tree.session_leaves
            ][:max_items]
        else:
            if fact_loader:
                return [
                    fact_loader[fid]
                    for fid in collected
                    if fid in fact_loader
                ][:max_items]
            return collected[:max_items]

    # ── cell context ──────────────────────────────────────────────────────────

    def get_cell(self, cell_id: str) -> SessionLeaf | None:
        """Back-reference: return a SessionLeaf by cell_id."""
        with self._lock:
            for tree in self._trees.values():
                if cell_id in tree.session_leaves:
                    return tree.session_leaves[cell_id]
        return None

    def get_cell_context(self, cell_id: str, *, window: int = 1) -> list[SessionLeaf]:
        """Return cell plus ±window adjacent cells in the same session."""
        leaf = self.get_cell(cell_id)
        if leaf is None:
            return []
        tree_id = f"session:{leaf.session_id}"
        with self._lock:
            tree = self._trees.get(tree_id)
        if tree is None:
            return [leaf]
        # Collect all leaves sorted by cell_index
        all_leaves = sorted(
            tree.session_leaves.values(), key=lambda l: l.cell_index
        )
        idx_map = {l.cell_id: i for i, l in enumerate(all_leaves)}
        pos = idx_map.get(cell_id)
        if pos is None:
            return [leaf]
        start = max(0, pos - window)
        end = min(len(all_leaves), pos + window + 1)
        return all_leaves[start:end]


# ── tree traversal ─────────────────────────────────────────────────────────────

def _traverse(
    tree: MemTree,
    node: MemTreeNode,
    *,
    time_filter: tuple[float, float] | None,
    collected: list[str],
    max_items: int,
) -> None:
    """DFS traversal collecting leaf child_ids that fall within the time filter."""
    if len(collected) >= max_items:
        return

    if time_filter is not None:
        t_lo, t_hi = time_filter
        if node.time_end < t_lo or node.time_start > t_hi:
            return  # node entirely outside time window

    if node.level == 0:
        for child_id in node.child_ids:
            if len(collected) >= max_items:
                break
            collected.append(child_id)
        return

    for child_id in node.child_ids:
        if len(collected) >= max_items:
            break
        child_node = tree.nodes.get(child_id)
        if child_node is not None:
            _traverse(tree, child_node, time_filter=time_filter,
                      collected=collected, max_items=max_items)
        else:
            # child_id is a leaf-level reference (session cell or fact)
            collected.append(child_id)


# ── serialisation ──────────────────────────────────────────────────────────────

def _tree_to_dict(tree: MemTree) -> dict:
    validate_tree_structure(tree)
    return {
        "tree_id": tree.tree_id,
        "tree_type": tree.tree_type,
        "tree_key": tree.tree_key,
        "label": tree.label,
        "k": tree.k,
        "root_node_id": tree.root_node_id,
        "nodes": {nid: _node_to_dict(n) for nid, n in tree.nodes.items()},
        "session_leaves": {
            cid: _leaf_to_dict(l) for cid, l in tree.session_leaves.items()
        },
        "fact_ids_ordered": tree.fact_ids_ordered,
        "centroid": tree.centroid,
        "version": tree.version,
        "lifecycle_state": tree.lifecycle_state,
        "deactivated": tree.deactivated,
        "replacement_tree_ids": tree.replacement_tree_ids,
        "created_at": tree.created_at,
        "updated_at": tree.updated_at,
        "dirty_l0_node_ids": sorted(tree.dirty_l0_node_ids),
        "dirty_internal_node_ids_by_level": {
            str(level): sorted(node_ids)
            for level, node_ids in sorted(tree.dirty_internal_node_ids_by_level.items())
        },
    }


def _node_to_dict(n: MemTreeNode) -> dict:
    return {
        "node_id": n.node_id,
        "tree_id": n.tree_id,
        "level": n.level,
        "time_start": n.time_start,
        "time_end": n.time_end,
        "summary": n.summary,
        "summary_dirty": n.summary_dirty,
        "child_ids": n.child_ids,
        "item_count": n.item_count,
        "parent_id": n.parent_id,
        "prev_leaf_id": n.prev_leaf_id,
        "next_leaf_id": n.next_leaf_id,
    }


def _leaf_to_dict(l: SessionLeaf) -> dict:
    return {
        "cell_id": l.cell_id,
        "session_id": l.session_id,
        "cell_index": l.cell_index,
        "time_start": l.time_start,
        "time_end": l.time_end,
        "raw_turns": l.raw_turns,
        "raw_text": l.raw_text,
        "fact_ids": l.fact_ids,
    }


def _tree_from_dict(d: dict) -> MemTree:
    nodes = {
        nid: _node_from_dict(n) for nid, n in (d.get("nodes") or {}).items()
    }
    session_leaves = {
        cid: _leaf_from_dict(l) for cid, l in (d.get("session_leaves") or {}).items()
    }
    tree = MemTree(
        tree_id=str(d["tree_id"]),
        tree_type=str(d.get("tree_type", "entity")),
        tree_key=str(d.get("tree_key", "")),
        label=str(d.get("label", "")),
        k=int(d.get("k", 3)),
        root_node_id=str(d.get("root_node_id", "")),
        nodes=nodes,
        session_leaves=session_leaves,
        fact_ids_ordered=list(d.get("fact_ids_ordered") or []),
        centroid=d.get("centroid"),
        version=int(d.get("version", 1)),
        lifecycle_state=str(d.get("lifecycle_state", "active")),
        deactivated=bool(d.get("deactivated", False)),
        replacement_tree_ids=list(d.get("replacement_tree_ids") or []),
        created_at=float(d.get("created_at", 0.0)),
        updated_at=float(d.get("updated_at", 0.0)),
        dirty_l0_node_ids={str(x) for x in list(d.get("dirty_l0_node_ids") or [])},
        dirty_internal_node_ids_by_level={
            int(level): {str(x) for x in list(node_ids or [])}
            for level, node_ids in (d.get("dirty_internal_node_ids_by_level") or {}).items()
        },
    )
    validate_tree_structure(tree)
    return tree


def _node_from_dict(d: dict) -> MemTreeNode:
    return MemTreeNode(
        node_id=str(d["node_id"]),
        tree_id=str(d.get("tree_id", "")),
        level=int(d.get("level", 0)),
        time_start=float(d.get("time_start", 0.0)),
        time_end=float(d.get("time_end", 0.0)),
        summary=str(d.get("summary", "")),
        summary_dirty=bool(d.get("summary_dirty", False)),
        child_ids=list(d.get("child_ids") or []),
        item_count=int(d.get("item_count", 0)),
        parent_id=str(d["parent_id"]) if d.get("parent_id") is not None else None,
        prev_leaf_id=str(d["prev_leaf_id"]) if d.get("prev_leaf_id") is not None else None,
        next_leaf_id=str(d["next_leaf_id"]) if d.get("next_leaf_id") is not None else None,
    )


def _leaf_from_dict(d: dict) -> SessionLeaf:
    return SessionLeaf(
        cell_id=str(d["cell_id"]),
        session_id=str(d.get("session_id", "")),
        cell_index=int(d.get("cell_index", 0)),
        time_start=float(d.get("time_start", 0.0)),
        time_end=float(d.get("time_end", 0.0)),
        raw_turns=list(d.get("raw_turns") or []),
        raw_text=str(d.get("raw_text", "")),
        fact_ids=list(d.get("fact_ids") or []),
    )
