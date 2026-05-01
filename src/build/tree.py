"""Core MemTree build, insert, and dirty-collection logic.

All functions are pure (no LLM calls). Summary generation is handled by
SummaryManager; this module only constructs tree structure and tracks
which nodes need new summaries.
"""

from __future__ import annotations

import math
import time
import uuid
from typing import TYPE_CHECKING

from src.build.tree_types import (
    MemTree,
    MemTreeNode,
    SessionLeaf,
    SummaryRequest,
)

if TYPE_CHECKING:
    from src.utils.types import ManagedFact, MemCell


# ─── internal helpers ────────────────────────────────────────────────────────

def _new_node_id(tree_id: str, level: int, index: int) -> str:
    return f"{tree_id}:L{level}:{index}"


def _reset_tree_to_empty_root(tree: MemTree) -> None:
    root_id = _new_node_id(tree.tree_id, 1, 0)
    now = time.time()
    tree.nodes = {
        root_id: MemTreeNode(
            node_id=root_id,
            tree_id=tree.tree_id,
            level=1,
            time_start=now,
            time_end=now,
            summary="",
            summary_dirty=True,
            child_ids=[],
            item_count=0,
            parent_id=None,
        )
    }
    tree.root_node_id = root_id
    clear_dirty_queues(tree)
    tree.dirty_internal_node_ids_by_level.setdefault(1, set()).add(root_id)


def _bucket_indices(n: int, k: int) -> list[list[int]]:
    """Partition n items into groups of at most k indices, avoiding size-1 last buckets.

    When n % k == 1 the naive approach produces a single-child last bucket which
    creates a fanout=1 internal node.  Instead we shorten the second-to-last
    group by 1 so both of the final two groups have at least 2 elements.
    """
    if n == 0:
        return []
    effective_k = max(2, k)
    buckets: list[list[int]] = []
    i = 0
    remaining = n
    while remaining > 0:
        # If taking a full bucket would leave exactly 1 left, take one less now
        # so the remaining 2 items fill the next (final) bucket of size 2.
        take = effective_k if remaining - effective_k != 1 else effective_k - 1
        take = min(take, remaining)
        buckets.append(list(range(i, i + take)))
        i += take
        remaining -= take
    return buckets


def _compute_effective_k(n_leaves: int, k: int) -> int:
    """When n_leaves < k, avoid a single-child root by expanding fan-out."""
    if n_leaves <= k:
        return max(2, n_leaves)
    return max(2, k)


def _fact_time_bounds(fact: "ManagedFact", *, default_time: float) -> tuple[float, float]:
    start_candidates: list[float] = []
    end_candidates: list[float] = []
    if fact.time_start is not None:
        start_candidates.append(float(fact.time_start))
    if fact.time_end is not None:
        end_candidates.append(float(fact.time_end))
    for occurrence in fact.occurrences:
        if occurrence.time_start is not None:
            start_candidates.append(float(occurrence.time_start))
        if occurrence.time_end is not None:
            end_candidates.append(float(occurrence.time_end))
    if not start_candidates and end_candidates:
        start_candidates.extend(end_candidates)
    if not end_candidates and start_candidates:
        end_candidates.extend(start_candidates)
    if not start_candidates:
        return default_time, default_time
    return min(start_candidates), max(end_candidates)


def _next_node_index(tree: MemTree, level: int) -> int:
    prefix = f"{tree.tree_id}:L{level}:"
    max_index = -1
    for node_id in tree.nodes:
        if not node_id.startswith(prefix):
            continue
        try:
            max_index = max(max_index, int(node_id.rsplit(":", 1)[1]))
        except (TypeError, ValueError):
            continue
    return max_index + 1


def _node_numeric_index(node_id: str) -> int:
    try:
        return int(str(node_id).rsplit(":", 1)[1])
    except (TypeError, ValueError, IndexError):
        return 0


def _mark_node_dirty(tree: MemTree, node_id: str) -> None:
    node = tree.nodes.get(node_id)
    if node is None:
        return
    node.summary_dirty = True
    if node.level == 0:
        tree.dirty_l0_node_ids.add(node.node_id)
    else:
        _refresh_internal_node_from_children(tree, node)
        tree.dirty_internal_node_ids_by_level.setdefault(node.level, set()).add(node.node_id)


def _payload_sort_key(*, time_start: float, time_end: float, payload_id: str) -> tuple[float, float, str]:
    return (float(time_start), float(time_end), str(payload_id))


def _leaf_sort_key(node: MemTreeNode) -> tuple[float, float, str]:
    payload_id = node.child_ids[0] if node.child_ids else node.node_id
    return _payload_sort_key(time_start=node.time_start, time_end=node.time_end, payload_id=payload_id)


def _refresh_internal_node_from_children(tree: MemTree, node: MemTreeNode) -> None:
    child_nodes = [tree.nodes[cid] for cid in node.child_ids if cid in tree.nodes]
    if not child_nodes:
        node.item_count = 0
        return
    node.time_start = min(child.time_start for child in child_nodes)
    node.time_end = max(child.time_end for child in child_nodes)
    node.item_count = sum(child.item_count for child in child_nodes)


def _rightmost_node_id_at_level(tree: MemTree, target_level: int) -> str | None:
    node = tree.nodes.get(tree.root_node_id)
    if node is None or node.level < target_level:
        return None
    while node.level > target_level:
        if not node.child_ids:
            return None
        next_node = tree.nodes.get(node.child_ids[-1])
        if next_node is None:
            return None
        node = next_node
    return node.node_id


def _min_internal_fanout(tree: MemTree) -> int:
    return max(1, math.ceil(tree.k / 2))


def _leftmost_leaf_id(tree: MemTree) -> str | None:
    node = tree.nodes.get(tree.root_node_id)
    if node is None:
        return None
    while node.level > 0:
        if not node.child_ids:
            return None
        node = tree.nodes.get(node.child_ids[0])
        if node is None:
            return None
    return node.node_id


def _find_l0_node_by_payload(tree: MemTree, payload_id: str) -> str | None:
    for node_id, node in tree.nodes.items():
        if node.level == 0 and node.child_ids == [payload_id]:
            return node_id
    return None


def _find_insertion_neighbors(
    tree: MemTree,
    *,
    time_start: float,
    time_end: float,
    payload_id: str,
) -> tuple[str | None, str | None]:
    target_key = _payload_sort_key(
        time_start=time_start,
        time_end=time_end,
        payload_id=payload_id,
    )
    current_id = _leftmost_leaf_id(tree)
    previous_id: str | None = None
    while current_id is not None:
        current = tree.nodes.get(current_id)
        if current is None:
            break
        if _leaf_sort_key(current) > target_key:
            return previous_id, current_id
        previous_id = current_id
        current_id = current.next_leaf_id
    return previous_id, None


def _insert_leaf_into_chain(
    tree: MemTree,
    leaf_node_id: str,
    *,
    after_leaf_id: str | None,
    before_leaf_id: str | None,
) -> None:
    leaf = tree.nodes[leaf_node_id]
    leaf.prev_leaf_id = after_leaf_id
    leaf.next_leaf_id = before_leaf_id
    if after_leaf_id and after_leaf_id in tree.nodes:
        tree.nodes[after_leaf_id].next_leaf_id = leaf_node_id
    if before_leaf_id and before_leaf_id in tree.nodes:
        tree.nodes[before_leaf_id].prev_leaf_id = leaf_node_id


def _split_internal_node_upwards(tree: MemTree, node_id: str) -> None:
    current_id = node_id
    while True:
        current = tree.nodes.get(current_id)
        if current is None or current.level == 0 or len(current.child_ids) <= tree.k:
            return

        split_at = len(current.child_ids) // 2
        right_child_ids = current.child_ids[split_at:]
        current.child_ids = current.child_ids[:split_at]
        _refresh_internal_node_from_children(tree, current)
        _mark_node_dirty(tree, current.node_id)

        sibling_id = _new_node_id(tree.tree_id, current.level, _next_node_index(tree, current.level))
        sibling = MemTreeNode(
            node_id=sibling_id,
            tree_id=tree.tree_id,
            level=current.level,
            time_start=current.time_start,
            time_end=current.time_end,
            summary="",
            summary_dirty=True,
            child_ids=list(right_child_ids),
            item_count=0,
            parent_id=current.parent_id,
        )
        for child_id in right_child_ids:
            child = tree.nodes.get(child_id)
            if child is not None:
                child.parent_id = sibling_id
        tree.nodes[sibling_id] = sibling
        _refresh_internal_node_from_children(tree, sibling)
        _mark_node_dirty(tree, sibling_id)

        if current.node_id == tree.root_node_id:
            new_root_id = _new_node_id(tree.tree_id, current.level + 1, _next_node_index(tree, current.level + 1))
            new_root = MemTreeNode(
                node_id=new_root_id,
                tree_id=tree.tree_id,
                level=current.level + 1,
                time_start=min(current.time_start, sibling.time_start),
                time_end=max(current.time_end, sibling.time_end),
                summary="",
                summary_dirty=True,
                child_ids=[current.node_id, sibling_id],
                item_count=current.item_count + sibling.item_count,
                parent_id=None,
            )
            current.parent_id = new_root_id
            sibling.parent_id = new_root_id
            tree.nodes[new_root_id] = new_root
            tree.root_node_id = new_root_id
            _mark_node_dirty(tree, new_root_id)
            return

        parent = tree.nodes.get(current.parent_id or "")
        if parent is None:
            return
        try:
            parent_index = parent.child_ids.index(current.node_id)
        except ValueError:
            parent_index = len(parent.child_ids) - 1
        parent.child_ids.insert(parent_index + 1, sibling_id)
        sibling.parent_id = parent.node_id
        _refresh_internal_node_from_children(tree, parent)
        _mark_node_dirty(tree, parent.node_id)
        current_id = parent.node_id


def _insert_child_into_parent(
    tree: MemTree,
    *,
    parent_id: str,
    child_id: str,
    child_index: int,
) -> None:
    parent = tree.nodes.get(parent_id)
    child = tree.nodes.get(child_id)
    if parent is None or child is None:
        return
    clamped_index = max(0, min(child_index, len(parent.child_ids)))
    parent.child_ids.insert(clamped_index, child_id)
    child.parent_id = parent_id
    _refresh_internal_node_from_children(tree, parent)
    _mark_node_dirty(tree, parent_id)
    _split_internal_node_upwards(tree, parent_id)


def _attach_leaf_to_tree(
    tree: MemTree,
    leaf_node_id: str,
    *,
    after_leaf_id: str | None,
    before_leaf_id: str | None,
) -> None:
    leaf = tree.nodes.get(leaf_node_id)
    root = tree.nodes.get(tree.root_node_id)
    if leaf is None:
        return

    if root is None:
        tree.root_node_id = leaf_node_id
        leaf.parent_id = None
        return

    if root.level > 0 and not root.child_ids and root.item_count == 0:
        del tree.nodes[root.node_id]
        tree.root_node_id = leaf_node_id
        leaf.parent_id = None
        return

    if root.level == 0:
        new_root_id = _new_node_id(tree.tree_id, 1, _next_node_index(tree, 1))
        left_id, right_id = (leaf_node_id, root.node_id) if before_leaf_id == root.node_id else (root.node_id, leaf_node_id)
        new_root = MemTreeNode(
            node_id=new_root_id,
            tree_id=tree.tree_id,
            level=1,
            time_start=min(tree.nodes[left_id].time_start, tree.nodes[right_id].time_start),
            time_end=max(tree.nodes[left_id].time_end, tree.nodes[right_id].time_end),
            summary="",
            summary_dirty=True,
            child_ids=[left_id, right_id],
            item_count=tree.nodes[left_id].item_count + tree.nodes[right_id].item_count,
            parent_id=None,
        )
        tree.nodes[left_id].parent_id = new_root_id
        tree.nodes[right_id].parent_id = new_root_id
        tree.nodes[new_root_id] = new_root
        tree.root_node_id = new_root_id
        _mark_node_dirty(tree, new_root_id)
        return

    anchor_leaf_id = after_leaf_id or before_leaf_id
    if anchor_leaf_id is None:
        parent_id = _rightmost_node_id_at_level(tree, 1)
        if parent_id is None:
            return
        _insert_child_into_parent(
            tree,
            parent_id=parent_id,
            child_id=leaf_node_id,
            child_index=len(tree.nodes[parent_id].child_ids),
        )
        return

    anchor_leaf = tree.nodes.get(anchor_leaf_id)
    if anchor_leaf is None or anchor_leaf.parent_id is None:
        return
    parent = tree.nodes.get(anchor_leaf.parent_id)
    if parent is None:
        return
    try:
        anchor_index = parent.child_ids.index(anchor_leaf_id)
    except ValueError:
        anchor_index = len(parent.child_ids)
    child_index = anchor_index + 1 if after_leaf_id is not None else anchor_index
    _insert_child_into_parent(
        tree,
        parent_id=parent.node_id,
        child_id=leaf_node_id,
        child_index=child_index,
    )


def _unlink_leaf_chain(tree: MemTree, leaf_node: MemTreeNode) -> None:
    if leaf_node.prev_leaf_id and leaf_node.prev_leaf_id in tree.nodes:
        tree.nodes[leaf_node.prev_leaf_id].next_leaf_id = leaf_node.next_leaf_id
    if leaf_node.next_leaf_id and leaf_node.next_leaf_id in tree.nodes:
        tree.nodes[leaf_node.next_leaf_id].prev_leaf_id = leaf_node.prev_leaf_id


def _parent_and_siblings(tree: MemTree, node_id: str) -> tuple[MemTreeNode | None, int, MemTreeNode | None, MemTreeNode | None]:
    node = tree.nodes.get(node_id)
    if node is None or node.parent_id is None:
        return None, -1, None, None
    parent = tree.nodes.get(node.parent_id)
    if parent is None:
        return None, -1, None, None
    try:
        index = parent.child_ids.index(node_id)
    except ValueError:
        return parent, -1, None, None
    left = tree.nodes.get(parent.child_ids[index - 1]) if index > 0 else None
    right = tree.nodes.get(parent.child_ids[index + 1]) if index + 1 < len(parent.child_ids) else None
    return parent, index, left, right


def _borrow_or_merge_underflow(tree: MemTree, node_id: str) -> None:
    node = tree.nodes.get(node_id)
    if node is None:
        return

    if node.node_id == tree.root_node_id:
        if node.level > 0 and len(node.child_ids) == 1:
            child_id = node.child_ids[0]
            child = tree.nodes.get(child_id)
            if child is not None:
                child.parent_id = None
                tree.root_node_id = child_id
            del tree.nodes[node.node_id]
            if child is not None:
                _mark_node_and_ancestors_dirty(tree, child_id)
        elif node.level > 0 and not node.child_ids:
            node.item_count = 0
            node.time_start = time.time()
            node.time_end = node.time_start
            _mark_node_and_ancestors_dirty(tree, node.node_id)
        else:
            _mark_node_and_ancestors_dirty(tree, node.node_id)
        return

    parent, index, left, right = _parent_and_siblings(tree, node_id)
    if parent is None or index < 0:
        return
    if len(node.child_ids) >= _min_internal_fanout(tree):
        _mark_node_and_ancestors_dirty(tree, node.node_id)
        return

    donor = left if left is not None and len(left.child_ids) > _min_internal_fanout(tree) else None
    borrow_from_left = donor is not None
    if donor is None and right is not None and len(right.child_ids) > _min_internal_fanout(tree):
        donor = right

    if donor is not None:
        if borrow_from_left:
            moved_child_id = donor.child_ids.pop()
            node.child_ids.insert(0, moved_child_id)
        else:
            moved_child_id = donor.child_ids.pop(0)
            node.child_ids.append(moved_child_id)
        moved_child = tree.nodes.get(moved_child_id)
        if moved_child is not None:
            moved_child.parent_id = node.node_id
        _refresh_internal_node_from_children(tree, donor)
        _refresh_internal_node_from_children(tree, node)
        _mark_node_and_ancestors_dirty(tree, donor.node_id)
        _mark_node_and_ancestors_dirty(tree, node.node_id)
        return

    merge_target = left if left is not None else right
    if merge_target is None:
        _mark_node_and_ancestors_dirty(tree, node.node_id)
        return

    if merge_target is left:
        merged_children = merge_target.child_ids + node.child_ids
        remove_id = node.node_id
        keep = merge_target
    else:
        merged_children = node.child_ids + merge_target.child_ids
        remove_id = merge_target.node_id
        keep = node

    if len(merged_children) > tree.k:
        _mark_node_and_ancestors_dirty(tree, node.node_id)
        return

    keep.child_ids = merged_children
    for child_id in merged_children:
        child = tree.nodes.get(child_id)
        if child is not None:
            child.parent_id = keep.node_id
    _refresh_internal_node_from_children(tree, keep)
    parent.child_ids.remove(remove_id)
    del tree.nodes[remove_id]
    _mark_node_and_ancestors_dirty(tree, keep.node_id)
    if parent.node_id in tree.nodes:
        _refresh_internal_node_from_children(tree, parent)
        _borrow_or_merge_underflow(tree, parent.node_id)


def delete_fact(tree: MemTree, fact_id: str) -> bool:
    """Delete one fact payload leaf from an entity/scene tree and rebalance upward."""
    leaf_id = _find_l0_node_by_payload(tree, fact_id)
    if leaf_id is None:
        return False
    leaf = tree.nodes.get(leaf_id)
    if leaf is None:
        return False
    _unlink_leaf_chain(tree, leaf)
    if fact_id in tree.fact_ids_ordered:
        tree.fact_ids_ordered.remove(fact_id)
    parent_id = leaf.parent_id
    del tree.nodes[leaf_id]
    tree.dirty_l0_node_ids.discard(leaf_id)
    if parent_id is not None and parent_id in tree.nodes:
        parent = tree.nodes[parent_id]
        if leaf_id in parent.child_ids:
            parent.child_ids.remove(leaf_id)
        _refresh_internal_node_from_children(tree, parent)
        _borrow_or_merge_underflow(tree, parent_id)
    elif tree.root_node_id == leaf_id:
        if tree.nodes:
            leftmost = _leftmost_leaf_id(tree)
            if leftmost is not None:
                tree.root_node_id = leftmost
                tree.nodes[leftmost].parent_id = None
        else:
            _reset_tree_to_empty_root(tree)
    tree.version += 1
    tree.updated_at = time.time()
    validate_tree_structure(tree)
    return True


def delete_cell(tree: MemTree, cell_id: str) -> bool:
    """Delete one session-cell payload leaf from a session tree and rebalance upward."""
    leaf_id = _find_l0_node_by_payload(tree, cell_id)
    if leaf_id is None:
        return False
    leaf = tree.nodes.get(leaf_id)
    if leaf is None:
        return False
    _unlink_leaf_chain(tree, leaf)
    tree.session_leaves.pop(cell_id, None)
    parent_id = leaf.parent_id
    del tree.nodes[leaf_id]
    tree.dirty_l0_node_ids.discard(leaf_id)
    if parent_id is not None and parent_id in tree.nodes:
        parent = tree.nodes[parent_id]
        if leaf_id in parent.child_ids:
            parent.child_ids.remove(leaf_id)
        _refresh_internal_node_from_children(tree, parent)
        _borrow_or_merge_underflow(tree, parent_id)
    elif tree.root_node_id == leaf_id:
        if tree.nodes:
            leftmost = _leftmost_leaf_id(tree)
            if leftmost is not None:
                tree.root_node_id = leftmost
                tree.nodes[leftmost].parent_id = None
        else:
            _reset_tree_to_empty_root(tree)
    tree.version += 1
    tree.updated_at = time.time()
    validate_tree_structure(tree)
    return True


def _mark_node_and_ancestors_dirty(tree: MemTree, node_id: str) -> None:
    current_id: str | None = node_id
    visited: set[str] = set()
    while current_id and current_id not in visited:
        visited.add(current_id)
        current = tree.nodes.get(current_id)
        if current is None:
            break
        _mark_node_dirty(tree, current_id)
        current_id = current.parent_id


def clear_dirty_queues(tree: MemTree) -> None:
    tree.dirty_l0_node_ids.clear()
    tree.dirty_internal_node_ids_by_level.clear()


def rebuild_dirty_queues_from_flags(tree: MemTree) -> None:
    clear_dirty_queues(tree)
    for node in tree.nodes.values():
        if not node.summary_dirty:
            continue
        if node.level == 0:
            tree.dirty_l0_node_ids.add(node.node_id)
        else:
            tree.dirty_internal_node_ids_by_level.setdefault(node.level, set()).add(node.node_id)


def _build_single_leaf_hierarchy(
    *,
    tree_id: str,
    tree_type: str,
    tree_label: str,
    level0_ids: list[str],
    nodes: dict[str, MemTreeNode],
    effective_k: int,
) -> dict[int, list[SummaryRequest]]:
    requests_by_level: dict[int, list[SummaryRequest]] = {0: []}
    current_level_ids = list(level0_ids)
    level = 1
    while len(current_level_ids) > 1:
        next_level_ids: list[str] = []
        req_this_level: list[SummaryRequest] = []
        for group_num, indices in enumerate(_bucket_indices(len(current_level_ids), effective_k)):
            child_ids = [current_level_ids[i] for i in indices]
            ts = min(nodes[cid].time_start for cid in child_ids)
            te = max(nodes[cid].time_end for cid in child_ids)
            item_count = sum(nodes[cid].item_count for cid in child_ids)
            node_id = _new_node_id(tree_id, level, group_num)
            node = MemTreeNode(
                node_id=node_id,
                tree_id=tree_id,
                level=level,
                time_start=ts,
                time_end=te,
                summary="",
                summary_dirty=True,
                child_ids=child_ids,
                item_count=item_count,
                parent_id=None,
            )
            for child_id in child_ids:
                nodes[child_id].parent_id = node_id
            nodes[node_id] = node
            next_level_ids.append(node_id)
            req_this_level.append(SummaryRequest(
                request_id=f"{tree_id}:{node_id}",
                tree_id=tree_id,
                node_id=node_id,
                tree_type=tree_type,
                tree_label=tree_label,
                level=level,
                time_start=ts,
                time_end=te,
                input_text="",
            ))
        requests_by_level[level] = req_this_level
        current_level_ids = next_level_ids
        level += 1
    return requests_by_level


def validate_tree_structure(tree: MemTree) -> None:
    """Validate that L0 nodes are single payload leaves and internals point to nodes."""
    if tree.root_node_id not in tree.nodes:
        raise ValueError(f"tree {tree.tree_id} has missing root_node_id={tree.root_node_id}")
    root = tree.nodes[tree.root_node_id]
    if root.parent_id is not None:
        raise ValueError(f"tree {tree.tree_id} root {tree.root_node_id} must have parent_id=None")
    l0_nodes: list[MemTreeNode] = []
    for node_id, node in tree.nodes.items():
        if node.level == 0:
            l0_nodes.append(node)
            if len(node.child_ids) != 1:
                raise ValueError(
                    f"tree {tree.tree_id} L0 node {node_id} must have exactly one payload child, "
                    f"got {len(node.child_ids)}"
                )
            payload_id = node.child_ids[0]
            if tree.tree_type == "session":
                if payload_id not in tree.session_leaves:
                    raise ValueError(
                        f"tree {tree.tree_id} L0 node {node_id} references missing cell_id={payload_id}"
                    )
            else:
                if not payload_id.startswith("fact_"):
                    raise ValueError(
                        f"tree {tree.tree_id} L0 node {node_id} must reference fact_id, got {payload_id}"
                    )
            if node.item_count != 1:
                raise ValueError(
                    f"tree {tree.tree_id} L0 node {node_id} item_count must be 1, got {node.item_count}"
                )
            if node.parent_id is None and node.node_id != tree.root_node_id:
                raise ValueError(
                    f"tree {tree.tree_id} non-root L0 node {node_id} has no parent_id"
                )
            continue
        if not node.child_ids:
            if node_id == tree.root_node_id and node.item_count == 0:
                continue
            raise ValueError(f"tree {tree.tree_id} internal node {node_id} has no children")
        if len(node.child_ids) > tree.k:
            raise ValueError(
                f"tree {tree.tree_id} internal node {node_id} fanout {len(node.child_ids)} exceeds k={tree.k}"
            )
        for child_id in node.child_ids:
            child = tree.nodes.get(child_id)
            if child is None:
                raise ValueError(
                    f"tree {tree.tree_id} internal node {node_id} references non-node child_id={child_id}"
                )
            if child.level != node.level - 1:
                raise ValueError(
                    f"tree {tree.tree_id} edge {node_id}->{child_id} crosses invalid levels "
                    f"{node.level}->{child.level}"
                )
            if child.parent_id != node_id:
                raise ValueError(
                    f"tree {tree.tree_id} child {child_id} has parent_id={child.parent_id}, "
                    f"expected {node_id}"
                )
    # Verify the doubly-linked chain by walking it, not by re-sorting.
    # Re-sorting would require the caller's insertion order to exactly match
    # the _leaf_sort_key order, but batch builds sort by f.time_start while
    # L0 nodes use _fact_time_bounds (includes occurrence times) — they can differ.
    l0_by_id = {n.node_id: n for n in l0_nodes}
    leftmost = next((n for n in l0_nodes if n.prev_leaf_id is None), None)
    if l0_nodes and leftmost is None:
        raise ValueError(
            f"tree {tree.tree_id} L0 chain has no node with prev_leaf_id=None"
        )
    visited_chain: list[MemTreeNode] = []
    current: MemTreeNode | None = leftmost
    visited_ids: set[str] = set()
    while current is not None:
        if current.node_id in visited_ids:
            raise ValueError(
                f"tree {tree.tree_id} L0 chain cycle detected at {current.node_id}"
            )
        visited_ids.add(current.node_id)
        # verify back-pointer
        if visited_chain:
            prev_node = visited_chain[-1]
            if current.prev_leaf_id != prev_node.node_id:
                raise ValueError(
                    f"tree {tree.tree_id} L0 chain back-pointer broken at {current.node_id}: "
                    f"prev={current.prev_leaf_id}, expected={prev_node.node_id}"
                )
            if prev_node.next_leaf_id != current.node_id:
                raise ValueError(
                    f"tree {tree.tree_id} L0 chain forward-pointer broken at {prev_node.node_id}: "
                    f"next={prev_node.next_leaf_id}, expected={current.node_id}"
                )
        visited_chain.append(current)
        nxt_id = current.next_leaf_id
        current = l0_by_id.get(nxt_id) if nxt_id else None
        if nxt_id and nxt_id not in l0_by_id:
            raise ValueError(
                f"tree {tree.tree_id} L0 chain points to non-L0/missing node {nxt_id}"
            )
    if len(visited_chain) != len(l0_nodes):
        unreachable = set(l0_by_id.keys()) - visited_ids
        raise ValueError(
            f"tree {tree.tree_id} L0 chain covers {len(visited_chain)}/{len(l0_nodes)} nodes; "
            f"unreachable: {sorted(unreachable)[:5]}"
        )
    # Verify chain is sorted non-decreasingly by (time_start, time_end). The
    # fact_id / cell_id tie-breaker in _leaf_sort_key is a total-order
    # convenience for the builder, not an invariant: after merge_user_forests
    # remaps source fact_ids onto target canonicals, two equal-time leaves
    # may swap fact_id string order, which is harmless. We only enforce
    # non-decreasing time here.
    for i in range(1, len(visited_chain)):
        prev_times = (visited_chain[i - 1].time_start, visited_chain[i - 1].time_end)
        curr_times = (visited_chain[i].time_start, visited_chain[i].time_end)
        if curr_times < prev_times:
            raise ValueError(
                f"tree {tree.tree_id} L0 chain ordering violation between "
                f"{visited_chain[i-1].node_id} and {visited_chain[i].node_id}: "
                f"{prev_times} > {curr_times}"
            )


# ─── build from facts (actor / scene trees) ──────────────────────────────────

def build_tree_from_facts(
    tree_id: str,
    tree_type: str,
    tree_key: str,
    label: str,
    k: int,
    facts: "list[ManagedFact]",
) -> tuple[MemTree, dict[int, list[SummaryRequest]]]:
    """Bottom-up tree construction from a time-sorted list of ManagedFacts.

    Returns the MemTree (nodes created, summaries empty) and a dict mapping
    level → SummaryRequests for that level. Caller must process level by level,
    bottom-up, filling in node.summary before building the next level's requests.
    """
    now = time.time()
    if not facts:
        # degenerate empty tree
        root_id = _new_node_id(tree_id, 1, 0)
        root_node = MemTreeNode(
            node_id=root_id,
            tree_id=tree_id,
            level=1,
            time_start=now,
            time_end=now,
            summary="",
            summary_dirty=True,
            child_ids=[],
            item_count=0,
        )
        tree = MemTree(
            tree_id=tree_id,
            tree_type=tree_type,
            tree_key=tree_key,
            label=label,
            k=k,
            root_node_id=root_id,
            nodes={root_id: root_node},
            session_leaves={},
            fact_ids_ordered=[],
            centroid=None,
            version=1,
            lifecycle_state="active",
            deactivated=False,
            replacement_tree_ids=[],
            created_at=now,
            updated_at=now,
        )
        return tree, {}

    effective_k = _compute_effective_k(len(facts), k)
    nodes: dict[str, MemTreeNode] = {}
    requests_by_level: dict[int, list[SummaryRequest]] = {}

    # Re-sort by _fact_time_bounds so that L0 node time_start values are
    # consistent with the chain order. The caller sorts by f.time_start, but
    # _fact_time_bounds can return an earlier timestamp (from occurrence times),
    # which would make the chain order disagree with the sorted fact list.
    time_bounds_pre = [_fact_time_bounds(f, default_time=now) for f in facts]
    facts_with_bounds = sorted(
        zip(facts, time_bounds_pre),
        key=lambda pair: (pair[1][0], pair[1][1], pair[0].fact_id),
    )
    facts = [f for f, _ in facts_with_bounds]
    time_bounds_list = [b for _, b in facts_with_bounds]

    # Level 0: one node per fact payload
    fact_ids = [f.fact_id for f in facts]
    time_bounds = time_bounds_list
    fact_map = {f.fact_id: f for f in facts}

    level0_ids: list[str] = []
    req_level0: list[SummaryRequest] = []
    for leaf_num, fact_id in enumerate(fact_ids):
        node_id = _new_node_id(tree_id, 0, leaf_num)
        ts, te = time_bounds[leaf_num]
        prev_leaf_id = _new_node_id(tree_id, 0, leaf_num - 1) if leaf_num > 0 else None
        next_leaf_id = _new_node_id(tree_id, 0, leaf_num + 1) if leaf_num + 1 < len(fact_ids) else None
        node = MemTreeNode(
            node_id=node_id,
            tree_id=tree_id,
            level=0,
            time_start=ts,
            time_end=te,
            summary="",
            summary_dirty=True,
            child_ids=[fact_id],
            item_count=1,
            parent_id=None,
            prev_leaf_id=prev_leaf_id,
            next_leaf_id=next_leaf_id,
        )
        nodes[node_id] = node
        level0_ids.append(node_id)
        requests_by_level.setdefault(0, [])
        req_level0.append(SummaryRequest(
            request_id=f"{tree_id}:{node_id}",
            tree_id=tree_id,
            node_id=node_id,
            tree_type=tree_type,
            tree_label=label,
            level=0,
            time_start=node.time_start,
            time_end=node.time_end,
            input_text=fact_map[fact_id].fact_text,
        ))
    requests_by_level = _build_single_leaf_hierarchy(
        tree_id=tree_id,
        tree_type=tree_type,
        tree_label=label,
        level0_ids=level0_ids,
        nodes=nodes,
        effective_k=effective_k,
    )
    requests_by_level[0] = req_level0
    root_node_id = level0_ids[0]
    if len(level0_ids) > 1:
        root_level = max(node.level for node in nodes.values())
        root_node_id = _new_node_id(tree_id, root_level, 0)

    ts_tree = min(n.time_start for n in nodes.values())
    te_tree = max(n.time_end for n in nodes.values())

    tree = MemTree(
        tree_id=tree_id,
        tree_type=tree_type,
        tree_key=tree_key,
        label=label,
        k=effective_k,
        root_node_id=root_node_id,
        nodes=nodes,
        session_leaves={},
        fact_ids_ordered=fact_ids,
        centroid=None,
        version=1,
        lifecycle_state="active",
        deactivated=False,
        replacement_tree_ids=[],
        created_at=now,
        updated_at=now,
        dirty_l0_node_ids=set(level0_ids),
        dirty_internal_node_ids_by_level={
            level: {req.node_id for req in reqs}
            for level, reqs in requests_by_level.items()
            if level > 0 and reqs
        },
    )
    validate_tree_structure(tree)
    return tree, requests_by_level


# ─── build from cells (session trees) ────────────────────────────────────────

def build_tree_from_cells(
    tree_id: str,
    session_id: str,
    k: int,
    cells: "list[MemCell]",
    cell_to_facts: dict[str, list[str]],
) -> tuple[MemTree, dict[int, list[SummaryRequest]]]:
    """Session tree variant: leaves are SessionLeaf objects (not facts)."""
    now = time.time()
    session_leaves: dict[str, SessionLeaf] = {}

    if not cells:
        root_id = _new_node_id(tree_id, 1, 0)
        root_node = MemTreeNode(
            node_id=root_id,
            tree_id=tree_id,
            level=1,
            time_start=now,
            time_end=now,
            summary="",
            summary_dirty=True,
            child_ids=[],
            item_count=0,
        )
        tree = MemTree(
            tree_id=tree_id,
            tree_type="session",
            tree_key=session_id,
            label=session_id,
            k=k,
            root_node_id=root_id,
            nodes={root_id: root_node},
            session_leaves={},
            fact_ids_ordered=[],
            centroid=None,
            version=1,
            lifecycle_state="active",
            deactivated=False,
            replacement_tree_ids=[],
            created_at=now,
            updated_at=now,
        )
        return tree, {}

    effective_k = _compute_effective_k(len(cells), k)
    nodes: dict[str, MemTreeNode] = {}
    requests_by_level: dict[int, list[SummaryRequest]] = {}

    # Re-sort cells by (time_start, time_end, cell_id) so L0 node timestamps
    # are consistent with the chain order and _leaf_sort_key.
    # Callers may sort by cell_index which can differ from time order.
    cells = sorted(cells, key=lambda c: (c.time_start, c.time_end, c.cell_id))

    # Create SessionLeaf for every cell
    for cell in cells:
        raw_turns = [
            {"role": t.speaker_tag, "content": t.text}
            for t in (cell.turns or [])
        ]
        leaf = SessionLeaf(
            cell_id=cell.cell_id,
            session_id=cell.session_id,
            cell_index=cell.cell_index,
            time_start=cell.time_start,
            time_end=cell.time_end,
            raw_turns=raw_turns,
            raw_text=cell.text,
            fact_ids=list(cell_to_facts.get(cell.cell_id, [])),
        )
        session_leaves[cell.cell_id] = leaf

    # Level 0: one node per cell payload
    level0_ids: list[str] = []
    req_level0: list[SummaryRequest] = []
    for leaf_num, cell in enumerate(cells):
        node_id = _new_node_id(tree_id, 0, leaf_num)
        prev_leaf_id = _new_node_id(tree_id, 0, leaf_num - 1) if leaf_num > 0 else None
        next_leaf_id = _new_node_id(tree_id, 0, leaf_num + 1) if leaf_num + 1 < len(cells) else None
        node = MemTreeNode(
            node_id=node_id,
            tree_id=tree_id,
            level=0,
            time_start=cell.time_start,
            time_end=cell.time_end,
            summary="",
            summary_dirty=True,
            child_ids=[cell.cell_id],
            item_count=1,
            parent_id=None,
            prev_leaf_id=prev_leaf_id,
            next_leaf_id=next_leaf_id,
        )
        nodes[node_id] = node
        level0_ids.append(node_id)
        req_level0.append(SummaryRequest(
            request_id=f"{tree_id}:{node_id}",
            tree_id=tree_id,
            node_id=node_id,
            tree_type="session",
            tree_label=session_id,
            level=0,
            time_start=cell.time_start,
            time_end=cell.time_end,
            input_text=session_leaves[cell.cell_id].raw_text,
        ))
    requests_by_level = _build_single_leaf_hierarchy(
        tree_id=tree_id,
        tree_type="session",
        tree_label=session_id,
        level0_ids=level0_ids,
        nodes=nodes,
        effective_k=effective_k,
    )
    requests_by_level[0] = req_level0
    root_node_id = level0_ids[0]
    if len(level0_ids) > 1:
        root_level = max(node.level for node in nodes.values())
        root_node_id = _new_node_id(tree_id, root_level, 0)
    ts_tree = min(n.time_start for n in nodes.values())
    te_tree = max(n.time_end for n in nodes.values())

    tree = MemTree(
        tree_id=tree_id,
        tree_type="session",
        tree_key=session_id,
        label=session_id,
        k=effective_k,
        root_node_id=root_node_id,
        nodes=nodes,
        session_leaves=session_leaves,
        fact_ids_ordered=[],
        centroid=None,
        version=1,
        lifecycle_state="active",
        deactivated=False,
        replacement_tree_ids=[],
        created_at=now,
        updated_at=now,
        dirty_l0_node_ids=set(level0_ids),
        dirty_internal_node_ids_by_level={
            level: {req.node_id for req in reqs}
            for level, reqs in requests_by_level.items()
            if level > 0 and reqs
        },
    )
    validate_tree_structure(tree)
    return tree, requests_by_level


# ─── incremental insert ───────────────────────────────────────────────────────

def insert_fact(tree: MemTree, fact: "ManagedFact") -> None:
    ts, te = _fact_time_bounds(fact, default_time=time.time())
    after_leaf_id, before_leaf_id = _find_insertion_neighbors(
        tree,
        time_start=ts,
        time_end=te,
        payload_id=fact.fact_id,
    )
    leaf_id = _new_node_id(tree.tree_id, 0, _next_node_index(tree, 0))
    tree.nodes[leaf_id] = MemTreeNode(
        node_id=leaf_id,
        tree_id=tree.tree_id,
        level=0,
        time_start=ts,
        time_end=te,
        summary="",
        summary_dirty=True,
        child_ids=[fact.fact_id],
        item_count=1,
        parent_id=None,
        prev_leaf_id=None,
        next_leaf_id=None,
    )
    _insert_leaf_into_chain(
        tree,
        leaf_id,
        after_leaf_id=after_leaf_id,
        before_leaf_id=before_leaf_id,
    )
    _attach_leaf_to_tree(
        tree,
        leaf_id,
        after_leaf_id=after_leaf_id,
        before_leaf_id=before_leaf_id,
    )
    if after_leaf_id and after_leaf_id in tree.nodes:
        anchor_fact_id = tree.nodes[after_leaf_id].child_ids[0]
        try:
            insert_at = tree.fact_ids_ordered.index(anchor_fact_id) + 1
        except ValueError:
            insert_at = len(tree.fact_ids_ordered)
    elif before_leaf_id and before_leaf_id in tree.nodes:
        anchor_fact_id = tree.nodes[before_leaf_id].child_ids[0]
        try:
            insert_at = tree.fact_ids_ordered.index(anchor_fact_id)
        except ValueError:
            insert_at = 0
    else:
        insert_at = len(tree.fact_ids_ordered)
    tree.fact_ids_ordered.insert(insert_at, fact.fact_id)
    _mark_node_and_ancestors_dirty(tree, leaf_id)
    tree.version += 1
    tree.updated_at = time.time()
    validate_tree_structure(tree)


def insert_cell(tree: MemTree, cell: "MemCell", fact_ids: list[str]) -> None:
    """Insert a MemCell as a fresh L0 single-payload leaf using time order."""
    raw_turns = [
        {"role": t.speaker_tag, "content": t.text}
        for t in (cell.turns or [])
    ]
    leaf = SessionLeaf(
        cell_id=cell.cell_id,
        session_id=cell.session_id,
        cell_index=cell.cell_index,
        time_start=cell.time_start,
        time_end=cell.time_end,
        raw_turns=raw_turns,
        raw_text=cell.text,
        fact_ids=fact_ids,
    )
    tree.session_leaves[cell.cell_id] = leaf

    after_leaf_id, before_leaf_id = _find_insertion_neighbors(
        tree,
        time_start=cell.time_start,
        time_end=cell.time_end,
        payload_id=cell.cell_id,
    )
    leaf_id = _new_node_id(tree.tree_id, 0, _next_node_index(tree, 0))
    tree.nodes[leaf_id] = MemTreeNode(
        node_id=leaf_id,
        tree_id=tree.tree_id,
        level=0,
        time_start=cell.time_start,
        time_end=cell.time_end,
        summary="",
        summary_dirty=True,
        child_ids=[cell.cell_id],
        item_count=1,
        parent_id=None,
        prev_leaf_id=None,
        next_leaf_id=None,
    )
    _insert_leaf_into_chain(
        tree,
        leaf_id,
        after_leaf_id=after_leaf_id,
        before_leaf_id=before_leaf_id,
    )
    _attach_leaf_to_tree(
        tree,
        leaf_id,
        after_leaf_id=after_leaf_id,
        before_leaf_id=before_leaf_id,
    )
    _mark_node_and_ancestors_dirty(tree, leaf_id)
    tree.version += 1
    tree.updated_at = time.time()
    validate_tree_structure(tree)


# ─── dirty request collection ─────────────────────────────────────────────────

def collect_dirty_requests(
    tree: MemTree,
    fact_map: "dict[str, ManagedFact] | None" = None,
) -> dict[int, list[SummaryRequest]]:
    """Collect SummaryRequests for all dirty nodes, grouped by level.

    Level 0 is returned first (input: fact/cell texts).
    Higher levels come after (input: child summaries — must be filled by caller).
    """
    requests_by_level: dict[int, list[SummaryRequest]] = {}

    dirty_node_ids: list[str] = [
        node_id for node_id in sorted(tree.dirty_l0_node_ids) if node_id in tree.nodes
    ]
    for level in sorted(tree.dirty_internal_node_ids_by_level):
        dirty_node_ids.extend(
            node_id
            for node_id in sorted(tree.dirty_internal_node_ids_by_level.get(level, set()))
            if node_id in tree.nodes
        )

    if not dirty_node_ids:
        dirty_node_ids = [
            node.node_id
            for node in sorted(tree.nodes.values(), key=lambda n: (n.level, n.node_id))
            if node.summary_dirty
        ]

    for node_id in dirty_node_ids:
        node = tree.nodes.get(node_id)
        if node is None:
            continue
        if not node.summary_dirty:
            continue
        if node.level > 0 and not node.child_ids:
            node.summary = ""
            node.summary_dirty = False
            tree.dirty_internal_node_ids_by_level.get(node.level, set()).discard(node.node_id)
            continue

        if node.level == 0:
            input_text = _build_level0_input(tree, node, fact_map)
        else:
            input_text = ""  # caller fills after lower level summaries land

        req = SummaryRequest(
            request_id=f"{tree.tree_id}:{node.node_id}",
            tree_id=tree.tree_id,
            node_id=node.node_id,
            tree_type=tree.tree_type,
            tree_label=tree.label,
            level=node.level,
            time_start=node.time_start,
            time_end=node.time_end,
            input_text=input_text,
        )
        requests_by_level.setdefault(node.level, []).append(req)

    return requests_by_level


def _build_level0_input(
    tree: MemTree,
    node: MemTreeNode,
    fact_map: "dict[str, ManagedFact] | None",
) -> str:
    if tree.tree_type == "session":
        parts = [
            tree.session_leaves[cid].raw_text
            for cid in node.child_ids
            if cid in tree.session_leaves
        ]
    else:
        parts = []
        for fid in node.child_ids:
            if fact_map and fid in fact_map:
                parts.append(fact_map[fid].fact_text)
    return "\n\n".join(parts)


# ─── fill upper-level request input after lower summaries are ready ───────────

def fill_upper_level_inputs(
    tree: MemTree,
    requests_by_level: dict[int, list[SummaryRequest]],
    results_by_node: dict[str, str],
) -> dict[int, list[SummaryRequest]]:
    """Replace empty input_text in levels > 0 using freshly generated summaries.

    `results_by_node` maps node_id → summary string.
    Returns an updated dict with the same structure.
    """
    updated: dict[int, list[SummaryRequest]] = {}
    for level, reqs in requests_by_level.items():
        if level == 0:
            updated[level] = reqs
            continue
        new_reqs: list[SummaryRequest] = []
        for req in reqs:
            node = tree.nodes.get(req.node_id)
            if node is None:
                new_reqs.append(req)
                continue
            child_summaries = [
                results_by_node.get(cid, tree.nodes[cid].summary if cid in tree.nodes else "")
                for cid in node.child_ids
            ]
            input_text = "\n\n".join(s for s in child_summaries if s)
            new_reqs.append(SummaryRequest(
                request_id=req.request_id,
                tree_id=req.tree_id,
                node_id=req.node_id,
                tree_type=req.tree_type,
                tree_label=req.tree_label,
                level=req.level,
                time_start=req.time_start,
                time_end=req.time_end,
                input_text=input_text,
            ))
        updated[level] = new_reqs
    return updated
