"""Efficient multi-forest merge.

Combines several already-built ``UserForest`` snapshots into a new target
forest without rebuilding trees from scratch. The pipeline is:

1. Merge ``DeletableFactManager`` via its built-in ``merge_from`` (dedup runs
   automatically). A source→target ``fact_id`` remap is reconstructed from
   the target's normalized-text index, with a FAISS cosine fallback for
   LLM-judged duplicates whose canonical text differs from the source text.
2. Merge routers: ``EntityRouter.merge_from`` for entity lifecycle state and
   ``SceneRouter.merge_from`` for clusters (returns a ``cluster_id`` remap).
3. Group source trees by their target tree_id:
     * ``entity:<key>``  — identity stable across forests.
     * ``tree:<cluster>`` — remapped via the scene router cluster remap.
     * ``session:<sid>`` — identity stable (collisions merged like entities).
4. For each group:
     * One member  → *copy path*: deep-copy the tree, rekey node ids if the
       tree_id changed, remap internal ``fact_ids`` to target canonicals,
       and also copy node_index entries from the source to avoid re-embedding
       unchanged summaries.
     * Multiple    → *merge path*: keep the tree with the most leaves as the
       trunk, insert every leaf from the other members via ``insert_fact`` /
       ``insert_cell``, flush dirty ancestors, and refresh node_index for
       that tree only.
5. Persist the target forest.

Only trees that actually got new leaves inserted pay for LLM summary calls.
Copied trees preserve their original summaries and root embeddings.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.build.node_index import NodeEntry
from src.build.tree import insert_cell, insert_fact, rebuild_dirty_queues_from_flags
from src.build.tree_types import MemTree

if TYPE_CHECKING:
    from src.build.node_index import NodeIndex
    from src.extraction.fact_manager import FactManager
    from src.forest.user_forest import UserForest
    from src.utils.types import ManagedFact


# ── result type ───────────────────────────────────────────────────────────────

@dataclass
class ForestMergeResult:
    """Summary of a :func:`merge_user_forests` run."""

    target_user_id: str
    source_user_ids: list[str]
    facts_inserted: int           # new canonical facts created in target
    facts_merged: int             # source facts absorbed into existing canonicals
    trees_copied: int             # trees that had only one source (no flush)
    trees_merged: int             # trees whose trunk absorbed leaves from others
    flushed_tree_ids: list[str]   # trees that required LLM re-summarization
    total_trees: int
    elapsed_sec: float


# ── fact remap ────────────────────────────────────────────────────────────────

def _build_fact_remap(
    source_fm: "FactManager",
    target_fm: "FactManager",
    *,
    fallback_min_cosine: float = 0.90,
) -> dict[str, str]:
    """Return ``{src_fact_id: target_fact_id}`` after a merge_from call.

    Fast path: look up each source fact's normalized text in the target's
    ``_exact_index``. This covers:
      - source fact was absorbed as a duplicate with identical normalized text
      - source fact was newly inserted (target's canonical text == source text)

    Fallback: source facts whose text got LLM-matched to an existing target
    canonical with a *different* text are invisible to the exact index. For
    those, search the target FAISS index by embedding and accept matches
    above ``fallback_min_cosine``.
    """
    remap: dict[str, str] = {}
    unresolved: list["ManagedFact"] = []

    with source_fm._lock:
        source_facts = list(source_fm._facts.values())

    with target_fm._lock:
        for src_fact in source_facts:
            key = target_fm._normalize_fact_text(src_fact.fact_text)
            tgt_fid = target_fm._exact_index.get(key)
            if tgt_fid is None and src_fact.fact_id in target_fm._facts:
                tgt_fid = src_fact.fact_id
            if tgt_fid is not None:
                remap[src_fact.fact_id] = tgt_fid
            else:
                unresolved.append(src_fact)

    if unresolved and target_fm.embedding_client is not None and target_fm._index.ntotal > 0:
        for src_fact in unresolved:
            hits = target_fm.search_similar_fact_text(src_fact.fact_text, top_k=1)
            if hits and hits[0][1] >= fallback_min_cosine:
                remap[src_fact.fact_id] = hits[0][0].fact_id

    return remap


# ── tree rekey / remap ────────────────────────────────────────────────────────

def _rekey_tree(tree: MemTree, new_tree_id: str) -> MemTree:
    """Deep-copy ``tree`` and rewrite its ``tree_id`` + every node id.

    Node ids are of the form ``"{tree_id}:L{level}:{index}"`` (see
    ``_new_node_id`` in tree.py), so we can do a prefix swap. L0 ``child_ids``
    carry fact_ids / cell_ids and are left untouched — the caller remaps
    those via :func:`_remap_tree_fact_ids`.
    """
    old_prefix = tree.tree_id + ":"
    new_prefix = new_tree_id + ":"

    def _rekey(nid: str | None) -> str | None:
        if nid is None:
            return None
        if nid.startswith(old_prefix):
            return new_prefix + nid[len(old_prefix):]
        return nid

    clone = copy.deepcopy(tree)
    clone.tree_id = new_tree_id
    clone.root_node_id = _rekey(clone.root_node_id) or clone.root_node_id

    new_nodes = {}
    for node in clone.nodes.values():
        node.node_id = _rekey(node.node_id) or node.node_id
        node.tree_id = new_tree_id
        node.parent_id = _rekey(node.parent_id)
        node.prev_leaf_id = _rekey(node.prev_leaf_id)
        node.next_leaf_id = _rekey(node.next_leaf_id)
        if node.level > 0:
            node.child_ids = [_rekey(cid) or cid for cid in node.child_ids]
        new_nodes[node.node_id] = node
    clone.nodes = new_nodes

    # Start with clean dirty queues — we preserve existing summaries.
    clone.dirty_l0_node_ids = set()
    clone.dirty_internal_node_ids_by_level = {}
    return clone


def _remap_tree_fact_ids(tree: MemTree, fact_remap: dict[str, str]) -> None:
    """Rewrite fact_id references inside ``tree`` to target-forest fact_ids.

    Unmapped source fact_ids are left in place — the query-time fact loader
    handles missing lookups gracefully, and removing L0 leaves would break
    the sibling chain and fan-out invariants.
    """
    if tree.tree_type == "session":
        for leaf in tree.session_leaves.values():
            leaf.fact_ids = [fact_remap.get(fid, fid) for fid in leaf.fact_ids]
        return

    tree.fact_ids_ordered = [fact_remap.get(fid, fid) for fid in tree.fact_ids_ordered]
    for node in tree.nodes.values():
        if node.level != 0 or not node.child_ids:
            continue
        src_fid = node.child_ids[0]
        node.child_ids = [fact_remap.get(src_fid, src_fid)]


def _leaf_count(tree: MemTree) -> int:
    return len(tree.session_leaves) or len(tree.fact_ids_ordered)


# ── node_index carry-over ─────────────────────────────────────────────────────

def _copy_node_index_entries(
    *,
    source_idx: "NodeIndex",
    target_idx: "NodeIndex",
    src_tree_id: str,
    dst_tree_id: str,
) -> int:
    """Copy all NodeIndex entries for ``src_tree_id`` from source to target.

    Rekeys node ids when the tree was rekeyed. Searchable (root) entries go
    through ``add_node(searchable=True)``; non-searchable browse entries get
    registered with ``searchable=False`` so they land only in ``_emb_store``.
    """
    with source_idx._lock:
        src_entries = [e for e in source_idx._entries if e.tree_id == src_tree_id]
        emb_store_snapshot = dict(source_idx._emb_store)

    old_prefix = src_tree_id + ":"
    new_prefix = dst_tree_id + ":"

    def _rekey(node_id: str) -> str:
        if node_id.startswith(old_prefix):
            return new_prefix + node_id[len(old_prefix):]
        return node_id

    count = 0
    searchable_old_ids: set[str] = set()

    for entry in src_entries:
        searchable_old_ids.add(entry.node_id)
        emb = emb_store_snapshot.get(entry.node_id)
        if emb is None:
            continue
        target_idx.add_node(
            NodeEntry(
                node_id=_rekey(entry.node_id),
                tree_id=dst_tree_id,
                level=entry.level,
            ),
            list(emb),
            searchable=True,
        )
        count += 1

    # Browse-only (non-root) entries — emb_store key only.
    for old_nid, emb in emb_store_snapshot.items():
        if old_nid in searchable_old_ids:
            continue
        if not (old_nid == src_tree_id or old_nid.startswith(old_prefix)):
            continue
        target_idx.add_node(
            NodeEntry(
                node_id=_rekey(old_nid),
                tree_id=dst_tree_id,
                level="Ln",
            ),
            list(emb),
            searchable=False,
        )
        count += 1

    return count


# ── main entry point ──────────────────────────────────────────────────────────

def merge_user_forests(
    *,
    sources: list["UserForest"],
    target: "UserForest",
) -> ForestMergeResult:
    """Merge every ``UserForest`` in ``sources`` into the (fresh) ``target``.

    ``target`` must be an empty, freshly constructed forest — the merge
    populates its FactManager, routers, trees, cell_store, summary cache,
    session registry, and node_index in place, then calls ``target.save()``.
    """
    start = time.time()

    target_user_id = target.user_id
    source_user_ids = [s.user_id for s in sources]

    # ── 1. Facts ─────────────────────────────────────────────────────────────
    total_inserted = 0
    total_merged = 0
    source_remaps: list[dict[str, str]] = []
    for src in sources:
        write_result = target._fact_manager.merge_from(src._fact_manager)
        total_inserted += write_result.inserted_count
        total_merged += write_result.merged_count
        source_remaps.append(_build_fact_remap(src._fact_manager, target._fact_manager))

    # ── 2. Routers ───────────────────────────────────────────────────────────
    scene_remaps: list[dict[str, str]] = []
    for src in sources:
        target._tree_builder.entity_router.merge_from(src._tree_builder.entity_router)
        scene_remap = target._tree_builder.scene_router.merge_from(src._tree_builder.scene_router)
        scene_remaps.append(scene_remap)

    # Union session alias maps so downstream session lookups work.
    with target._tree_builder._lock:
        for src in sources:
            target._tree_builder._session_alias_map.update(src._tree_builder.session_alias_map)

    # Union summary caches — future deletion rebuilds reuse unchanged summaries.
    for src in sources:
        target._summary_cache.update(src._summary_cache)
    target._tree_builder.set_summary_cache(target._summary_cache)

    # ── 3. Group source trees by target tree_id ─────────────────────────────
    # Each member is (source_forest, fact_remap, source_tree, target_tree_id).
    merge_groups: dict[str, list[tuple["UserForest", dict[str, str], MemTree]]] = {}
    for src, fact_remap, scene_remap in zip(sources, source_remaps, scene_remaps):
        for src_tree in src._tree_builder.trees.values():
            if src_tree.tree_type == "scene":
                new_cluster_id = scene_remap.get(src_tree.tree_key, src_tree.tree_key)
                target_tree_id = f"tree:{new_cluster_id}"
            else:
                target_tree_id = src_tree.tree_id
            merge_groups.setdefault(target_tree_id, []).append((src, fact_remap, src_tree))

    # ── 4. For each group, copy or merge ────────────────────────────────────
    trees_copied = 0
    trees_merged = 0
    refreshed_tree_ids: set[str] = set()
    target_trees = target._tree_builder._trees

    def _seed_fact_catalog(tree: MemTree) -> None:
        for fid in tree.fact_ids_ordered:
            fact = target._fact_manager.get_fact(fid)
            if fact is not None:
                target._tree_builder._fact_catalog[fid] = fact
        for leaf in tree.session_leaves.values():
            for fid in leaf.fact_ids:
                fact = target._fact_manager.get_fact(fid)
                if fact is not None:
                    target._tree_builder._fact_catalog[fid] = fact

    def _install_clone(src_uf: "UserForest", src_tree: MemTree, target_tree_id: str,
                       fact_remap: dict[str, str]) -> MemTree:
        if src_tree.tree_id == target_tree_id:
            clone = copy.deepcopy(src_tree)
            clone.dirty_l0_node_ids = set()
            clone.dirty_internal_node_ids_by_level = {}
        else:
            clone = _rekey_tree(src_tree, target_tree_id)
        _remap_tree_fact_ids(clone, fact_remap)
        if clone.tree_type == "scene":
            clone.tree_key = target_tree_id[len("tree:"):] if target_tree_id.startswith("tree:") else clone.tree_key
        target_trees[target_tree_id] = clone
        target._tree_store.register(clone)
        _copy_node_index_entries(
            source_idx=src_uf._node_index,
            target_idx=target._node_index,
            src_tree_id=src_tree.tree_id,
            dst_tree_id=target_tree_id,
        )
        if clone.tree_type == "session":
            for cell_id in clone.session_leaves:
                if cell_id in src_uf._cell_store:
                    target._cell_store[cell_id] = src_uf._cell_store[cell_id]
        _seed_fact_catalog(clone)
        return clone

    for target_tree_id, members in merge_groups.items():
        # Sort biggest-leaves-first so members[0] is the trunk.
        members.sort(key=lambda m: _leaf_count(m[2]), reverse=True)

        trunk_uf, trunk_remap, trunk_src = members[0]
        trunk = _install_clone(trunk_uf, trunk_src, target_tree_id, trunk_remap)

        if len(members) == 1:
            trees_copied += 1
            continue

        # Merge path: insert leaves from every other member into the trunk.
        for src_uf, fact_remap, src_tree in members[1:]:
            if trunk.tree_type == "session":
                for cell_id, leaf in src_tree.session_leaves.items():
                    if cell_id in trunk.session_leaves:
                        continue
                    cell = src_uf._cell_store.get(cell_id)
                    if cell is None:
                        continue
                    remapped_fact_ids = [
                        fact_remap.get(fid, fid) for fid in leaf.fact_ids
                    ]
                    insert_cell(trunk, cell, remapped_fact_ids)
                    target._cell_store[cell_id] = cell
                    for fid in remapped_fact_ids:
                        fact = target._fact_manager.get_fact(fid)
                        if fact is not None:
                            target._tree_builder._fact_catalog[fid] = fact
            else:
                existing_fact_ids = set(trunk.fact_ids_ordered)
                for src_fid in src_tree.fact_ids_ordered:
                    tgt_fid = fact_remap.get(src_fid, src_fid)
                    if tgt_fid in existing_fact_ids:
                        continue
                    fact = target._fact_manager.get_fact(tgt_fid)
                    if fact is None:
                        continue
                    insert_fact(trunk, fact)
                    target._tree_builder._fact_catalog[tgt_fid] = fact
                    existing_fact_ids.add(tgt_fid)

        refreshed_tree_ids.add(target_tree_id)
        trees_merged += 1

    # ── 5. Flush dirty trees (merge path only) + refresh their node_index ──
    flushed: list[str] = []
    if refreshed_tree_ids:
        for tid in refreshed_tree_ids:
            tree = target_trees.get(tid)
            if tree is not None:
                rebuild_dirty_queues_from_flags(tree)
        flushed = list(target._tree_builder.flush(tree_ids=refreshed_tree_ids))
        target._refresh_node_index_for_trees(refreshed_tree_ids)

    # ── 6. Rebuild session registry from session trees so deletion works ──
    from src.forest.memforest import _populate_registry_from_trees
    _populate_registry_from_trees(target._registry, target._tree_store)

    # ── 7. Persist everything ───────────────────────────────────────────────
    target._rewire_query_pipeline()
    target._node_index.save(target._dir / "node_index")
    target.save()

    return ForestMergeResult(
        target_user_id=target_user_id,
        source_user_ids=source_user_ids,
        facts_inserted=total_inserted,
        facts_merged=total_merged,
        trees_copied=trees_copied,
        trees_merged=trees_merged,
        flushed_tree_ids=flushed,
        total_trees=len(merge_groups),
        elapsed_sec=time.time() - start,
    )
