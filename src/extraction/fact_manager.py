"""Persistent canonical fact manager for vNext MemoryItems."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.api.client import OpenAIChatClient, OpenAIEmbeddingClient
from src.config import MemForestConfig, load_default_config
from src.extraction.dedup import (
    EmbeddingTextClient,
    FactEquivalenceJudge,
    JsonChatClient,
    LLMFactEquivalenceJudge,
)
from src.utils.time import render_time_text
from src.utils.types import (
    DuplicateFactRecord,
    FactOccurrence,
    FactWriteResult,
    MemCell,
    ManagedFact,
    MemoryItem,
    SessionExtractionResult,
)


class FactManager:
    """Persistent canonical fact store backed by FAISS plus duplicate merge logs."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        storage_dir: str | Path,
        embedding_client: EmbeddingTextClient | None = None,
        judge: FactEquivalenceJudge | None = None,
        embedding_model_name: str = "",
        vector_dim: int = 1024,
        persist_on_write: bool = False,
        top_k: int = 8,
        similarity_threshold: float = 0.972,
        max_llm_pairs_per_item: int = 4,
        normalize_embeddings: bool = True,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.embedding_client = embedding_client
        self.judge = judge
        self.embedding_model_name = str(embedding_model_name or "")
        self.vector_dim = int(vector_dim)
        self.persist_on_write = bool(persist_on_write)
        self.top_k = max(1, int(top_k))
        self.similarity_threshold = float(similarity_threshold)
        self.max_llm_pairs_per_item = max(1, int(max_llm_pairs_per_item))
        self.normalize_embeddings = bool(normalize_embeddings)

        self.manifest_path = self.storage_dir / "manifest.json"
        self.facts_path = self.storage_dir / "facts.jsonl"
        self.index_path = self.storage_dir / "faiss.index"
        self.ids_path = self.storage_dir / "faiss_ids.json"
        self.duplicates_path = self.storage_dir / "duplicates.jsonl"

        self._lock = threading.RLock()
        self._facts: dict[str, ManagedFact] = {}
        self._faiss_ids: list[str] = []
        self._duplicate_records: list[DuplicateFactRecord] = []
        self._exact_index: dict[str, str] = {}
        self._index = faiss.IndexFlatIP(self.vector_dim)

        self._ensure_dirs()
        self.load()

    @classmethod
    def from_dir(
        cls,
        storage_dir: str | Path,
        *,
        embedding_client: EmbeddingTextClient | None = None,
        judge: FactEquivalenceJudge | None = None,
        config: MemForestConfig | None = None,
    ) -> "FactManager":
        cfg = config or load_default_config()
        manifest_path = Path(storage_dir) / "manifest.json"
        vector_dim = int(cfg.api.embedding.dimension)
        embedding_model_name = str(cfg.api.embedding.model_name)
        if manifest_path.exists():
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            vector_dim = int(raw.get("vector_dimension", vector_dim))
            embedding_model_name = str(raw.get("embedding_model_name", embedding_model_name))
        return cls(
            storage_dir=storage_dir,
            embedding_client=embedding_client,
            judge=judge,
            embedding_model_name=embedding_model_name,
            vector_dim=vector_dim,
            persist_on_write=False,
            top_k=int(cfg.extraction.fact_manager.top_k),
            similarity_threshold=float(cfg.extraction.fact_manager.similarity_threshold),
            max_llm_pairs_per_item=int(cfg.extraction.fact_manager.max_llm_pairs_per_item),
            normalize_embeddings=bool(cfg.extraction.fact_manager.normalize_embeddings),
        )

    @classmethod
    def from_config(
        cls,
        config: MemForestConfig,
        *,
        embedding_client: EmbeddingTextClient | None = None,
        judge: FactEquivalenceJudge | None = None,
        chat_client: JsonChatClient | None = None,
    ) -> "FactManager":
        embedding_client = embedding_client or OpenAIEmbeddingClient(config.api.embedding)
        if judge is None:
            judge_client = chat_client or OpenAIChatClient(config.api.llm)
            judge = LLMFactEquivalenceJudge(
                chat_client=judge_client,
                model_name=config.api.llm.model_name,
                temperature=0.0,
                max_tokens=200,
                top_p=0.2,
            )
        return cls(
            storage_dir=config.extraction.fact_manager.storage_dir,
            embedding_client=embedding_client,
            judge=judge,
            embedding_model_name=config.api.embedding.model_name,
            vector_dim=config.api.embedding.dimension,
            persist_on_write=config.extraction.fact_manager.persist_on_write,
            top_k=config.extraction.fact_manager.top_k,
            similarity_threshold=config.extraction.fact_manager.similarity_threshold,
            max_llm_pairs_per_item=config.extraction.fact_manager.max_llm_pairs_per_item,
            normalize_embeddings=config.extraction.fact_manager.normalize_embeddings,
        )

    def add_session_result(self, result: SessionExtractionResult, *, persist: bool = False) -> FactWriteResult:
        cell_fallbacks = {
            cell.cell_id: cell
            for cell in result.cells
        }
        return self.add_memory_items(
            self._apply_cell_time_fallbacks(result.memory_items, cell_fallbacks),
            persist=persist,
        )

    def add_memory_items(self, items: list[MemoryItem], *, persist: bool = False) -> FactWriteResult:
        with self._lock:
            valid_items = [item for item in items if str(item.fact_text or "").strip()]
            skipped_count = len(items) - len(valid_items)
            if not valid_items:
                return FactWriteResult(
                    input_count=len(items),
                    inserted_count=0,
                    merged_count=0,
                    skipped_count=skipped_count,
                    inserted_facts=[],
                    canonical_fact_ids=[],
                    duplicate_records=[],
                )

            groups = self._group_exact_duplicates(valid_items)
            representative_items = [group[0] for group in groups]
            if representative_items and self.embedding_client is None:
                raise RuntimeError("FactManager requires an embedding client to insert new canonical facts.")

            vectors = self._embed_items(representative_items) if representative_items else []
            inserted_count = 0
            inserted_facts: list[ManagedFact] = []
            merged_count = 0
            duplicate_records: list[DuplicateFactRecord] = []
            touched_fact_ids: list[str] = []

            # --- Phase A: exact match + FAISS candidate collection ---
            exact_matched: dict[int, str] = {}          # group_index → canonical_fact_id
            # (group_index, candidate_fact_id_or_group_index, similarity, is_intra_batch)
            needs_llm: list[tuple[int, str | int, float, bool]] = []
            no_existing_match: list[int] = []           # group indices with no existing-index candidates

            for group_index, group in enumerate(groups):
                representative = group[0]
                normalized_key = self._normalize_fact_text(representative.fact_text)
                canonical_fact_id = self._exact_index.get(normalized_key)
                if canonical_fact_id:
                    exact_matched[group_index] = canonical_fact_id
                    continue

                vector = vectors[group_index]
                candidates = self._find_candidate_pairs(vector)
                if candidates:
                    for fact_id, similarity in candidates[:self.max_llm_pairs_per_item]:
                        needs_llm.append((group_index, fact_id, similarity, False))
                else:
                    no_existing_match.append(group_index)

            # Intra-batch dedup: pairwise similarity among unmatched groups
            if len(no_existing_match) > 1:
                batch_vecs = np.stack([vectors[gi] for gi in no_existing_match])
                temp_index = faiss.IndexFlatIP(self.vector_dim)
                temp_index.add(batch_vecs)
                k = min(self.top_k, len(no_existing_match))
                scores, indices = temp_index.search(batch_vecs, k)
                seen_intra: set[tuple[int, int]] = set()
                for row, gi_a in enumerate(no_existing_match):
                    pairs_added = 0
                    for col in range(k):
                        j = int(indices[row][col])
                        gi_b = no_existing_match[j]
                        if gi_a >= gi_b:
                            continue
                        sim = float(scores[row][col])
                        if sim < self.similarity_threshold:
                            continue
                        pair_key = (gi_a, gi_b)
                        if pair_key in seen_intra:
                            continue
                        seen_intra.add(pair_key)
                        needs_llm.append((gi_a, gi_b, sim, True))
                        pairs_added += 1
                        if pairs_added >= self.max_llm_pairs_per_item:
                            break

            # --- Phase B: batch LLM judge (concurrent) ---
            llm_results: dict[int, tuple[str, float]] = {}        # group_index → (existing fact_id, similarity)
            intra_merges: list[tuple[int, int, float]] = []  # (gi_a, gi_b, similarity) — a absorbs b
            if needs_llm and self.judge is not None:
                pairs = []
                traces = []
                for gi, target, _sim, is_intra in needs_llm:
                    if is_intra:
                        pairs.append((groups[gi][0].fact_text, groups[target][0].fact_text))
                        traces.append({"request_id": f"fact-manager:intra:{gi}:{target}"})
                    else:
                        pairs.append((self._facts[target].fact_text, groups[gi][0].fact_text))
                        traces.append({"request_id": f"fact-manager:{groups[gi][0].item_id}:{target}"})
                if hasattr(self.judge, "are_equivalent_batch"):
                    batch_results = self.judge.are_equivalent_batch(
                        pairs, traces=traces, max_workers=64,
                    )
                else:
                    batch_results = [
                        self.judge.are_equivalent(a, b, trace=t)
                        for (a, b), t in zip(pairs, traces)
                    ]
                seen_groups: set[int] = set()
                for (gi, target, similarity, is_intra), (equivalent, _preferred) in zip(needs_llm, batch_results):
                    if not equivalent:
                        continue
                    if is_intra:
                        intra_merges.append((gi, target, similarity))
                    else:
                        if gi not in seen_groups:
                            llm_results[gi] = (target, similarity)
                            seen_groups.add(gi)

            # Resolve intra-batch merges via union-find
            intra_uf: dict[int, int] = {}  # group_index → root group_index
            for gi_a, gi_b, _sim in intra_merges:
                root_a = gi_a
                while intra_uf.get(root_a, root_a) != root_a:
                    root_a = intra_uf[root_a]
                root_b = gi_b
                while intra_uf.get(root_b, root_b) != root_b:
                    root_b = intra_uf[root_b]
                if root_a != root_b:
                    intra_uf[root_b] = root_a

            # Build merged group mapping: secondary → primary group_index
            intra_merged_into: dict[int, int] = {}  # secondary gi → primary gi
            for gi_a, gi_b, _sim in intra_merges:
                root_a = gi_a
                while intra_uf.get(root_a, root_a) != root_a:
                    root_a = intra_uf[root_a]
                root_b = gi_b
                while intra_uf.get(root_b, root_b) != root_b:
                    root_b = intra_uf[root_b]
                if root_a == root_b and gi_b != root_b:
                    intra_merged_into[gi_b] = root_b
                elif root_a == root_b and gi_a != root_a:
                    intra_merged_into[gi_a] = root_a
                # gi_b merges into gi_a (already ensured gi_a < gi_b in collection)
                if gi_b not in intra_merged_into and gi_a not in intra_merged_into:
                    intra_merged_into[gi_b] = gi_a

            # --- Phase C: apply decisions ---
            for group_index, group in enumerate(groups):
                if group_index in exact_matched:
                    canonical_fact_id = exact_matched[group_index]
                    touched_fact_ids.append(canonical_fact_id)
                    merged_count += len(group)
                    self._merge_items_into_fact(
                        canonical_fact_id,
                        group,
                        similarity=1.0,
                        reason="exact_canonical_match",
                        duplicate_records=duplicate_records,
                        include_first=True,
                    )
                elif group_index in llm_results:
                    matched_fact_id, matched_similarity = llm_results[group_index]
                    touched_fact_ids.append(matched_fact_id)
                    merged_count += len(group)
                    self._merge_items_into_fact(
                        matched_fact_id,
                        group,
                        similarity=matched_similarity,
                        reason="embedding_llm_match",
                        duplicate_records=duplicate_records,
                        include_first=True,
                    )
                elif group_index in intra_merged_into:
                    # Will be merged when primary group is inserted — skip here, handled below
                    pass
                else:
                    # Insert as new canonical fact (possibly absorbing intra-batch duplicates)
                    representative = group[0]
                    normalized_key = self._normalize_fact_text(representative.fact_text)
                    vector = vectors[group_index]
                    # Collect intra-batch secondary groups
                    absorbed_groups = [group]
                    for sec_gi, pri_gi in list(intra_merged_into.items()):
                        if pri_gi == group_index:
                            absorbed_groups.append(groups[sec_gi])
                    all_items_for_fact = [item for g in absorbed_groups for item in g]
                    new_fact = self._build_managed_fact(representative, all_items_for_fact)
                    self._facts[new_fact.fact_id] = new_fact
                    self._faiss_ids.append(new_fact.fact_id)
                    self._index.add(vector.reshape(1, -1))
                    self._exact_index[normalized_key] = new_fact.fact_id
                    touched_fact_ids.append(new_fact.fact_id)
                    inserted_count += 1
                    inserted_facts.append(new_fact)
                    # Record duplicates for secondary items
                    extra_items = all_items_for_fact[1:]
                    if extra_items:
                        merged_count += len(extra_items)
                        for duplicate_item in extra_items:
                            duplicate_records.append(
                                DuplicateFactRecord(
                                    duplicate_item_id=duplicate_item.item_id,
                                    canonical_fact_id=new_fact.fact_id,
                                    similarity=1.0,
                                    reason="exact_batch_match" if duplicate_item in group else "embedding_llm_batch_match",
                                    source_session_id=duplicate_item.session_id,
                                    source_cell_id=duplicate_item.cell_id,
                                    fact_text=duplicate_item.fact_text,
                                )
                            )

            self._duplicate_records.extend(duplicate_records)
            if persist or self.persist_on_write:
                self.save()
            return FactWriteResult(
                input_count=len(items),
                inserted_count=inserted_count,
                merged_count=merged_count,
                skipped_count=skipped_count,
                inserted_facts=inserted_facts,
                canonical_fact_ids=self._ordered_unique(touched_fact_ids),
                duplicate_records=duplicate_records,
            )

    def merge_from(
        self,
        other: "FactManager | str | Path",
        *,
        persist: bool = False,
    ) -> FactWriteResult:
        """Merge all facts from *other* into this FactManager with automatic dedup.

        ``other`` can be a :class:`FactManager` instance **or** a path to a
        storage directory (in which case a read-only FactManager is loaded from
        that path — no embedding/judge clients are needed on the source side).

        Internally the source facts are converted to :class:`MemoryItem` objects
        (one per occurrence) and fed through :meth:`add_memory_items`, so the
        full exact-match → embedding → LLM-judge pipeline runs automatically.
        """
        if isinstance(other, (str, Path)):
            other = FactManager(
                storage_dir=other,
                embedding_client=None,
                judge=None,
                vector_dim=self.vector_dim,
            )
        items = self._facts_to_memory_items(other.iter_facts())
        return self.add_memory_items(items, persist=persist)

    @staticmethod
    def merge_dirs(
        sources: list[str | Path],
        target: "FactManager",
        *,
        persist: bool = False,
    ) -> FactWriteResult:
        """Merge facts from multiple storage directories into *target*.

        All source facts are collected first, then inserted in a single
        :meth:`add_memory_items` call so that LLM-judge pairs across all
        sources are batched together.
        """
        all_items: list[MemoryItem] = []
        for src in sources:
            src_fm = FactManager(
                storage_dir=src,
                embedding_client=None,
                judge=None,
                vector_dim=target.vector_dim,
            )
            all_items.extend(target._facts_to_memory_items(src_fm.iter_facts()))
        return target.add_memory_items(all_items, persist=persist)

    @staticmethod
    def _facts_to_memory_items(facts: list[ManagedFact]) -> list[MemoryItem]:
        """Expand canonical facts into MemoryItem list (one per occurrence)."""
        items: list[MemoryItem] = []
        for fact in facts:
            if not fact.occurrences:
                # Fact with no occurrences — create a synthetic item
                items.append(MemoryItem(
                    item_id=fact.fact_id,
                    session_id=fact.first_session_id,
                    cell_id=fact.first_cell_id,
                    fact_text=fact.fact_text,
                    source_turn_ids=[],
                    source_spans=[],
                    participants=[],
                    origin=fact.origin,
                    semantic_role=fact.semantic_role,
                    entities=list(fact.entities),
                    topics=list(fact.topics),
                    time_text="",
                    time_start=fact.time_start,
                    time_end=fact.time_end,
                    attribute_keys=list(fact.attribute_keys),
                    domain_keys=list(fact.domain_keys),
                    collection_keys=list(fact.collection_keys),
                    detail_level=fact.detail_level,
                    confidence=fact.confidence,
                    metadata=dict(fact.metadata),
                ))
                continue
            for occ in fact.occurrences:
                items.append(MemoryItem(
                    item_id=occ.item_id,
                    session_id=occ.session_id,
                    cell_id=occ.cell_id,
                    fact_text=fact.fact_text,
                    source_turn_ids=list(occ.source_turn_ids),
                    source_spans=list(occ.source_spans),
                    participants=list(occ.participants),
                    origin=fact.origin,
                    semantic_role=fact.semantic_role,
                    entities=list(fact.entities),
                    topics=list(fact.topics),
                    time_text=occ.time_text,
                    time_start=occ.time_start,
                    time_end=occ.time_end,
                    attribute_keys=list(fact.attribute_keys),
                    domain_keys=list(fact.domain_keys),
                    collection_keys=list(fact.collection_keys),
                    detail_level=fact.detail_level,
                    confidence=fact.confidence,
                    metadata=dict(occ.metadata),
                ))
        return items

    def search_similar_fact_text(
        self,
        text: str,
        *,
        top_k: int | None = None,
    ) -> list[tuple[ManagedFact, float]]:
        with self._lock:
            if self._index.ntotal == 0:
                return []
            if self.embedding_client is None:
                raise RuntimeError("FactManager requires an embedding client for similarity search.")
            vector = self._embed_texts([text])[0]
            return self._search_by_vector_locked(vector, top_k)

    def search_similar_fact_by_vector(
        self,
        vector: "np.ndarray | list[float]",
        *,
        top_k: int | None = None,
    ) -> list[tuple[ManagedFact, float]]:
        """Search by a pre-computed query embedding. Used by ForestRetriever's
        BU path to avoid re-embedding the question.

        The vector must match the FAISS dimension; if `normalize_embeddings`
        is True it will be re-normalized before search.
        """
        import numpy as _np
        with self._lock:
            if self._index.ntotal == 0:
                return []
            arr = _np.asarray(vector, dtype=_np.float32).reshape(-1)
            if arr.shape[0] != self.vector_dim:
                raise ValueError(
                    f"query vector dim {arr.shape[0]} != FactManager vector_dim {self.vector_dim}"
                )
            if self.normalize_embeddings:
                n = float(_np.linalg.norm(arr))
                if n > 0:
                    arr = arr / n
            return self._search_by_vector_locked(arr, top_k)

    def _search_by_vector_locked(
        self,
        vector: "np.ndarray",
        top_k: int | None,
    ) -> list[tuple[ManagedFact, float]]:
        """Inner search; caller must hold self._lock and ensure ntotal > 0."""
        k = min(max(1, int(top_k or self.top_k)), self._index.ntotal)
        scores, indices = self._index.search(vector.reshape(1, -1), k)
        out: list[tuple[ManagedFact, float]] = []
        seen: set[str] = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            fact_id = self._faiss_ids[int(idx)]
            if fact_id in seen or fact_id not in self._facts:
                continue
            seen.add(fact_id)
            out.append((self._facts[fact_id], float(score)))
        return out

    def get_fact(self, fact_id: str) -> ManagedFact | None:
        with self._lock:
            return self._facts.get(str(fact_id))

    def iter_facts(self) -> list[ManagedFact]:
        with self._lock:
            return [self._facts[fact_id] for fact_id in self._faiss_ids if fact_id in self._facts]

    def save(self) -> None:
        with self._lock:
            self._ensure_dirs()
            manifest = {
                "schema_version": self.SCHEMA_VERSION,
                "embedding_model_name": self.embedding_model_name,
                "vector_dimension": self.vector_dim,
                "faiss_metric": "ip",
            }
            self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.facts_path.open("w", encoding="utf-8") as fh:
                for fact in self.iter_facts():
                    fh.write(json.dumps(asdict(fact), ensure_ascii=False) + "\n")
            faiss.write_index(self._index, str(self.index_path))
            self.ids_path.write_text(json.dumps(self._faiss_ids, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.duplicates_path.open("w", encoding="utf-8") as fh:
                for record in self._duplicate_records:
                    fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def load(self) -> None:
        with self._lock:
            self._facts = {}
            self._faiss_ids = []
            self._duplicate_records = []
            self._exact_index = {}
            self._index = faiss.IndexFlatIP(self.vector_dim)

            if self.manifest_path.exists():
                raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                self.embedding_model_name = str(raw.get("embedding_model_name", self.embedding_model_name))
                self.vector_dim = int(raw.get("vector_dimension", self.vector_dim))
                self._index = faiss.IndexFlatIP(self.vector_dim)

            if self.facts_path.exists():
                with self.facts_path.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        payload = json.loads(line)
                        occurrences = [
                            FactOccurrence(**occurrence)
                            for occurrence in payload.get("occurrences", [])
                        ]
                        payload["occurrences"] = occurrences
                        fact = ManagedFact(**payload)
                        self._facts[fact.fact_id] = fact
                        self._exact_index[self._normalize_fact_text(fact.fact_text)] = fact.fact_id

            if self.ids_path.exists():
                self._faiss_ids = list(json.loads(self.ids_path.read_text(encoding="utf-8")))

            if self.index_path.exists():
                self._index = faiss.read_index(str(self.index_path))

            if self.duplicates_path.exists():
                with self.duplicates_path.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        self._duplicate_records.append(DuplicateFactRecord(**json.loads(line)))

    def _find_candidate_pairs(
        self,
        vector: np.ndarray,
    ) -> list[tuple[str, float]]:
        """FAISS search only — return candidate (fact_id, similarity) pairs above threshold."""
        if self._index.ntotal == 0:
            return []
        k = min(self.top_k, self._index.ntotal)
        scores, indices = self._index.search(vector.reshape(1, -1), k)
        candidates: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            similarity = float(score)
            if similarity < self.similarity_threshold:
                continue
            fact_id = self._faiss_ids[int(idx)]
            if fact_id in self._facts:
                candidates.append((fact_id, similarity))
        return candidates

    def _merge_items_into_fact(
        self,
        fact_id: str,
        items: list[MemoryItem],
        *,
        similarity: float,
        reason: str,
        duplicate_records: list[DuplicateFactRecord],
        include_first: bool,
    ) -> None:
        fact = self._facts[fact_id]
        new_occurrences = list(fact.occurrences)
        append_items = items if include_first else items[1:]
        for item in append_items:
            new_occurrences.append(self._item_to_occurrence(item))
            duplicate_records.append(
                DuplicateFactRecord(
                    duplicate_item_id=item.item_id,
                    canonical_fact_id=fact_id,
                    similarity=float(similarity),
                    reason=reason,
                    source_session_id=item.session_id,
                    source_cell_id=item.cell_id,
                    fact_text=item.fact_text,
                )
            )
        merged_metadata = dict(fact.metadata)
        alt_texts = list(merged_metadata.get("alternate_fact_texts", []))
        for item in items:
            if item.fact_text != fact.fact_text and item.fact_text not in alt_texts:
                alt_texts.append(item.fact_text)
        if alt_texts:
            merged_metadata["alternate_fact_texts"] = alt_texts
        self._facts[fact_id] = replace(
            fact,
            entities=self._ordered_unique([*fact.entities, *[e for item in items for e in item.entities]]),
            topics=self._ordered_unique([*fact.topics, *[t for item in items for t in item.topics]]),
            attribute_keys=self._ordered_unique([*fact.attribute_keys, *[k for item in items for k in item.attribute_keys]]),
            domain_keys=self._ordered_unique([*fact.domain_keys, *[k for item in items for k in item.domain_keys]]),
            collection_keys=self._ordered_unique([*fact.collection_keys, *[k for item in items for k in item.collection_keys]]),
            confidence=max([fact.confidence, *[float(item.confidence) for item in items]]),
            occurrences=new_occurrences,
            metadata=merged_metadata,
            time_start=_min_time(fact.time_start, *[occ.time_start for occ in new_occurrences]),
            time_end=_max_time(fact.time_end, *[occ.time_end for occ in new_occurrences]),
        )

    def _build_managed_fact(self, representative: MemoryItem, group: list[MemoryItem]) -> ManagedFact:
        fact_id = self._build_fact_id(representative)
        occurrences = [self._item_to_occurrence(item) for item in group]
        metadata = dict(representative.metadata)
        alt_texts = [item.fact_text for item in group[1:] if item.fact_text != representative.fact_text]
        if alt_texts:
            metadata["alternate_fact_texts"] = self._ordered_unique(alt_texts)
        return ManagedFact(
            fact_id=fact_id,
            fact_text=representative.fact_text,
            embedding_id=fact_id,
            origin=representative.origin,
            semantic_role=representative.semantic_role,
            entities=list(representative.entities),
            topics=list(representative.topics),
            attribute_keys=list(representative.attribute_keys),
            domain_keys=list(representative.domain_keys),
            collection_keys=list(representative.collection_keys),
            detail_level=representative.detail_level,
            confidence=float(representative.confidence),
            first_session_id=representative.session_id,
            first_cell_id=representative.cell_id,
            occurrences=occurrences,
            metadata=metadata,
            time_start=_min_time(*[occ.time_start for occ in occurrences]),
            time_end=_max_time(*[occ.time_end for occ in occurrences]),
        )

    def _item_to_occurrence(self, item: MemoryItem) -> FactOccurrence:
        return FactOccurrence(
            session_id=item.session_id,
            cell_id=item.cell_id,
            item_id=item.item_id,
            source_turn_ids=list(item.source_turn_ids),
            source_spans=list(item.source_spans),
            participants=list(item.participants),
            time_text=item.time_text,
            time_start=item.time_start,
            time_end=item.time_end,
            metadata=dict(item.metadata),
        )

    def _group_exact_duplicates(self, items: list[MemoryItem]) -> list[list[MemoryItem]]:
        groups: dict[str, list[MemoryItem]] = {}
        for item in items:
            groups.setdefault(self._normalize_fact_text(item.fact_text), []).append(item)
        ordered_groups: list[list[MemoryItem]] = []
        for _, group in sorted(groups.items(), key=lambda row: min(item.item_id for item in row[1])):
            ordered_groups.append(sorted(group, key=self._representative_sort_key, reverse=True))
        return ordered_groups

    def _embed_items(self, items: list[MemoryItem]) -> list[np.ndarray]:
        return self._embed_texts([item.fact_text for item in items])

    def _embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        if self.embedding_client is None:
            raise RuntimeError("FactManager requires an embedding client for embedding operations.")
        vectors = self.embedding_client.embed_texts(texts)
        out: list[np.ndarray] = []
        for vector in vectors:
            arr = np.asarray(vector, dtype="float32")
            if arr.ndim != 1:
                raise ValueError("Embedding vector must be 1-dimensional.")
            if self.normalize_embeddings:
                arr = self._normalize_vector(arr)
            out.append(arr)
        return out

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            return vector.astype("float32")
        return (vector / norm).astype("float32")

    def _representative_sort_key(self, item: MemoryItem) -> tuple[float, int, int, int]:
        return (
            float(item.confidence),
            len(item.source_spans),
            sum(1 for ch in item.fact_text if ch.isdigit()),
            len(item.fact_text),
        )

    def _build_fact_id(self, item: MemoryItem) -> str:
        raw = f"{item.session_id}|{item.cell_id}|{item.item_id}|{item.fact_text}"
        return f"fact_{hashlib.md5(raw.encode('utf-8')).hexdigest()[:16]}"

    def _normalize_fact_text(self, text: str) -> str:
        return " ".join(str(text or "").strip().lower().split())

    def _apply_cell_time_fallbacks(
        self,
        items: list[MemoryItem],
        cell_fallbacks: dict[str, MemCell],
    ) -> list[MemoryItem]:
        hydrated: list[MemoryItem] = []
        for item in items:
            if item.time_start is not None or item.time_end is not None:
                hydrated.append(item)
                continue
            cell = cell_fallbacks.get(item.cell_id)
            if cell is None:
                hydrated.append(item)
                continue
            hydrated.append(replace(
                item,
                time_text=item.time_text or _render_time_range_text(cell.time_start, cell.time_end),
                time_start=float(cell.time_start),
                time_end=float(cell.time_end),
            ))
        return hydrated

    def _ensure_dirs(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _ordered_unique(self, values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out


def _min_time(*values: float | None) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return min(clean) if clean else None


def _max_time(*values: float | None) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return max(clean) if clean else None


def _render_time_range_text(time_start: float, time_end: float) -> str:
    start_text = render_time_text(time_start)
    end_text = render_time_text(time_end)
    if float(time_start) == float(time_end):
        return start_text
    return f"{start_text} to {end_text}"
