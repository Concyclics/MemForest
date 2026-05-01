"""Data-driven online scene routing for scene tree construction."""

from __future__ import annotations

import json
import math
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from src.build.tree_types import SceneBootstrapState, SceneCluster
from src.config.tree_config import SceneRouterConfig

if TYPE_CHECKING:
    from src.api.client import OpenAIEmbeddingClient
    from src.utils.types import ManagedFact


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return dot / (na * nb)


def _update_centroid(centroid: list[float], new_vec: list[float], n: int) -> list[float]:
    alpha = 1.0 / (1.0 + n)
    return [c * (1.0 - alpha) + v * alpha for c, v in zip(centroid, new_vec)]


class SceneRouter:
    """Routes facts into long-range scene clusters without fixed domain seeds."""

    def __init__(
        self,
        *,
        embedding_client: "OpenAIEmbeddingClient",
        config: SceneRouterConfig,
    ) -> None:
        self._embedding_client = embedding_client
        self._config = config
        self._clusters: dict[str, SceneCluster] = {}
        self._lock = threading.RLock()
        self._insert_count = 0
        self._pending_splits: list[str] = []
        self._bootstrap_state = SceneBootstrapState(initialized=False, buffered_fact_ids=[], cluster_count=0)
        self._bootstrap_buffer: list["ManagedFact"] = []
        # Routing-text → embedding cache.  Avoids re-embedding the same
        # content across bootstrap / assign_many calls and across
        # incremental ingest rounds.
        self._embedding_cache: dict[str, list[float]] = {}

    def bootstrap(self, facts: list["ManagedFact"]) -> None:
        """Initialise scene prototypes from the current input facts."""
        with self._lock:
            if self._bootstrap_state.initialized:
                return
        candidate_facts = [f for f in facts if f.fact_text.strip()]
        if len(candidate_facts) < self._config.bootstrap_min_facts:
            with self._lock:
                self._bootstrap_buffer = list(candidate_facts)
                self._bootstrap_state = SceneBootstrapState(
                    initialized=False,
                    buffered_fact_ids=[f.fact_id for f in candidate_facts],
                    cluster_count=0,
                )
            return
        sample = candidate_facts[: self._config.bootstrap_max_facts]
        embeddings = self._embed_routing_texts(sample)
        with self._lock:
            self._bootstrap_locked(sample, embeddings)

    def assign(self, fact: "ManagedFact") -> list[str]:
        embedding = self._embed_routing_texts([fact])[0]
        with self._lock:
            if not self._bootstrap_state.initialized:
                self._bootstrap_buffer.append(fact)
                self._bootstrap_state = SceneBootstrapState(
                    initialized=False,
                    buffered_fact_ids=[f.fact_id for f in self._bootstrap_buffer],
                    cluster_count=len(self._clusters),
                )
                if len(self._bootstrap_buffer) >= self._config.bootstrap_min_facts:
                    self._bootstrap_locked(
                        self._bootstrap_buffer,
                        self._embed_routing_texts(self._bootstrap_buffer),
                    )
            cluster_ids = self._assign_locked(fact, embedding)
            self._insert_count += 1
            return cluster_ids

    def assign_many(self, facts: list["ManagedFact"]) -> dict[str, list[str]]:
        """Batch-route a set of facts using one embedding request.

        After assignment, automatically merges same-domain clusters whose
        centroids are above theta_merge, reducing fragmentation.
        """
        if not facts:
            return {}
        embeddings = self._embed_routing_texts(facts)
        assignments: dict[str, list[str]] = {}
        with self._lock:
            if not self._bootstrap_state.initialized:
                self._bootstrap_buffer.extend(facts)
                self._bootstrap_state = SceneBootstrapState(
                    initialized=False,
                    buffered_fact_ids=[f.fact_id for f in self._bootstrap_buffer],
                    cluster_count=len(self._clusters),
                )
                if len(self._bootstrap_buffer) >= self._config.bootstrap_min_facts:
                    self._bootstrap_locked(
                        self._bootstrap_buffer,
                        self._embed_routing_texts(self._bootstrap_buffer),
                    )
            for fact, embedding in zip(facts, embeddings):
                assignments[fact.fact_id] = self._assign_locked(fact, embedding)
                self._insert_count += 1
        # Post-assignment merge: collapse same-domain clusters that ended
        # up very similar after absorbing their facts.
        merged = self.merge_clusters()
        if merged:
            # Rewrite assignments to use surviving cluster ids.
            alive = set(self.all_cluster_ids())
            for fact_id in assignments:
                assignments[fact_id] = [cid for cid in assignments[fact_id] if cid in alive]
        return assignments

    def merge_check(self) -> list[tuple[str, str]]:
        """Find cluster pairs within the same domain that are similar enough to merge."""
        pairs: list[tuple[str, str]] = []
        with self._lock:
            # Group by domain gate to avoid O(n^2) across all clusters.
            by_domain: dict[str, list[str]] = {}
            for cid, cluster in self._clusters.items():
                by_domain.setdefault(cluster.scope_key, []).append(cid)
            for domain_ids in by_domain.values():
                for i in range(len(domain_ids)):
                    for j in range(i + 1, len(domain_ids)):
                        left = self._clusters[domain_ids[i]]
                        right = self._clusters[domain_ids[j]]
                        combined = left.fact_count + right.fact_count
                        if combined > self._config.max_cluster_size:
                            continue  # merging would exceed cap
                        similarity = _cosine_similarity(left.centroid, right.centroid)
                        if similarity >= self._config.theta_merge:
                            pairs.append((left.cluster_id, right.cluster_id))
        return pairs

    def merge_clusters(self, pairs: list[tuple[str, str]] | None = None) -> int:
        """Merge cluster pairs (smaller into larger).  Returns merge count.

        If *pairs* is None, runs merge_check() first.
        """
        if pairs is None:
            pairs = self.merge_check()
        if not pairs:
            return 0
        merged_count = 0
        with self._lock:
            absorbed: set[str] = set()  # clusters already merged away
            for left_id, right_id in pairs:
                if left_id in absorbed or right_id in absorbed:
                    continue
                left = self._clusters.get(left_id)
                right = self._clusters.get(right_id)
                if left is None or right is None:
                    continue
                # Merge smaller into larger.
                if left.fact_count >= right.fact_count:
                    keeper, donor = left, right
                else:
                    keeper, donor = right, left
                _merge_cluster_into(keeper, donor)
                self._clusters.pop(donor.cluster_id, None)
                absorbed.add(donor.cluster_id)
                merged_count += 1
            if merged_count:
                self._bootstrap_state = SceneBootstrapState(
                    initialized=self._bootstrap_state.initialized,
                    buffered_fact_ids=list(self._bootstrap_state.buffered_fact_ids),
                    cluster_count=len(self._clusters),
                )
        return merged_count

    def merge_from(
        self,
        other: "SceneRouter | str | Path",
    ) -> dict[str, str]:
        """Merge all clusters from *other* into this router.

        For each source cluster, find the best matching target cluster in the
        same domain (by centroid cosine similarity).  If similarity ≥ theta_merge
        and the combined size would not exceed max_cluster_size, merge the source
        cluster into the target cluster.  Otherwise, add it as a new cluster.

        ``other`` can be a :class:`SceneRouter` instance or a path to a saved
        router JSON file.

        Returns a mapping ``{source_cluster_id: target_cluster_id}`` so the
        caller can remap fact→cluster assignments from the source memory base.
        """
        if isinstance(other, (str, Path)):
            source_clusters = self._load_clusters_from_path(Path(other))
        else:
            with other._lock:
                source_clusters = list(other._clusters.values())

        remap: dict[str, str] = {}
        with self._lock:
            for src in source_clusters:
                # Find best match in the same domain.
                best_id: str | None = None
                best_sim = -1.0
                for tgt in self._clusters.values():
                    if tgt.scope_key != src.scope_key:
                        continue
                    if tgt.fact_count + src.fact_count > self._config.max_cluster_size:
                        continue
                    sim = _cosine_similarity(tgt.centroid, src.centroid)
                    if sim > best_sim:
                        best_sim = sim
                        best_id = tgt.cluster_id

                if best_id is not None and best_sim >= self._config.theta_merge:
                    # Merge source into existing target cluster.
                    _merge_cluster_into(self._clusters[best_id], src)
                    remap[src.cluster_id] = best_id
                else:
                    # Add as a new cluster (with a fresh ID to avoid collisions).
                    new_id = f"scene:{src.label}_{uuid.uuid4().hex[:8]}"
                    new_cluster = SceneCluster(
                        cluster_id=new_id,
                        label=src.label,
                        scope_key=src.scope_key,
                        centroid=list(src.centroid),
                        fact_count=src.fact_count,
                        fact_ids=list(src.fact_ids),
                        created_at=src.created_at,
                        updated_at=src.updated_at,
                        latest_fact_time=src.latest_fact_time,
                        parent_cluster_id=src.parent_cluster_id,
                        topic_histogram=dict(src.topic_histogram),
                        domain_histogram=dict(src.domain_histogram),
                    )
                    self._clusters[new_id] = new_cluster
                    remap[src.cluster_id] = new_id

            self._bootstrap_state = SceneBootstrapState(
                initialized=True,
                buffered_fact_ids=[],
                cluster_count=len(self._clusters),
            )
        return remap

    def relabel_clusters(
        self,
        *,
        chat_client,
        model_name: str,
        fact_lookup,
        max_samples_per_cluster: int = 6,
        max_inflight: int = 32,
        min_fact_count: int = 2,
        temperature: float = 0.0,
        max_tokens: int = 64,
        timeout: float = 30.0,
    ) -> dict[str, dict]:
        """Replace heuristic cluster labels with LLM-generated short phrases.

        For each cluster with at least ``min_fact_count`` facts, sample up to
        ``max_samples_per_cluster`` fact texts and ask the model for a short
        snake_case label (2-4 words).  Single-fact clusters keep the heuristic
        label — an LLM call there is pure overhead.

        Parameters
        ----------
        chat_client:
            Any object exposing ``generate_json(system_prompt, user_prompt, ...)``
            (``OpenAIChatClient`` satisfies this).
        fact_lookup:
            Callable ``fact_id -> ManagedFact | None`` — used to resolve fact
            text for sampled ids.

        Returns
        -------
        dict[cluster_id, {"old": str, "new": str, "samples": int, "error": str | None}]
            One entry per cluster that was *attempted* (skipped clusters are
            absent).  Caller can use this to diff / audit the rewrite.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with self._lock:
            targets = [
                (cid, cluster)
                for cid, cluster in self._clusters.items()
                if cluster.fact_count >= min_fact_count
            ]

        if not targets:
            return {}

        sys_prompt = (
            "You label clusters of factual notes about one user's life. "
            "Given a handful of related facts, reply with a short snake_case "
            "label (2-4 words, lowercase, underscore-separated) that captures "
            "their shared theme. Be specific, not generic. "
            'Reply as JSON: {"label": "..."}.'
        )

        def _build_user_prompt(cluster: SceneCluster, samples: list[str]) -> str:
            domain = cluster.scope_key or "general"
            top_topics = sorted(
                cluster.topic_histogram.items(), key=lambda kv: -kv[1]
            )[:5]
            top_domains = sorted(
                cluster.domain_histogram.items(), key=lambda kv: -kv[1]
            )[:5]
            topic_hint = ", ".join(f"{k}({v})" for k, v in top_topics) or "(none)"
            domain_hint = ", ".join(f"{k}({v})" for k, v in top_domains) or "(none)"
            numbered = "\n".join(f"{i+1}. {text}" for i, text in enumerate(samples))
            return (
                f"Primary domain: {domain}\n"
                f"Top topics: {topic_hint}\n"
                f"Top domain_keys: {domain_hint}\n"
                f"Current heuristic label: {cluster.label}\n"
                f"Representative facts ({len(samples)} of {cluster.fact_count}):\n"
                f"{numbered}\n\n"
                "Return a single short snake_case label that covers all these facts."
            )

        def _call_one(cluster_id: str, cluster: SceneCluster) -> tuple[str, dict]:
            # Take the first K facts — for this router, earliest-inserted facts
            # tend to be closest to the bootstrap seed (cluster centroid), so
            # they are more representative than evenly-spaced samples which
            # span the full diversity and push the LLM toward generic labels.
            sample_ids = cluster.fact_ids[:max_samples_per_cluster]
            samples: list[str] = []
            for fid in sample_ids:
                fact = fact_lookup(fid)
                if fact is None:
                    continue
                text = str(getattr(fact, "fact_text", "")).strip()
                if text:
                    samples.append(text[:300])
            if not samples:
                return cluster_id, {"old": cluster.label, "new": cluster.label,
                                    "samples": 0, "error": "no_fact_text"}
            user_prompt = _build_user_prompt(cluster, samples)
            try:
                response = chat_client.generate_json(
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    step_label="scene_relabel",
                )
                raw_label = str(response.get("label") or "").strip()
            except Exception as exc:
                return cluster_id, {"old": cluster.label, "new": cluster.label,
                                    "samples": len(samples),
                                    "error": f"{type(exc).__name__}: {exc}"}
            new_label = _sanitize_label(raw_label) or cluster.label
            return cluster_id, {"old": cluster.label, "new": new_label,
                                "samples": len(samples), "error": None}

        audit: dict[str, dict] = {}
        worker_count = max(1, min(max_inflight, len(targets)))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            future_to_cid = {
                pool.submit(_call_one, cid, cluster): cid
                for cid, cluster in targets
            }
            for future in as_completed(future_to_cid):
                cid, info = future.result()
                audit[cid] = info

        with self._lock:
            for cid, info in audit.items():
                cluster = self._clusters.get(cid)
                if cluster is None or info.get("error") is not None:
                    continue
                new_label = info["new"]
                if new_label and new_label != cluster.label:
                    cluster.label = new_label

        return audit

    def judge_merge_candidates(
        self,
        *,
        chat_client,
        model_name: str,
        fact_lookup,
        theta_low: float = 0.80,
        theta_high: float | None = None,
        max_samples_per_side: int = 4,
        max_inflight: int = 32,
        respect_size_cap: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 64,
        timeout: float = 30.0,
    ) -> dict[str, dict]:
        """LLM adjudication for same-domain cluster pairs in the judge band.

        Enumerates same-domain pairs with cosine similarity in
        ``[theta_low, theta_high)`` and asks the model whether each pair should
        be merged. Pairs already merge-worthy (``sim >= theta_high``) are
        handled by :meth:`merge_clusters`; this pass targets the borderline band
        the heuristic leaves undecided.

        Decisions are applied greedily in descending-similarity order via
        :func:`_merge_cluster_into`. A cluster that has already been absorbed is
        skipped if it appears in a later pair.

        Parameters
        ----------
        theta_high:
            Upper bound of the judge band. Defaults to
            ``self._config.theta_merge`` (0.88 in the stock config).
        respect_size_cap:
            If ``True``, skip pairs whose combined ``fact_count`` would exceed
            ``self._config.max_cluster_size``. Matches :meth:`merge_check`.

        Returns
        -------
        dict[pair_key, {"a", "b", "sim", "decision", "merged", "error", "samples_a", "samples_b"}]
            One entry per pair that was *evaluated*.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if theta_high is None:
            theta_high = float(self._config.theta_merge)

        # Snapshot candidate pairs under the lock.
        candidates: list[tuple[str, str, float]] = []
        with self._lock:
            by_domain: dict[str, list[str]] = {}
            for cid, cluster in self._clusters.items():
                by_domain.setdefault(cluster.scope_key, []).append(cid)
            for domain_ids in by_domain.values():
                for i in range(len(domain_ids)):
                    for j in range(i + 1, len(domain_ids)):
                        left = self._clusters[domain_ids[i]]
                        right = self._clusters[domain_ids[j]]
                        if respect_size_cap and (
                            left.fact_count + right.fact_count
                            > self._config.max_cluster_size
                        ):
                            continue
                        sim = _cosine_similarity(left.centroid, right.centroid)
                        if theta_low <= sim < theta_high:
                            candidates.append((left.cluster_id, right.cluster_id, sim))

        if not candidates:
            return {}

        sys_prompt = (
            "You decide whether two small clusters of factual notes about one "
            "user's life should be merged into a single theme. Two clusters "
            "should merge only if they are about the same specific topic from "
            "the user's life — not just the same broad domain. "
            "Reply as JSON: {\"merge\": true|false, \"reason\": \"short phrase\"}."
        )

        def _samples_for(cluster: SceneCluster) -> list[str]:
            samples: list[str] = []
            for fid in cluster.fact_ids[:max_samples_per_side]:
                fact = fact_lookup(fid)
                if fact is None:
                    continue
                text = str(getattr(fact, "fact_text", "")).strip()
                if text:
                    samples.append(text[:300])
            return samples

        def _build_prompt(
            a: SceneCluster, b: SceneCluster, sim: float,
            samples_a: list[str], samples_b: list[str],
        ) -> str:
            block_a = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(samples_a))
            block_b = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(samples_b))
            return (
                f"Primary domain: {a.scope_key or 'general'}\n"
                f"Centroid cosine similarity: {sim:.3f}\n\n"
                f"Cluster A — label: {a.label} (size {a.fact_count})\n"
                f"A facts ({len(samples_a)} of {a.fact_count}):\n{block_a}\n\n"
                f"Cluster B — label: {b.label} (size {b.fact_count})\n"
                f"B facts ({len(samples_b)} of {b.fact_count}):\n{block_b}\n\n"
                "Do clusters A and B describe the SAME specific topic from the "
                "user's life? If yes, reply {\"merge\": true}. If they share "
                "only the broad domain but describe different things, reply "
                "{\"merge\": false}."
            )

        def _pair_key(a_id: str, b_id: str) -> str:
            return f"{a_id}||{b_id}"

        def _call_one(a_id: str, b_id: str, sim: float) -> tuple[str, dict]:
            with self._lock:
                a = self._clusters.get(a_id)
                b = self._clusters.get(b_id)
            if a is None or b is None:
                return _pair_key(a_id, b_id), {
                    "a": a_id, "b": b_id, "sim": sim,
                    "decision": None, "merged": False, "error": "missing_cluster",
                    "samples_a": 0, "samples_b": 0,
                }
            samples_a = _samples_for(a)
            samples_b = _samples_for(b)
            if not samples_a or not samples_b:
                return _pair_key(a_id, b_id), {
                    "a": a_id, "b": b_id, "sim": sim,
                    "decision": None, "merged": False, "error": "no_fact_text",
                    "samples_a": len(samples_a), "samples_b": len(samples_b),
                }
            user_prompt = _build_prompt(a, b, sim, samples_a, samples_b)
            try:
                response = chat_client.generate_json(
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    step_label="scene_judge_merge",
                )
                decision_raw = response.get("merge")
                if isinstance(decision_raw, str):
                    decision = decision_raw.strip().lower() in {"true", "yes", "1"}
                else:
                    decision = bool(decision_raw)
                reason = str(response.get("reason") or "").strip()[:120]
            except Exception as exc:
                return _pair_key(a_id, b_id), {
                    "a": a_id, "b": b_id, "sim": sim,
                    "decision": None, "merged": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "samples_a": len(samples_a), "samples_b": len(samples_b),
                }
            return _pair_key(a_id, b_id), {
                "a": a_id, "b": b_id, "sim": sim,
                "a_label": a.label, "b_label": b.label,
                "a_size": a.fact_count, "b_size": b.fact_count,
                "decision": decision, "reason": reason,
                "merged": False, "error": None,
                "samples_a": len(samples_a), "samples_b": len(samples_b),
            }

        audit: dict[str, dict] = {}
        worker_count = max(1, min(max_inflight, len(candidates)))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            future_to_key = {
                pool.submit(_call_one, a_id, b_id, sim): _pair_key(a_id, b_id)
                for a_id, b_id, sim in candidates
            }
            for future in as_completed(future_to_key):
                key, info = future.result()
                audit[key] = info

        # Apply approved merges greedily in descending-similarity order.
        approved = [
            (info["a"], info["b"], info["sim"])
            for info in audit.values()
            if info.get("decision") is True and info.get("error") is None
        ]
        approved.sort(key=lambda t: -t[2])
        absorbed: set[str] = set()
        with self._lock:
            for a_id, b_id, sim in approved:
                if a_id in absorbed or b_id in absorbed:
                    continue
                left = self._clusters.get(a_id)
                right = self._clusters.get(b_id)
                if left is None or right is None:
                    continue
                if respect_size_cap and (
                    left.fact_count + right.fact_count
                    > self._config.max_cluster_size
                ):
                    continue
                if left.fact_count >= right.fact_count:
                    keeper, donor = left, right
                else:
                    keeper, donor = right, left
                _merge_cluster_into(keeper, donor)
                self._clusters.pop(donor.cluster_id, None)
                absorbed.add(donor.cluster_id)
                audit[_pair_key(a_id, b_id)]["merged"] = True
            if absorbed:
                self._bootstrap_state = SceneBootstrapState(
                    initialized=self._bootstrap_state.initialized,
                    buffered_fact_ids=list(self._bootstrap_state.buffered_fact_ids),
                    cluster_count=len(self._clusters),
                )

        return audit

    @staticmethod
    def _load_clusters_from_path(path: Path) -> list[SceneCluster]:
        """Load clusters from a saved router JSON without needing an embedding client."""
        import time as _time
        raw = json.loads(path.read_text(encoding="utf-8"))
        clusters = []
        for value in raw.get("clusters") or []:
            clusters.append(SceneCluster(
                cluster_id=str(value["cluster_id"]),
                label=str(value.get("label", "scene")),
                scope_key=str(value.get("scope_key", "general")),
                centroid=list(value.get("centroid") or []),
                fact_count=int(value.get("fact_count", 0)),
                fact_ids=list(value.get("fact_ids") or []),
                created_at=float(value.get("created_at", _time.time())),
                updated_at=float(value.get("updated_at", _time.time())),
                latest_fact_time=_maybe_float(value.get("latest_fact_time")),
                parent_cluster_id=value.get("parent_cluster_id"),
                topic_histogram={str(k): int(v) for k, v in (value.get("topic_histogram") or {}).items()},
                domain_histogram={str(k): int(v) for k, v in (value.get("domain_histogram") or {}).items()},
            ))
        return clusters

    def get_pending_splits(self) -> list[str]:
        with self._lock:
            out = list(self._pending_splits)
            self._pending_splits.clear()
        return out

    def get_cluster(self, cluster_id: str) -> SceneCluster | None:
        with self._lock:
            return self._clusters.get(cluster_id)

    def all_cluster_ids(self) -> list[str]:
        with self._lock:
            return list(self._clusters.keys())

    def remove_cluster(self, cluster_id: str) -> None:
        with self._lock:
            self._clusters.pop(cluster_id, None)
            self._bootstrap_state = SceneBootstrapState(
                initialized=self._bootstrap_state.initialized,
                buffered_fact_ids=list(self._bootstrap_state.buffered_fact_ids),
                cluster_count=len(self._clusters),
            )

    def add_cluster(self, cluster: SceneCluster) -> None:
        with self._lock:
            self._clusters[cluster.cluster_id] = cluster
            self._bootstrap_state = SceneBootstrapState(
                initialized=self._bootstrap_state.initialized,
                buffered_fact_ids=list(self._bootstrap_state.buffered_fact_ids),
                cluster_count=len(self._clusters),
            )

    def remove_fact_ids(self, fact_ids_to_remove: set[str]) -> None:
        if not fact_ids_to_remove:
            return
        with self._lock:
            kept_clusters: dict[str, SceneCluster] = {}
            for cluster_id, cluster in self._clusters.items():
                kept_fact_ids = [fid for fid in cluster.fact_ids if fid not in fact_ids_to_remove]
                if not kept_fact_ids:
                    continue
                kept_clusters[cluster_id] = SceneCluster(
                    cluster_id=cluster.cluster_id,
                    label=cluster.label,
                    scope_key=cluster.scope_key,
                    centroid=list(cluster.centroid),
                    fact_count=len(kept_fact_ids),
                    fact_ids=kept_fact_ids,
                    created_at=cluster.created_at,
                    updated_at=time.time(),
                    latest_fact_time=cluster.latest_fact_time,
                    parent_cluster_id=cluster.parent_cluster_id,
                    topic_histogram=dict(cluster.topic_histogram),
                    domain_histogram=dict(cluster.domain_histogram),
                )
            self._clusters = kept_clusters
            self._pending_splits = [
                cluster_id
                for cluster_id in self._pending_splits
                if cluster_id in self._clusters
            ]
            self._bootstrap_state = SceneBootstrapState(
                initialized=self._bootstrap_state.initialized,
                buffered_fact_ids=[
                    fid for fid in self._bootstrap_state.buffered_fact_ids
                    if fid not in fact_ids_to_remove
                ],
                cluster_count=len(self._clusters),
            )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                "bootstrap_state": {
                    "initialized": self._bootstrap_state.initialized,
                    "buffered_fact_ids": list(self._bootstrap_state.buffered_fact_ids),
                    "cluster_count": self._bootstrap_state.cluster_count,
                },
                "clusters": [
                    {
                        "cluster_id": cluster.cluster_id,
                        "label": cluster.label,
                        "scope_key": cluster.scope_key,
                        "centroid": list(cluster.centroid),
                        "fact_count": cluster.fact_count,
                        "fact_ids": list(cluster.fact_ids),
                        "created_at": cluster.created_at,
                        "updated_at": cluster.updated_at,
                        "latest_fact_time": cluster.latest_fact_time,
                        "parent_cluster_id": cluster.parent_cluster_id,
                        "topic_histogram": dict(cluster.topic_histogram),
                        "domain_histogram": dict(cluster.domain_histogram),
                    }
                    for cluster in self._clusters.values()
                ],
            }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        # Persist embedding cache: texts as JSON, vectors as float32 numpy.
        self._save_embedding_cache(path)

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        state_raw = raw.get("bootstrap_state") or {}
        clusters: dict[str, SceneCluster] = {}
        for value in raw.get("clusters") or []:
            cluster = SceneCluster(
                cluster_id=str(value["cluster_id"]),
                label=str(value.get("label", "scene")),
                scope_key=str(value.get("scope_key", "general")),
                centroid=list(value.get("centroid") or []),
                fact_count=int(value.get("fact_count", 0)),
                fact_ids=list(value.get("fact_ids") or []),
                created_at=float(value.get("created_at", time.time())),
                updated_at=float(value.get("updated_at", time.time())),
                latest_fact_time=_maybe_float(value.get("latest_fact_time")),
                parent_cluster_id=value.get("parent_cluster_id"),
                topic_histogram={str(k): int(v) for k, v in (value.get("topic_histogram") or {}).items()},
                domain_histogram={str(k): int(v) for k, v in (value.get("domain_histogram") or {}).items()},
            )
            clusters[cluster.cluster_id] = cluster
        with self._lock:
            self._clusters = clusters
            self._bootstrap_state = SceneBootstrapState(
                initialized=bool(state_raw.get("initialized", bool(clusters))),
                buffered_fact_ids=list(state_raw.get("buffered_fact_ids") or []),
                cluster_count=int(state_raw.get("cluster_count", len(clusters))),
            )
            self._bootstrap_buffer = []
            self._embedding_cache = {}
            self._load_embedding_cache(path)

    def _bootstrap_locked(self, facts: list["ManagedFact"], embeddings: list[list[float]] | None = None) -> None:
        facts = [fact for fact in facts if fact.fact_text.strip()]
        if not facts:
            return
        sample = facts[: self._config.bootstrap_max_facts]
        embeddings = embeddings[: len(sample)] if embeddings is not None else self._embed_routing_texts(sample)
        seed_indices = _farthest_first_indices(
            embeddings,
            min(self._config.bootstrap_cluster_count, len(sample)),
        )
        now = time.time()
        self._clusters = {}
        for idx in seed_indices:
            fact = sample[idx]
            label = _label_for_fact(fact)
            scope_key = _scope_key_for_fact(fact)
            cluster_id = f"scene:{label}_{uuid.uuid4().hex[:8]}"
            self._clusters[cluster_id] = SceneCluster(
                cluster_id=cluster_id,
                label=label,
                scope_key=scope_key,
                centroid=list(embeddings[idx]),
                fact_count=0,
                fact_ids=[],
                created_at=now,
                updated_at=now,
                latest_fact_time=None,
                parent_cluster_id=None,
                topic_histogram=_histogram(fact.topics),
                domain_histogram=_histogram(fact.domain_keys),
            )
        self._bootstrap_buffer = []
        self._bootstrap_state = SceneBootstrapState(initialized=True, buffered_fact_ids=[], cluster_count=len(self._clusters))

    def _embed_routing_texts(self, facts: list["ManagedFact"]) -> list[list[float]]:
        """Embed routing texts with dedup + caching.

        Identical routing texts (same origin/role/topics/domains/fact_text)
        are only embedded once.  Results are cached across calls so that
        bootstrap embeddings are reused by assign_many, and incremental
        ingest doesn't re-embed already-seen facts.
        """
        texts = [_routing_text_for_fact(fact) for fact in facts]
        # Identify which texts need embedding (not yet cached).
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []
        for i, text in enumerate(texts):
            if text not in self._embedding_cache:
                uncached_texts.append(text)
                uncached_indices.append(i)
        # Batch-embed only the new texts.
        if uncached_texts:
            new_embeddings = self._embedding_client.embed_texts(uncached_texts)
            for text, emb in zip(uncached_texts, new_embeddings):
                self._embedding_cache[text] = emb
        # Assemble results in original order from cache.
        return [self._embedding_cache[text] for text in texts]

    def _save_embedding_cache(self, router_path: Path) -> None:
        """Save embedding cache as keys JSON + float32 numpy matrix."""
        import numpy as np

        keys_path = router_path.with_name(router_path.stem + "_cache_keys.json")
        vecs_path = router_path.with_name(router_path.stem + "_cache_vecs.npy")
        with self._lock:
            if not self._embedding_cache:
                for p in (keys_path, vecs_path):
                    if p.exists():
                        p.unlink()
                return
            keys = list(self._embedding_cache.keys())
            vecs = np.array(
                [self._embedding_cache[k] for k in keys], dtype=np.float32
            )
        keys_path.write_text(json.dumps(keys, ensure_ascii=False), encoding="utf-8")
        np.save(vecs_path, vecs)

    def _load_embedding_cache(self, router_path: Path) -> None:
        """Restore embedding cache from keys JSON + float32 numpy matrix."""
        import numpy as np

        keys_path = router_path.with_name(router_path.stem + "_cache_keys.json")
        vecs_path = router_path.with_name(router_path.stem + "_cache_vecs.npy")
        if not keys_path.exists() or not vecs_path.exists():
            return
        try:
            keys = json.loads(keys_path.read_text(encoding="utf-8"))
            vecs = np.load(vecs_path).tolist()
            if len(keys) == len(vecs):
                self._embedding_cache = dict(zip(keys, vecs))
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    def _assign_locked(self, fact: "ManagedFact", embedding: list[float]) -> list[str]:
        if not self._clusters:
            cluster = _new_cluster_from_fact(fact, embedding)
            self._clusters[cluster.cluster_id] = cluster
            self._register_fact(cluster, fact, embedding)
            return [cluster.cluster_id]

        # ── Domain-gated routing ──────────────────────────────────────
        # Strict gate: only consider clusters whose primary domain
        # matches the fact's domain.  No fallback to global candidates.
        # This prevents cross-domain mixing that made large clusters
        # degrade into topic-incoherent catch-alls.
        domain_key = _domain_gate_key(fact)
        candidate_clusters = [
            cluster for cluster in self._clusters.values()
            if cluster.scope_key == domain_key
        ]
        if not candidate_clusters:
            # No cluster for this domain yet — spawn one directly.
            cluster = _new_cluster_from_fact(fact, embedding)
            self._clusters[cluster.cluster_id] = cluster
            self._register_fact(cluster, fact, embedding)
            return [cluster.cluster_id]

        scored: list[tuple[float, str]] = []
        for cluster in candidate_clusters:
            similarity = _cosine_similarity(cluster.centroid, embedding)
            similarity += _soft_time_bonus(fact, cluster, self._config.max_time_gap_days)
            scored.append((similarity, cluster.cluster_id))
        scored.sort(reverse=True)
        best_score, best_id = scored[0]

        spawn_new = False
        best_cluster = self._clusters[best_id]
        if best_cluster.fact_count >= self._config.max_cluster_size:
            # Hard cap: never assign to a full cluster regardless of score.
            spawn_new = True
        elif best_score < self._config.theta_spawn:
            # No cluster is close enough within the domain — spawn a sub-cluster.
            spawn_new = True
        elif best_score < self._config.theta_assign:
            # Weakly related: only accept if the best cluster is not already
            # close to full.  Once a cluster exceeds half of max_cluster_size,
            # a loosely-matched fact is better off seeding its own cluster.
            if best_cluster.fact_count >= self._config.max_cluster_size // 2:
                spawn_new = True

        if spawn_new:
            cluster = _new_cluster_from_fact(fact, embedding)
            self._clusters[cluster.cluster_id] = cluster
            chosen_ids = [cluster.cluster_id]
        else:
            chosen_ids = [best_id]
            # Dual assignment: only within same-domain clusters.
            if len(scored) > 1:
                second_score, second_id = scored[1]
                second_cluster = self._clusters[second_id]
                if (second_score >= self._config.theta_second
                        and second_id != best_id
                        and second_cluster.fact_count < self._config.max_cluster_size):
                    chosen_ids.append(second_id)

        for cluster_id in chosen_ids:
            cluster = self._clusters[cluster_id]
            self._register_fact(cluster, fact, embedding)
        return chosen_ids

    def _register_fact(self, cluster: SceneCluster, fact: "ManagedFact", embedding: list[float]) -> None:
        cluster.centroid = _update_centroid(cluster.centroid, embedding, cluster.fact_count)
        if fact.fact_id not in cluster.fact_ids:
            cluster.fact_ids.append(fact.fact_id)
        cluster.fact_count += 1
        cluster.updated_at = time.time()
        cluster.latest_fact_time = fact.time_end if fact.time_end is not None else fact.time_start
        _update_histogram(cluster.topic_histogram, fact.topics)
        _update_histogram(cluster.domain_histogram, fact.domain_keys)
        # scope_key is fixed at creation — never updated — to prevent domain drift.
        # label is updated for display only (not used for routing).
        cluster.label = _label_from_histograms(cluster) or cluster.label
        if cluster.fact_count > self._config.max_cluster_size:
            self._pending_splits.append(cluster.cluster_id)


def _routing_text_for_fact(fact: "ManagedFact") -> str:
    topics = ", ".join(fact.topics[:3]) or "none"
    domains = ", ".join(fact.domain_keys[:3]) or "none"
    return (
        f"origin={fact.origin}\n"
        f"role={fact.semantic_role}\n"
        f"topics={topics}\n"
        f"domains={domains}\n"
        f"fact={fact.fact_text.strip()}"
    )


def _domain_gate_key(fact: "ManagedFact") -> str:
    """Return the primary domain key used for strict cluster gating.

    Uses the first domain_key if available, otherwise falls back to the
    first topic.  This key is set once on a cluster at creation and never
    updated, ensuring that facts are only routed to same-domain clusters.
    """
    if fact.domain_keys:
        return fact.domain_keys[0].strip().lower().replace(" ", "_") or "general"
    if fact.topics:
        return fact.topics[0].strip().lower().replace(" ", "_") or "general"
    return "general"


def _scope_key_for_fact(fact: "ManagedFact") -> str:
    """Return the scope key for a fact (used at cluster creation)."""
    return _domain_gate_key(fact)


def _label_for_fact(fact: "ManagedFact") -> str:
    parts: list[str] = []
    if fact.domain_keys:
        parts.append(fact.domain_keys[0].strip().lower().replace(" ", "_"))
    if fact.topics:
        topic = fact.topics[0].strip().lower().replace(" ", "_")
        if topic and topic not in parts:
            parts.append(topic)
    if not parts:
        parts.append(fact.semantic_role.strip().lower() or "scene")
    return "_".join(parts[:2])


def _new_cluster_from_fact(fact: "ManagedFact", embedding: list[float]) -> SceneCluster:
    now = time.time()
    label = _label_for_fact(fact)
    return SceneCluster(
        cluster_id=f"scene:{label}_{uuid.uuid4().hex[:8]}",
        label=label,
        scope_key=_scope_key_for_fact(fact),
        centroid=list(embedding),
        fact_count=0,
        fact_ids=[],
        created_at=now,
        updated_at=now,
        latest_fact_time=None,
        parent_cluster_id=None,
        topic_histogram=_histogram(fact.topics),
        domain_histogram=_histogram(fact.domain_keys),
    )


def _histogram(values: list[str]) -> dict[str, int]:
    hist: dict[str, int] = {}
    _update_histogram(hist, values)
    return hist


def _update_histogram(hist: dict[str, int], values: list[str]) -> None:
    for value in values:
        key = value.strip().lower().replace(" ", "_")
        if not key:
            continue
        hist[key] = hist.get(key, 0) + 1


def _sanitize_label(raw: str) -> str:
    """Normalise an LLM-supplied label to a short snake_case phrase.

    - Lowercase, collapse whitespace to underscores, strip non-[a-z0-9_] chars.
    - Keep at most the first 4 tokens (<= ~40 chars) to avoid essay labels.
    """
    import re
    if not raw:
        return ""
    cleaned = re.sub(r"[^a-z0-9\s_\-]", " ", raw.lower())
    cleaned = re.sub(r"[\s\-]+", "_", cleaned.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        return ""
    tokens = cleaned.split("_")[:4]
    label = "_".join(t for t in tokens if t)
    return label[:40]


def _label_from_histograms(cluster: SceneCluster) -> str | None:
    primary = _top_key(cluster.domain_histogram) or _top_key(cluster.topic_histogram)
    secondary = None
    if primary and primary in cluster.domain_histogram:
        secondary = _top_key({k: v for k, v in cluster.topic_histogram.items() if k != primary})
    if primary and secondary:
        return f"{primary}_{secondary}"
    return primary


def _scope_key_from_histograms(cluster: SceneCluster) -> str | None:
    return _top_key(cluster.domain_histogram) or _top_key(cluster.topic_histogram)


def _top_key(hist: dict[str, int]) -> str | None:
    if not hist:
        return None
    return sorted(hist.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _soft_time_bonus(fact: "ManagedFact", cluster: SceneCluster, max_gap_days: float) -> float:
    fact_time = fact.time_start if fact.time_start is not None else fact.time_end
    if fact_time is None or cluster.latest_fact_time is None:
        return 0.0
    gap_days = abs(fact_time - cluster.latest_fact_time) / 86400.0
    if gap_days >= max_gap_days:
        return 0.0
    return 0.08 * (1.0 - gap_days / max_gap_days)


def _farthest_first_indices(embeddings: list[list[float]], k: int) -> list[int]:
    if not embeddings:
        return []
    k = max(1, min(k, len(embeddings)))
    chosen = [0]
    while len(chosen) < k:
        best_idx = None
        best_score = -1.0
        for idx, emb in enumerate(embeddings):
            if idx in chosen:
                continue
            min_dist = min(1.0 - _cosine_similarity(emb, embeddings[cidx]) for cidx in chosen)
            if min_dist > best_score:
                best_score = min_dist
                best_idx = idx
        if best_idx is None:
            break
        chosen.append(best_idx)
    return chosen


def _merge_cluster_into(keeper: SceneCluster, donor: SceneCluster) -> None:
    """Merge *donor* into *keeper* in-place.  Caller removes donor from the registry."""
    total = keeper.fact_count + donor.fact_count
    if total > 0:
        w_k = keeper.fact_count / total
        w_d = donor.fact_count / total
        keeper.centroid = [
            w_k * a + w_d * b for a, b in zip(keeper.centroid, donor.centroid)
        ]
    for fid in donor.fact_ids:
        if fid not in keeper.fact_ids:
            keeper.fact_ids.append(fid)
    keeper.fact_count = total
    keeper.updated_at = time.time()
    if donor.latest_fact_time is not None:
        if keeper.latest_fact_time is None or donor.latest_fact_time > keeper.latest_fact_time:
            keeper.latest_fact_time = donor.latest_fact_time
    for key, count in donor.topic_histogram.items():
        keeper.topic_histogram[key] = keeper.topic_histogram.get(key, 0) + count
    for key, count in donor.domain_histogram.items():
        keeper.domain_histogram[key] = keeper.domain_histogram.get(key, 0) + count
    keeper.label = _label_from_histograms(keeper) or keeper.label


def _maybe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
