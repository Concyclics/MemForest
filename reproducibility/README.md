# VLDB revision reproducibility package

This directory contains the public evidence added for the VLDB revision. The
conference uses single-blind review, so the response links directly to this
repository rather than to a separate anonymized archive.

## Contents

- [`paper/MemForest_full_revision.pdf`](paper/MemForest_full_revision.pdf):
  public complete paper with the supplementary diagnostics omitted from the
  length-limited submission PDF.
- [`paper/MemForest_response_full_revision.pdf`](paper/MemForest_response_full_revision.pdf):
  combined revision response, revised paper, references, and Appendices A--F
  for reviewer inspection and author-side trimming.
- [`BASELINES.md`](BASELINES.md): pinned upstream versions, local deployment
  boundaries, timestamp policy, retrieval budget, and reproduction procedure.
- [`baselines/`](baselines/): pinned code overlays, configs, licenses, and
  portable runners for every main/revision baseline.
- [`evaluation/`](evaluation/): the frozen shared DeepSeek judge implementation.
- [`PROTOCOL.md`](PROTOCOL.md): datasets, model roles, answer generation, judge,
  pass@k, and reporting policy.
- [`RESULTS.md`](RESULTS.md): revision headline results and interpretation.
- [`manifests/`](manifests/): machine-readable source and release metadata.
- [`implementation/`](implementation/): the asynchronous
  `Forest`/`AsyncBPlusTree` source snapshot used by the revised online
  lock-granularity and snapshot semantics.
- [`results/`](results/): compact summaries and sanitized per-question records.
- [`scripts/verify_release.py`](scripts/verify_release.py): offline completeness,
  count, and headline-result checks.
- [`scripts/profile_mem0_retrieval_budget.py`](scripts/profile_mem0_retrieval_budget.py):
  profiles the fraction of each frozen Mem0 store exposed by top-k retrieval.
- [`scripts/run_mem0_budget_judge_control.py`](scripts/run_mem0_budget_judge_control.py):
  reruns the frozen-store top-50/top-200 answer and dual-judge control.
- [`results/semantic_audit/`](results/semantic_audit/): frozen source sheets,
  independent review records, author-signoff queues, and aggregate decisions.
- [`results/semantic_audit/author_adjudicated/`](results/semantic_audit/author_adjudicated/):
  final 249-row temporal, 300-row entity-routing, and 120-row judge-policy
  author adjudication, including 231 retained temporal mappings and 18
  documented exclusions.
- [`results/public_judge_three_backbone/`](results/public_judge_three_backbone/):
  complete 59,664-row public-judge regrade over three backbones, eight methods,
  and both benchmarks; includes compact per-question labels, summary, manifest,
  and validation.
- [`results/qwen_embed_main_protocol/`](results/qwen_embed_main_protocol/):
  protocol-matched Qwen MemForest-Embed answers, expansion metadata, source
  hashes, and manifests for the four revised main-table cells.
- [`results/write_path_traces/`](results/write_path_traces/): Figure 1 native
  write-rate coordinates, representative MemForest traces, measurement scope,
  and source hashes.
- [`results/write_conflicts/`](results/write_conflicts/): the 147-session
  entity/scene-tree write-concentration audit used in the R3-W3 response,
  including per-session/tree rows, aggregate statistics, and provenance.
- [`results/deepseek_cost_probe/`](results/deepseek_cost_probe/): three
  independent matched five-method, 20-message DeepSeek-V4-Flash probes after
  two disjoint excluded warmups, with API-returned billable token classes,
  direct mean/min/max costs, template/provider-hit validation, cache-isolation
  manifests, and sanitized traces. The original cold-start measurements remain
  under [`results/deepseek_cost_probe_cold/`](results/deepseek_cost_probe_cold/)
  for backend-semantics auditing.
- [`results/zep_local/native_budget_summary.csv`](results/zep_local/native_budget_summary.csv):
  six-cell native Graphiti object counts and serialized-context token lengths.
- [`manifests/runtime_configs.json`](manifests/runtime_configs.json): serving,
  concurrency, index, routing, and retrieval settings used by the revision.
- [`scripts/build_qwen_embed_main_records.py`](scripts/build_qwen_embed_main_records.py):
  normalization and protocol gates for the released Qwen Embed records.
- [`scripts/summarize_zep_native_budget.py`](scripts/summarize_zep_native_budget.py):
  regenerates the Zep native-object and context-token summary from query items.
- [`scripts/run_independent_semantic_audit.py`](scripts/run_independent_semantic_audit.py):
  reruns the optional local Qwen semantic review.
- [`scripts/summarize_independent_semantic_audit.py`](scripts/summarize_independent_semantic_audit.py):
  validates and joins the released semantic audit without API access.
- [`scripts/run_expanded_semantic_audit.py`](scripts/run_expanded_semantic_audit.py):
  reruns the expanded 249/300/120 local Qwen review.
- [`scripts/summarize_expanded_semantic_audit.py`](scripts/summarize_expanded_semantic_audit.py):
  validates and regenerates the expanded joined records without API access.
- [`scripts/analyze_session_tree_write_conflicts.py`](scripts/analyze_session_tree_write_conflicts.py):
  regenerates the R3-W3 write-concentration tables from saved matched-build
  fact stores and tree snapshots.
- [`SHA256SUMS`](SHA256SUMS): checksums for every released revision file.

The original submission's four large per-question CSV files remain in
[`../benchmark/`](../benchmark/). They are not silently overwritten. Corrected
or newly added revision cells are versioned in this directory.

## Quick verification

From the repository root:

```bash
python reproducibility/scripts/verify_release.py
```

The check is API-free. It verifies expected question counts, judge-error counts,
record hashes, Qwen Embed protocol fields, and the headline values cited by the
response. Re-running answer generation or LLM judging requires the external
baseline repositories and model services listed in
[`BASELINES.md`](BASELINES.md).

Every baseline implementation is indexed in
[`baselines/README.md`](baselines/README.md). We publish complete patches
relative to pinned upstream commits instead of mutable copies of third-party
repositories.

## Data policy

Per-question release files retain the benchmark question, gold answer,
generated answer, model identifier, judge label, and prompt version. Local
absolute paths, service URLs, credentials, and database directories are
removed. No API key is stored in this repository.
