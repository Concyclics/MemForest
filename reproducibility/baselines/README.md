# Baseline implementation package

This directory contains the released integration code and pinned overlays for
the baselines in the main and revision experiments. Third-party repositories
are not vendored. Each directory pins an upstream commit and provides the
applicable subset of:

- `upstream.patch`: our complete code overlay relative to that commit;
- `configs/`: checked-in experiment configurations, where applicable;
- `scripts/`: portable run and aggregation entry points;
- `UPSTREAM_LICENSE`: the license shipped by the pinned upstream repository;
- `README.md`: checkout, patch, and execution instructions.

| Method | Code directory | Upstream patch | Portable runner |
|---|---|---:|---:|
| EverMemOS | [`evermemos/`](evermemos/) | yes | yes |
| Mem0 | [`mem0/`](mem0/) | adapter + EverMemOS patch | config documented |
| LightMem | [`lightmem/`](lightmem/) | yes | LongMemEval + LoCoMo |
| MemoryOS | [`memoryos/`](memoryos/) | yes | LongMemEval + LoCoMo |
| MemPalace | [`mempalace/`](mempalace/) | yes | LongMemEval + LoCoMo |
| Zep Local | [`zep_local/`](zep_local/) | pinned MemoryData/Graphiti runner | both benchmarks |

The patches retain each method's native memory schema and execution structure.
Shared controls are applied at the experiment boundary: the selected generation
backbone, Qwen3-Embedding-0.6B where an external embedder is supported, final
retrieval budget `k=10`, benchmark timestamps, and the reported judge protocol.

Local endpoint URLs are defaults only. API keys are read from environment
variables and no credential is included in this repository.
