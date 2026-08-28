# Session-level MemTree write concentration

Source streams: `28dc39ac`, `bc8a6e93`, `7e00a6cb`.
Inputs are the saved canonical fact store and final entity/scene trees under `workdir_measured/<qid>`.

A collision is two or more canonical facts from one source session routed to the same final tree. The metric is an upper-bound conflict proxy for per-fact insertion; the released builder deduplicates affected tree IDs and dirty ancestors before the LLM refresh.
Session-pair overlap is computed only within each independent question/user stream before the three streams are aggregated.
For tree-level concurrent writes, session-pair overlap is the relevant potential lock-conflict proxy: excluding the global fallback, 16.6% of pairs share a specific entity/scene tree and 83.4% are disjoint. The global `entity:user` tree is a separable hotspot that can be partitioned into independently built subforests and consolidated through migration/merge.

| Scope | Sessions | Trees/session | Facts/touched tree (mean/median/p95/max) | Collision pairs | Excess assignments | Session-pair overlap |
|---|---:|---:|---:|---:|---:|---:|
| all | 147 | 7.60 | 5.39/3.0/19/53 | 67.1% | 81.5% | 89.3% |
| without_global_fallback | 147 | 6.65 | 5.04/2.0/21/53 | 63.7% | 80.2% | 16.6% |
