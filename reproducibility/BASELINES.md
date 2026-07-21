# Baseline reproduction methods

## Shared deployment policy

All locally reproducible systems use the same generative backbone within a
model row and `Qwen/Qwen3-Embedding-0.6B` with 1,024-dimensional output where an
external embedding model is required. The main controlled comparison uses a
final retrieval budget of `k=10`. Benchmark event/session timestamps are passed
through ingestion and retained in answer contexts. The answer path recommended
for each memory schema is the primary end-to-end setting; a schema-neutral
shared-answer prompt is reported only as a sensitivity diagnostic.

The exact upstream repositories and commits are recorded in
[`manifests/baseline_versions.json`](manifests/baseline_versions.json).
The complete implementation package is indexed from
[`baselines/README.md`](baselines/README.md). For each method, we publish the
code overlay relative to the pinned upstream commit, portable runners, actual
configs, and the upstream license rather than vendoring a mutable third-party
checkout.

## Mem0 corrected rerun

The submitted Mem0 LongMemEval artifact used the EverMemOS evaluation harness
at `539db77d5cc804c875246e34611fd266bf8c1e5d` and Mem0 at
`4642a1d6e372e985fff9683070b0127bf6d77621`. Although the adapter exposed a
timestamp argument, the retrieved temporal contexts used ingestion-time dates.
All 133 temporal questions in the submitted artifact showed this failure, and
the submitted 4B artifact covered only 485 of 500 questions.

The corrected local REST adapter applies the following procedure:

1. Parse the benchmark session/message timestamp and normalize it to UTC.
2. Pass the timestamp at ingestion and persist it in both `created_at` and
   `event_time` metadata.
3. Verify that search output exposes the stored event timestamp instead of the
   server wall clock.
4. Retrieve candidates from both conversation participants and globally rerank
   them to one final `k=10` answer context.
5. Pass the benchmark `question_date` as the answer reference date.
6. Require add, search, answer, and judge completeness for all 500 questions.

The rerun used 128 isolated local-Qdrant conversation shards. Sharding changes
throughput, not the per-conversation memory state. The primary table uses one
answer per question with three DeepSeek judgments; the additional pass@1-8 run
samples eight answers and is reported separately. Corrected primary and pass@k
per-question labels and summaries are in
[`results/mem0_corrected/`](results/mem0_corrected/).
The corrected adapter, config, and reference-date patch are public under
[`baselines/mem0/`](baselines/mem0/).

## EverMemOS

EverMemOS is based on the official repository at commit
`539db77d5cc804c875246e34611fd266bf8c1e5d`. We use its staged extraction,
memory construction, hybrid retrieval, answer pipeline, and official
schema-aware answer prompt with the selected local backbone. The timestamp
audit found benchmark dates in its retrieved contexts; it did not show the
Mem0 ingestion-time collapse.

For score reconciliation, note that published EverMemOS tables use a different
answer-model/protocol setting. We therefore report local controlled results and
published results as different configurations, not as exact reproductions of
one another.
The exact evaluation overlay, six model--dataset configs, preparation script,
and runner are under [`baselines/evermemos/`](baselines/evermemos/).

## LightMem, MemoryOS, and MemPalace

These systems are run from their official repositories and adapters, with the
local generation and embedding endpoints substituted where supported. Their
base commits are pinned in the manifest. The audit did not find the Mem0-style
runtime-date substitution in these methods. Method-specific retrieved objects
remain in their native schemas; `k=10` means the final ten retrieved memory
units, which can differ in granularity from ten MemForest facts.

The exact code is released separately for audit and execution:

- LightMem: [`baselines/lightmem/`](baselines/lightmem/), including its
  LongMemEval and LoCoMo entry points;
- MemoryOS: [`baselines/memoryos/`](baselines/memoryos/), including the
  isolated-store LongMemEval runner and sharded LoCoMo runner;
- MemPalace: [`baselines/mempalace/`](baselines/mempalace/), including native
  retrieval plus schema-aware answer generation for both benchmarks.

## Zep Local

`Zep Local` denotes a Graphiti-based local reproduction, not Zep Cloud and not
the deprecated Zep Community Edition. It follows the public implementation
accompanying *Are We Ready For An Agent-Native Memory System?*:

- MemoryData commit: `c63391c128e33eedb91115edf689f12acf4bbc63`;
- entry point: `methods/zep_local/main.py`;
- Graphiti: v0.24.1, commit
  `d2654003ffc11821bce73c493162a40181b23504`;
- Neo4j: 5.26.2;
- embedding: `Qwen/Qwen3-Embedding-0.6B`, dimension 1,024.

Reproduction path:

1. Ingest raw dialogue episodes through `Graphiti.add_episode(...)` with the
   benchmark event/session timestamp as `reference_time`.
2. Let Graphiti extract and resolve entities and temporal fact edges, including
   validity/invalidation metadata, into local Neo4j.
3. Use the released hybrid Graphiti search/reranking over edges, nodes, episodes,
   and communities with `SearchConfig.limit=10`.
4. Build the released schema-aware context and answer with the same backbone
   used for graph extraction/resolution in that row.
5. Judge the frozen answers with the shared DeepSeek appendix judge.

Graphiti's limit is a per-result-class cap, not a global ten-flat-fact cap. The
release therefore preserves native context sizes instead of presenting the row
as an equal-fact-budget representation test. Results for all three backbones
and both benchmarks are under [`results/zep_local/`](results/zep_local/).
The resumable runner and aggregator are under
[`baselines/zep_local/`](baselines/zep_local/).

## Shared answer judging

The frozen DeepSeek judge implementation is released under
[`evaluation/`](evaluation/). Baseline runners produce method-native retrieved
contexts and answers; judging is performed afterward so changes to a memory
implementation cannot silently change the evaluation prompt.

## Graphiti controlled representation diagnostic

The temporal-representation mini-benchmark is distinct from Zep Local. Its
Graphiti row inserts the same pre-extracted canonical facts used by the log,
validity-interval, scratchpad, and MemTree variants. This controls extraction
quality and isolates representation/retrieval. It must not be interpreted as an
end-to-end Zep or Zep Cloud result.

## Judge-prompt sensitivity

The judge sensitivity experiment freezes answers and varies only the judge
prompt. It contains 8,496 DeepSeek calls: three votes over 2,832
method-benchmark-prompt items, with zero API or parse errors. The historical
Mem0-family prompts raise scores for all tested systems, so this is protocol
sensitivity evidence rather than a baseline-specific correction. The summary
and manifest are in [`results/judge_prompt_sensitivity/`](results/judge_prompt_sensitivity/)
and [`manifests/judge_prompt_sensitivity.json`](manifests/judge_prompt_sensitivity.json).
