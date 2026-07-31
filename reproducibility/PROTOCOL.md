# Revision evaluation protocol

## Datasets

| Benchmark | Main scope | Questions | Notes |
|---|---:|---:|---|
| LongMemEval-S | full cleaned split | 500 | 133 temporal-reasoning questions |
| LoCoMo main | categories 1-4 | 1,540 | category 2 is the 321-question temporal subset |
| LoCoMo adversarial supplement | category 5 | 446 | reported separately from the main QA aggregate |

Dataset SHA-256 values are recorded in
[`manifests/revision_release.json`](manifests/revision_release.json).

## Model roles

The evaluated generation backbones are:

- `Qwen/Qwen3-4B-Instruct-2507-FP8`;
- `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`;
- `google/gemma-4-12B-it`.

For the Gemma generality study, Gemma replaces every generative stage from
memory construction through answer generation. The embedding model stays fixed
so the experiment changes the generative family rather than both model roles.

## Retrieval and answer generation

- The main retrieval-based comparison uses final `k=10`.
- Native retrieved units are not assumed to have equal granularity across
  systems. MemForest tree expansion and Zep Local heterogeneous graph objects
  are reported explicitly where relevant.
- Primary end-to-end rows use each system's released/recommended schema-aware
  answer path.
- The `shared_strong_v1` answer instruction is a sensitivity diagnostic only;
  forcing one schema-neutral prompt increased abstention and reduced scores
  across methods.
- Benchmark event timestamps are propagated through ingestion. LongMemEval
  answer generation receives the benchmark question/reference date.

## Judge

The main tables use one fixed `deepseek-v4-flash` judge call per frozen answer,
temperature 0, and thinking disabled. LongMemEval-S uses the released Mem0
prompt at `memory-benchmarks` commit `7ba1bd3`; LoCoMo categories 1--4 use the
tuned released prompt at commit `edcd6f1`. Prompt hashes and source paths are
recorded in
[`results/public_judge_three_backbone/input_manifest.json`](results/public_judge_three_backbone/input_manifest.json).
Judge output is normalized to `CORRECT` or `WRONG`; all 59,664 expected labels
must be present and unresolved rows must be zero.

Older `deepseek-chat` three-vote results are retained only for the strict-judge,
pass@k, and retrieval-budget diagnostics. They are not the source of the revised
main tables. Corrected Mem0 additionally releases pass@1--8 with eight sampled
answers (`temperature=0.7`, `top_p=0.95`) and the standard cumulative pass@k
definition. The executable strict/sensitivity judge is released at
[`evaluation/unified_deepseek_judge.py`](evaluation/unified_deepseek_judge.py).

## Reporting boundaries

- The original submission CSVs are preserved as an immutable audit snapshot.
- The corrected Mem0 rows replace the submitted Mem0 rows.
- Qwen MemForest-Embed uses a protocol-matched rerun with native top-10 tree
  browsing, full selected-unit expansion, and the default answer prompt.
- The permissive-judge and shared-answer runs are sensitivity diagnostics, not
  replacement leaderboard values.
- Zep Local is a reproducible local architecture-level baseline, not a claim
  about the managed Zep Cloud product.
- LoCoMo category 5 is reported separately because its adversarial answerability
  semantics differ from categories 1-4.
