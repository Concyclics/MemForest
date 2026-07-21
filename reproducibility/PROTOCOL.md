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

The revision uses `deepseek-chat` with the appendix prompts and temperature 0.
Judge output is parsed as `CORRECT` or `WRONG`; failures are retained and must
be zero before a complete result is reported. Prompt text and hashes are stored
beside the corresponding result files.
The executable judge is released at
[`evaluation/unified_deepseek_judge.py`](evaluation/unified_deepseek_judge.py).

Primary tables report one answer per question with three judge calls and the
mean accuracy across judge repetitions. The corrected 30B run has repetition
accuracies `[0.476, 0.476, 0.478]`, whose mean is 0.476666 and is reported as
47.7%; its majority-label count is 238/500 (47.6%). Corrected
Mem0 additionally releases pass@1-8 with eight independently sampled answers
(`temperature=0.7`, `top_p=0.95`) and the standard cumulative pass@k definition:
a question is correct at `k` when any of its first `k` samples is judged
correct. The first sample of this stochastic pass@k run is not the primary
table's separately generated answer, so its pass@1 can differ slightly.

## Reporting boundaries

- The original submission CSVs are preserved as an immutable audit snapshot.
- The corrected Mem0 rows replace the submitted Mem0 rows.
- The permissive-judge and shared-answer runs are sensitivity diagnostics, not
  replacement leaderboard values.
- Zep Local is a reproducible local architecture-level baseline, not a claim
  about the managed Zep Cloud product.
- LoCoMo category 5 is reported separately because its adversarial answerability
  semantics differ from categories 1-4.
