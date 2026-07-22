# Revision result index

## Corrected Mem0 LongMemEval-S

| Backbone | Questions | Overall pass@1 | Temporal pass@1 | Submitted overall | Submitted temporal |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B | 500 | 35.6% | 28.6% | 32.8% | 24.1% |
| Qwen3-30B | 500 | 47.7% | 36.8% | 40.2% | 27.8% |

The corrected rows evaluate all 500 questions and expose benchmark event time
in retrieved contexts. The primary numbers come from
`primary_pass1_three_judge.json`. The separately generated eight-sample pass@k
run starts at 36.6%/47.6% and is not substituted for the primary row. Files:
[`results/mem0_corrected/`](results/mem0_corrected/).

For Qwen3-30B, 47.7% is the rounded mean of three judge-run accuracies
`[47.6%, 47.6%, 47.8%]`; the majority-label count stored in the same artifact is
238/500 (47.6%).

### Corrected Mem0 store occupancy

The retained Qwen3-30B stores contain 121,594 memory points across 500 isolated
LongMemEval-S conversations. A conversation has 243.2 memories on average and
240 at the median (range 156--339). The table reports the fraction of the full
per-question memory store exposed by each retrieval budget, before any context
window truncation:

| Subset | Mean memories | Median memories | Top-10 share | Top-50 share | Top-200 share | Stores exhausted by top-200 |
|---|---:|---:|---:|---:|---:|---:|
| All 500 | 243.2 | 240 | 4.18% | 20.92% | 83.11% | 40/500 (8.0%) |
| Temporal (133) | 244.9 | 244 | 4.15% | 20.73% | 82.48% | 8/133 (6.0%) |

Thus top-200 is close to exhaustive recall for this corrected local Mem0
snapshot and is not comparable to the common final top-k=10 main table. This
occupancy result alone does not establish an accuracy gain; the controlled
accuracy sweep freezes the memory state and varies only retrieval/answer
context budget. File:
[`results/mem0_corrected/retrieval_budget_store_profile.json`](results/mem0_corrected/retrieval_budget_store_profile.json).

### Fixed-store top-50/top-200 control

The retained corrected Qwen3-30B stores were queried once to top-200 and sliced
at top-50. The same Mem0 answer path was used at both cutoffs, followed by
three `deepseek-chat` votes under both the appendix strict judge and the public
LongMemEval prompt from `memory-benchmarks` commit `7ba1bd3`.

| Cutoff | Strict overall | Public overall | Strict temporal | Public temporal | Mean prompt tokens |
|---:|---:|---:|---:|---:|---:|
| 50 | 46.60% | 49.40% | 37.59% | 41.35% | 2,284 |
| 200 | 45.80% | 48.60% | 37.59% | 38.35% | 7,919 |

The public prompt adds 2.80 points overall at both cutoffs. Top-200 minus
top-50 is -0.80 points overall under either judge, with paired 95% intervals
crossing zero. Thus this local control confirms judge sensitivity but does not
reproduce the managed v3 snapshot's positive budget delta or absolute score.
Files:
[`results/mem0_corrected/top50_top200_control/`](results/mem0_corrected/top50_top200_control/).

### Independent semantic audit

The final evidence-governance audit covers all 79 provisional temporal evidence
mappings, 200 stratified entity-routing facts, and all 40 strict/public judge
disagreements. Only 33/79 provisional temporal rows were fully upheld, so the
revision omits exact fragmentation precision from that mapping. Among 86 rows
with active entity keys, 84 pass semantic precision and five cover every salient
entity. Independent disagreement review prefers strict on 33/40 and public on
7/40; eight subjective boundary rows remain separated for optional author
sign-off. These are model-assisted independent reviews, not human gold. Files:
[`results/semantic_audit/`](results/semantic_audit/).

The expanded, pre-frozen audit contains 249 temporal mappings, 300
entity-routing facts, and 120 judge-calibration pairs. It fully reproduces
162/249 provisional temporal mappings; 124/127 active entity assignments pass
precision; and 108/120 judge adjudications are stable without additional
author sign-off. All exceptions remain in explicit sign-off queues. The larger
counts strengthen the bounded diagnostics but do not convert model-assisted
labels into human gold. Files:
[`results/semantic_audit/expanded/`](results/semantic_audit/expanded/).

## Zep Local full-system baseline

| Backbone | LongMemEval-S | LME temporal | LoCoMo cat. 1-4 | LoCoMo full |
|---|---:|---:|---:|---:|
| Qwen3-4B | 58.20% | 38.35% | 54.48% | 44.51% |
| Qwen3-30B | 66.60% | 50.38% | 67.60% | 54.98% |
| Gemma 4 12B IT | 66.80% | 48.12% | 62.01% | 49.60% |

Each row contains 500 LongMemEval-S and 1,986 LoCoMo questions. All six cells
completed with zero judge errors. Files:
[`results/zep_local/`](results/zep_local/).

## Gemma cross-family matrix

Under the appendix DeepSeek judge, all seven methods have complete 500-question
LongMemEval-S and 1,540-question LoCoMo category 1-4 coverage. Selected rows:

| Method | LongMemEval-S | LoCoMo cat. 1-4 |
|---|---:|---:|
| EverMemOS | 76.40% | 76.10% |
| MemPalace | 69.60% | 79.94% |
| MemForest agent browse | 76.80% | 75.00% |
| MemForest embedding browse | 77.20% | 73.83% |

The result supports cross-family portability but not universal dominance.
Files: [`results/gemma/`](results/gemma/).

## Judge-prompt sensitivity

With answers frozen, the historical permissive prompt changes temporal
accuracy by:

| Method | LoCoMo temporal | LongMemEval temporal |
|---|---:|---:|
| EverMemOS | +17.33 points | +4.92 points |
| Mem0 | +5.33 points | +4.92 points |
| MemForest | +16.00 points | +2.46 points |

This quantifies protocol sensitivity. Since MemForest also gains, it is not
used to claim that only baseline scores were inflated. Files:
[`results/judge_prompt_sensitivity/`](results/judge_prompt_sensitivity/).
