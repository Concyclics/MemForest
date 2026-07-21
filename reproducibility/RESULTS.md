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
