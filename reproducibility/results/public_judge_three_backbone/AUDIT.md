# Full Public-Judge Regrade Audit

## Scope

- Judge model: `deepseek-v4-flash`
- Frozen-answer cells: 3 answer backbones x 8 methods x
  (500 LongMemEval-S + 1,986 LoCoMo) = 59,664
- Final valid labels: 59,664
- Final unresolved rows: 0
- Retrieval and answer generation are frozen for all rows. Most rows reuse the
  retained runs; Qwen MemForest-Embed uses the protocol-matched rerun described
  below, and corrected Mem0 replaces the known timestamp-bug run.
- The API key is not stored in the run artifacts.

## Prompt Provenance

- LongMemEval: Mem0 public prompt at
  `7ba1bd330f6ef6acdc751b6e1f82ac8af0568873`.
- LoCoMo: Mem0 tuned public prompt at
  `edcd6f1d42400837b1fcb6997716f1769dc51a37`.
- Temperature was 0 and thinking was disabled.

## Complete Run

The final native-unit v3 run used 2,048 in-flight requests and completed all
59,664 calls without a failed or unresolved row. `validation.json` records the
exact counts; this compact release retains one final label per frozen answer.

## LoCoMo Reporting Boundary

The public LoCoMo prompt assumes a non-empty gold answer. Category 5
(adversarial/unanswerable) has blank gold answers in the source artifact, so a
public-prompt full-set score is not directly comparable to the public
categories 1--4 benchmark. Use the `cat1-4` rows in `summary.csv` for
public-score alignment. Keep Category 5 under the strict answerability judge or
report it separately.

## Qwen MemForest-Embed Provenance

The revised Qwen MemForest-Embed rows are a protocol-matched rerun rather than
a judge-only relabel of the original submission. Qwen3-Embedding-0.6B performs
native top-10 MemTree-unit browsing, selected units are fully expanded, and the
default MemForest answer prompt and matching Qwen backbone generate one answer
per question. The release contains 500 LongMemEval-S and 1,986 LoCoMo records
for each Qwen backbone, including expansion counts and context lengths, under
[`../qwen_embed_main_protocol/`](../qwen_embed_main_protocol/). Source answer
and retrieval hashes are recorded in the adjacent manifests.

All other Qwen rows align with retained main-table answer artifacts except that
corrected Mem0 replaces the known reference-date-bug answers. Gemma and Zep
Local rows use their complete revision artifacts.
