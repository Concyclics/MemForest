# Full Public-Judge Regrade Audit

## Scope

- Judge model: `deepseek-v4-flash`
- Frozen-answer cells: 3 answer backbones x 8 methods x
  (500 LongMemEval-S + 1,986 LoCoMo) = 59,664
- Final valid labels: 59,664
- Final unresolved rows: 0
- Retrieval and answer generation were not rerun.
- The API key is not stored in the run artifacts.

## Prompt Provenance

- LongMemEval: Mem0 public prompt at
  `7ba1bd330f6ef6acdc751b6e1f82ac8af0568873`.
- LoCoMo: Mem0 tuned public prompt at
  `edcd6f1d42400837b1fcb6997716f1769dc51a37`.
- Temperature was 0 and thinking was disabled.

## Recovery History

- The first asynchronous pass used 2,048 in-flight requests and a
  `max_tokens=128` output cap.
- The host soft file-descriptor limit was initially 1,024, causing 3,046
  connection failures. These rows were retried after raising the process limit
  to 8,192; successful rows were not called again.
- Sixteen LoCoMo responses remained unparseable because verbose reasoning was
  truncated or the JSON was slightly malformed. They were called again with
  `max_tokens=256` and an explicit-label JSON recovery parser.
- `judged_calls.jsonl` retains failed attempts for auditability.
  `validation.json` distinguishes historical failed attempts from final
  unresolved rows.

## LoCoMo Reporting Boundary

The public LoCoMo prompt assumes a non-empty gold answer. Category 5
(adversarial/unanswerable) has blank gold answers in the source artifact, so a
public-prompt full-set score is not directly comparable to the public
categories 1--4 benchmark. Use the `cat1-4` rows in `summary.csv` for
public-score alignment. Keep Category 5 under the strict answerability judge or
report it separately.

## Source Limitation

The original Qwen main-table answers for `MemForest (emb)` were not found in
the retained submission artifacts. The Qwen `memforest_embed` rows in this run
use the later complete shared-instruction sample-0 rerun. They are valid
diagnostic regrades but are not pure judge-only replacements for the original
67.6/78.4 LongMemEval-S and 61.4/66.9 LoCoMo table cells.

All other Qwen rows align with the retained main-table answer artifacts, except
that the intended corrected Mem0 LongMemEval answers replace the known
reference-date-bug run. Gemma and Zep Local rows use their complete revision
artifacts.
