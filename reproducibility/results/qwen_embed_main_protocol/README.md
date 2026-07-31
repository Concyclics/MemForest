# Qwen MemForest-Embed main-protocol records

These files close the provenance of the revised Qwen MemForest-Embed rows.
They are protocol-matched answer records, not relabeled outputs from the older
shared-instruction diagnostic.

For both Qwen3-4B and Qwen3-30B:

- retrieval uses Qwen3-Embedding-0.6B and native top-10 MemTree-unit browsing;
- every selected unit is fully expanded before answer generation;
- the answer prompt is `memforest_default_v1`;
- the answer backbone matches the corresponding MemForest-Planner row;
- one deterministic answer is retained for all 500 LongMemEval-S and 1,986
  LoCoMo questions.

The JSONL rows contain the generated answer, benchmark identifiers, expansion
counts, and context length. The adjacent manifests pin hashes of the original
answer and retrieval files without exposing machine-local paths. Main-table
labels are in `../public_judge_three_backbone/` and use one
`deepseek-v4-flash` call with the released benchmark prompt.

Run `../../scripts/verify_release.py` from the repository root to verify row
coverage, protocol fields, record hashes, and the corresponding main scores.
