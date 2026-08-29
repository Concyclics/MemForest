# Corrected Mem0 adapter

This directory publishes the Mem0-specific code used to correct benchmark time
propagation in the revision.

Base checkouts:

```bash
git clone https://github.com/EverMind-AI/EverMemOS.git
git -C EverMemOS checkout 539db77d5cc804c875246e34611fd266bf8c1e5d
git clone https://github.com/mem0ai/mem0.git
git -C mem0 checkout 4642a1d6e372e985fff9683070b0127bf6d77621
```

Install `aiohttp`, copy `mem0_adapter.py` to
`EverMemOS/evaluation/src/adapters/mem0_adapter.py`, copy
`mem0_local_qwen3.yaml` to `EverMemOS/evaluation/config/systems/`, and apply:

```bash
git -C EverMemOS apply \
  /path/to/MemForest/reproducibility/baselines/mem0/evermemos_reference_date.patch
```

The essential correction is in `_add_user_messages`. Add batches are bounded by
source session, and that session's benchmark time is normalized to UTC. The
evaluated local REST path writes it to both `metadata.created_at` and
`metadata.event_time`; the SDK fallback passes the same value through Mem0's
`timestamp` argument. Search results from both participant perspectives are
then globally reduced to one final `k=10` context. The small EverMemOS patch
prepends the benchmark `question_date` to the answer context when supplied by
the evaluation stage.

The local REST server should be configured with the selected generation and
embedding endpoints, a fresh Qdrant collection, and graph memory disabled. Run
each conversation in only one shard. The released LoCoMo correction rerun
contains 1,986 answer/search/top-10 records and 1,696 timestamp-audit rows for
each of Qwen3-4B, Qwen3-30B, and Gemma-4-12B, with no invalid timestamps,
coverage errors, or blank answers. The LongMemEval-S protocol uses 500
questions.

The exact rerun protocol and checks are recorded in:

- [`../../results/mem0_corrected/timestamp_rerun_manifest.json`](../../results/mem0_corrected/timestamp_rerun_manifest.json);
- [`../../results/mem0_corrected/timestamp_validation_summary.csv`](../../results/mem0_corrected/timestamp_validation_summary.csv).

The manifest records the evaluated adapter SHA-256 and the released adapter
SHA-256 separately. The released file preserves the evaluated timestamp and
batching logic with formatting and comment cleanup; the offline verifier checks
its recorded hash.

## Retrieval-budget audit

The revision also profiles how much of each frozen Mem0 store is exposed by a
given retrieval budget. This analysis reads the retained local Qdrant stores;
it does not rebuild memory or call an LLM:

```bash
python reproducibility/scripts/profile_mem0_retrieval_budget.py \
  --stores-root /path/to/corrected_mem0_run/stores \
  --dataset /path/to/longmemeval_s_cleaned.json \
  --output reproducibility/results/mem0_corrected/retrieval_budget_store_profile.json
```

The committed profile covers the corrected Qwen3-30B run. Accuracy at larger
budgets must be measured separately with the same frozen store, answer model,
answer prompt, and judge; public top-200 scores are not treated as if they came
from this local configuration.

## Fixed-store accuracy control

`reproducibility/scripts/run_mem0_budget_judge_control.py` implements the
top-50/top-200 control. It validates regenerated retrieval against each frozen
top-20 result, generates answers from one coherent top-200 ranking and its
top-50 prefix, and applies both judge prompts with three votes. The script is
resumable by stage and never records `DEEPSEEK_API_KEY`.

The committed outputs are under
`reproducibility/results/mem0_corrected/top50_top200_control/`. The managed
public result remains a separate protocol reference because model, answer path,
and proprietary pipeline differ.
