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
  ../MemForest/reproducibility/baselines/mem0/evermemos_reference_date.patch
```

The essential correction is in `_add_user_messages`: the first benchmark
message timestamp in each batch is normalized to UTC and written to both
`metadata.created_at` and `metadata.event_time`. Search results from both
participant perspectives are then globally reduced to one final `k=10` context.
The small EverMemOS patch prepends the benchmark `question_date` to the answer
context when supplied by the evaluation stage.

The local REST server should be configured with the selected generation and
embedding endpoints, isolated Qdrant storage, and graph memory disabled. Run
each conversation in only one shard. After execution, require exactly 500 add,
search, answer, and judge records and verify that temporal contexts contain
benchmark dates rather than server wall-clock dates.
