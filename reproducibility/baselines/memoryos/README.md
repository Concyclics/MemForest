# MemoryOS reproduction overlay

Upstream: `https://github.com/BAI-LAB/MemoryOS`
Commit: `8688d5128901a88a70a3ba961de8705a6cdab4c0`

`upstream.patch` contains the exact local endpoint, embedding, timestamp,
retrieval-budget, logging, and resume changes used by our experiments across
the upstream MemoryOS backends and evaluation code.

```bash
git clone https://github.com/BAI-LAB/MemoryOS.git
git -C MemoryOS checkout 8688d5128901a88a70a3ba961de8705a6cdab4c0
git -C MemoryOS apply /path/to/MemForest/reproducibility/baselines/memoryos/upstream.patch
```

LongMemEval-S is executed with the self-contained runner in `scripts/`, which
creates one isolated MemoryOS store per question and supports process-level
sharding:

```bash
MEMORYOS_ROOT=$PWD/MemoryOS \
LONGMEM_DATA=/path/to/longmemeval_s_cleaned.json \
LLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
bash /path/to/MemForest/reproducibility/baselines/memoryos/scripts/run_longmemeval_shards.sh
```

LoCoMo uses the patched official `eval/main_loco_parse.py` through
`scripts/run_locomo_shards.sh`. Set `MEMORYOS_ROOT`, `LOCOMO_DATA`, model and
endpoint variables analogously. The patch sets `top_k_sessions=10`; the
wrappers use `Qwen/Qwen3-Embedding-0.6B` with 1,024 dimensions by default.
