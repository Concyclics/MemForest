# LightMem reproduction overlay

Upstream: `https://github.com/zjunlp/LightMem`
Commit: `82230514d06b0401f16be25fa44a7985596dbcbd`

`upstream.patch` contains the exact code changes used in the benchmark:
OpenAI-compatible generation and embedding endpoints, Qwen/Gemma model
selection, resumable output, detailed call logging, LoCoMo parallel ingestion,
and a globally bounded retrieval context. It applies to the pinned checkout:

```bash
git clone https://github.com/zjunlp/LightMem.git
git -C LightMem checkout 82230514d06b0401f16be25fa44a7985596dbcbd
git -C LightMem apply /path/to/MemForest/reproducibility/baselines/lightmem/upstream.patch
```

LongMemEval-S:

```bash
LIGHTMEM_ROOT=$PWD/LightMem \
DATASET=/path/to/longmemeval_s_cleaned.json \
LLM_BASE_URL=http://127.0.0.1:8001/v1 \
LLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
EMBED_BASE_URL=http://127.0.0.1:8003/v1 \
bash /path/to/MemForest/reproducibility/baselines/lightmem/scripts/run_longmemeval.sh
```

LoCoMo uses `scripts/run_locomo.sh` with the same environment variables and
`DATASET=/path/to/locomo10.json`. Both entry points set the final retrieval
budget to ten. The LoCoMo runner preserves LightMem's native add-then-search
workflow; the LongMemEval runner preserves its official offline-update path.
