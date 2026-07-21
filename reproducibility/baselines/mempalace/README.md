# MemPalace reproduction overlay

Upstream: `https://github.com/milla-jovovich/mempalace`
Commit: `71736a3f4f1afe01982d5b0d4c36eb5405eaab91`

`upstream.patch` contains the exact benchmark changes used to support an
OpenAI-compatible Qwen3 embedding endpoint, local Qwen/Gemma answer models,
stage/call logging, and resumable LongMemEval execution.

```bash
git clone https://github.com/milla-jovovich/mempalace.git
git -C mempalace checkout 71736a3f4f1afe01982d5b0d4c36eb5405eaab91
git -C mempalace apply /path/to/MemForest/reproducibility/baselines/mempalace/upstream.patch
```

Portable end-to-end entry points are provided for both datasets:

```bash
MEMPALACE_ROOT=$PWD/mempalace \
DATASET=/path/to/longmemeval_s_cleaned.json \
LLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
bash /path/to/MemForest/reproducibility/baselines/mempalace/scripts/run_longmemeval.sh
```

Use `run_locomo.sh` with `DATASET=/path/to/locomo10.json` for LoCoMo. The
scripts first execute MemPalace's native retrieval and then generate answers
from the retrieved schema with the selected backbone. Both use top-k ten and
Qwen3-Embedding-0.6B/1,024 dimensions by default.
