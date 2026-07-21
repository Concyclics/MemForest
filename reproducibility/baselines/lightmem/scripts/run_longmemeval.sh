#!/usr/bin/env bash
set -euo pipefail

LIGHTMEM_ROOT="${LIGHTMEM_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET="${DATASET:?Set DATASET to longmemeval_s_cleaned.json}"
RUN_ROOT="${RUN_ROOT:-$PWD/runs/lightmem_longmemeval_$(date +%Y%m%d_%H%M%S)}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8001/v1}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
EMBED_BASE_URL="${EMBED_BASE_URL:-http://127.0.0.1:8003/v1}"
EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
EMBED_DIM="${EMBED_DIM:-1024}"

test -f "$LIGHTMEM_ROOT/experiments/longmemeval/run_lightmem_qwen.py" || {
  echo "Set LIGHTMEM_ROOT to the patched LightMem checkout" >&2
  exit 2
}

mkdir -p "$RUN_ROOT"
cd "$LIGHTMEM_ROOT"
exec "$PYTHON_BIN" experiments/longmemeval/run_lightmem_qwen.py \
  --data-path "$DATASET" \
  --benchmark longmemeval \
  --results-root "$RUN_ROOT" \
  --api-base-url "$LLM_BASE_URL" \
  --api-key "$LLM_API_KEY" \
  --llm-model "$LLM_MODEL" \
  --judge-api-base-url "$LLM_BASE_URL" \
  --judge-api-key "$LLM_API_KEY" \
  --judge-model "$LLM_MODEL" \
  --embedding-provider openai \
  --embedding-model-path "$EMBED_MODEL" \
  --embedding-api-base-url "$EMBED_BASE_URL" \
  --embedding-api-key EMPTY \
  --embedding-dims "$EMBED_DIM" \
  --answer-limit 10
