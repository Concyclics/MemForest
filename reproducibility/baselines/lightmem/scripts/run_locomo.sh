#!/usr/bin/env bash
set -euo pipefail

LIGHTMEM_ROOT="${LIGHTMEM_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET="${DATASET:?Set DATASET to locomo10.json}"
RUN_ROOT="${RUN_ROOT:-$PWD/runs/lightmem_locomo_$(date +%Y%m%d_%H%M%S)}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8001/v1}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
EMBED_BASE_URL="${EMBED_BASE_URL:-http://127.0.0.1:8003/v1}"
EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
EMBED_DIM="${EMBED_DIM:-1024}"
WORKERS="${WORKERS:-1}"

test -f "$LIGHTMEM_ROOT/experiments/locomo/add_locomo.py" || {
  echo "Set LIGHTMEM_ROOT to the patched LightMem checkout" >&2
  exit 2
}

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/results"
export PYTHONPATH="$LIGHTMEM_ROOT/src:$LIGHTMEM_ROOT/experiments/locomo:${PYTHONPATH:-}"

cd "$LIGHTMEM_ROOT/experiments/locomo"
"$PYTHON_BIN" add_locomo.py \
  --dataset "$DATASET" \
  --qdrant-pre-dir "$RUN_ROOT/qdrant_pre_update" \
  --qdrant-post-dir "$RUN_ROOT/qdrant_post_update" \
  --llm-api-key "$LLM_API_KEY" \
  --llm-base-url "$LLM_BASE_URL" \
  --llm-model "$LLM_MODEL" \
  --embedding-provider openai \
  --embedding-model-path "$EMBED_MODEL" \
  --embedding-api-base-url "$EMBED_BASE_URL" \
  --embedding-api-key EMPTY \
  --embedding-dims "$EMBED_DIM" \
  --workers "$WORKERS" \
  2>&1 | tee "$RUN_ROOT/logs/add.log"

"$PYTHON_BIN" search_locomo.py \
  --dataset "$DATASET" \
  --qdrant-dir "$RUN_ROOT/qdrant_pre_update" \
  --output-dir "$RUN_ROOT/results" \
  --embedder openai \
  --embedding-model-path "$EMBED_MODEL" \
  --embedding-api-base-url "$EMBED_BASE_URL" \
  --embedding-api-key EMPTY \
  --embedding-dims "$EMBED_DIM" \
  --limit-per-speaker 10 \
  --total-limit 10 \
  --allow-categories 1 2 3 4 5 \
  --llm-api-key "$LLM_API_KEY" \
  --llm-base-url "$LLM_BASE_URL" \
  --llm-model "$LLM_MODEL" \
  --judge-api-key "$LLM_API_KEY" \
  --judge-base-url "$LLM_BASE_URL" \
  --judge-model "$LLM_MODEL" \
  --resume \
  2>&1 | tee "$RUN_ROOT/logs/search.log"
