#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEMPALACE_ROOT="${MEMPALACE_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET="${DATASET:?Set DATASET to longmemeval_s_cleaned.json}"
RUN_ROOT="${RUN_ROOT:-$PWD/runs/mempalace_longmemeval_$(date +%Y%m%d_%H%M%S)}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8001/v1}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
EMBED_BASE_URL="${EMBED_BASE_URL:-http://127.0.0.1:8003/v1}"
EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
EMBED_DIM="${EMBED_DIM:-1024}"
TOP_K="${TOP_K:-10}"
ANSWER_WORKERS="${ANSWER_WORKERS:-128}"

test -f "$MEMPALACE_ROOT/benchmarks/longmemeval_bench.py" || {
  echo "Set MEMPALACE_ROOT to the patched MemPalace checkout" >&2
  exit 2
}

mkdir -p "$RUN_ROOT/retrieval" "$RUN_ROOT/answers"
export MEMPALACE_EMBEDDING_BASE_URL="$EMBED_BASE_URL"
export MEMPALACE_EMBEDDING_MODEL="$EMBED_MODEL"
export MEMPALACE_EMBEDDING_DIMS="$EMBED_DIM"
export MEMPALACE_EMBEDDING_API_KEY="${MEMPALACE_EMBEDDING_API_KEY:-EMPTY}"

cd "$MEMPALACE_ROOT"
"$PYTHON_BIN" benchmarks/longmemeval_bench.py "$DATASET" \
  --granularity "${MEMPALACE_GRANULARITY:-session}" \
  --mode "${MEMPALACE_MODE:-hybrid_v4}" \
  --embed-model openai \
  --out "$RUN_ROOT/retrieval/results.jsonl" \
  --perf-dir "$RUN_ROOT/retrieval/perf"

"$PYTHON_BIN" "$SCRIPT_DIR/generate_longmemeval_answers.py" \
  --retrieval-jsonl "$RUN_ROOT/retrieval/results.jsonl" \
  --dataset-json "$DATASET" \
  --out-jsonl "$RUN_ROOT/answers/answers.jsonl" \
  --summary-json "$RUN_ROOT/answers/summary.json" \
  --base-url "$LLM_BASE_URL" \
  --model "$LLM_MODEL" \
  --api-key "$LLM_API_KEY" \
  --workers "$ANSWER_WORKERS" \
  --top-k "$TOP_K" \
  --resume
