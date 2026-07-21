#!/usr/bin/env bash
set -euo pipefail

EVERMEMOS_ROOT="${EVERMEMOS_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET="${DATASET:-longmemeval}"
SYSTEM="${SYSTEM:-evermemos_longmemeval_local}"
RUN_NAME="${RUN_NAME:-${SYSTEM}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$EVERMEMOS_ROOT/evaluation/results/$RUN_NAME}"

test -f "$EVERMEMOS_ROOT/evaluation/cli.py" || {
  echo "Set EVERMEMOS_ROOT to the patched EverMemOS checkout" >&2
  exit 2
}

export VECTORIZE_PROVIDER="${VECTORIZE_PROVIDER:-vllm}"
export VECTORIZE_API_KEY="${VECTORIZE_API_KEY:-EMPTY}"
export VECTORIZE_BASE_URL="${VECTORIZE_BASE_URL:-http://127.0.0.1:8003/v1}"
export VECTORIZE_MODEL="${VECTORIZE_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
export VECTORIZE_MAX_CONCURRENT="${VECTORIZE_MAX_CONCURRENT:-128}"
export VECTORIZE_FALLBACK_PROVIDER="${VECTORIZE_FALLBACK_PROVIDER:-none}"

cd "$EVERMEMOS_ROOT"
exec "$PYTHON_BIN" -m evaluation.cli \
  --dataset "$DATASET" \
  --system "$SYSTEM" \
  --stages add search answer evaluate \
  --clean-groups \
  --run-name "$RUN_NAME" \
  --output-dir "$OUTPUT_DIR"
