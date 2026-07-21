#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEMORYOS_ROOT="${MEMORYOS_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ROOT="${RUN_ROOT:-$PWD/runs/memoryos_longmemeval_$(date +%Y%m%d_%H%M%S)}"

LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8001/v1}"
LLM_MODEL="${LLM_MODEL:-google/gemma-4-12B-it}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
EMBED_BASE_URL="${EMBED_BASE_URL:-http://127.0.0.1:8003/v1}"
EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
EMBED_DIM="${EMBED_DIM:-1024}"
LONGMEM_DATA="${LONGMEM_DATA:?Set LONGMEM_DATA to longmemeval_s_cleaned.json}"

FROM_INDEX="${FROM_INDEX:-0}"
TO_INDEX="${TO_INDEX:-500}"
MEMORYOS_LME_SHARDS="${MEMORYOS_LME_SHARDS:-64}"
MEMORYOS_LME_MAX_PARALLEL="${MEMORYOS_LME_MAX_PARALLEL:-64}"

mkdir -p "$RUN_ROOT/logs/memoryos_longmemeval" "$RUN_ROOT/memoryos_longmemeval/shards"

wait_for_url() {
  local url="$1"
  local name="$2"
  local max_try="${3:-60}"
  local i
  for i in $(seq 1 "$max_try"); do
    if curl --max-time 10 -fsS "$url" >/dev/null 2>&1; then
      echo "$name ready: $url"
      return 0
    fi
    sleep 2
  done
  echo "$name not ready: $url" >&2
  return 1
}

wait_for_url "$LLM_BASE_URL/models" "LLM"
wait_for_url "$EMBED_BASE_URL/models" "Embedding"

total=$((TO_INDEX - FROM_INDEX))
if (( total <= 0 )); then
  echo "Invalid range: FROM_INDEX=$FROM_INDEX TO_INDEX=$TO_INDEX" >&2
  exit 2
fi

{
  printf '{\n'
  printf '  "run_root": "%s",\n' "$RUN_ROOT"
  printf '  "dataset": "%s",\n' "$LONGMEM_DATA"
  printf '  "llm_base_url": "%s",\n' "$LLM_BASE_URL"
  printf '  "llm_model": "%s",\n' "$LLM_MODEL"
  printf '  "embedding_base_url": "%s",\n' "$EMBED_BASE_URL"
  printf '  "embedding_model": "%s",\n' "$EMBED_MODEL"
  printf '  "embedding_dim": %s,\n' "$EMBED_DIM"
  printf '  "from_index": %s,\n' "$FROM_INDEX"
  printf '  "to_index": %s,\n' "$TO_INDEX"
  printf '  "shards": %s,\n' "$MEMORYOS_LME_SHARDS"
  printf '  "max_parallel": %s,\n' "$MEMORYOS_LME_MAX_PARALLEL"
  printf '  "created_at": "%s"\n' "$(date --iso-8601=seconds)"
  printf '}\n'
} > "$RUN_ROOT/memoryos_longmemeval/manifest.json"

pids=()
for shard in $(seq 0 $((MEMORYOS_LME_SHARDS - 1))); do
  start=$((FROM_INDEX + (total * shard) / MEMORYOS_LME_SHARDS))
  end=$((FROM_INDEX + (total * (shard + 1)) / MEMORYOS_LME_SHARDS))
  if (( start == end )); then
    continue
  fi
  while (( $(jobs -pr | wc -l) >= MEMORYOS_LME_MAX_PARALLEL )); do
    sleep 5
  done

  shard_id="$(printf '%03d' "$shard")"
  shard_root="$RUN_ROOT/memoryos_longmemeval/shards/shard_$shard_id"
  log_path="$RUN_ROOT/logs/memoryos_longmemeval/shard_$shard_id.log"
  mkdir -p "$shard_root"
  (
    cd "$MEMORYOS_ROOT"
    MEMORYOS_ROOT="$MEMORYOS_ROOT" PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$SCRIPT_DIR/run_longmemeval.py" \
      --dataset "$LONGMEM_DATA" \
      --out-dir "$shard_root/output" \
      --store-dir "$shard_root/store" \
      --from-index "$start" \
      --to-index "$end" \
      --llm-base-url "$LLM_BASE_URL" \
      --llm-model "$LLM_MODEL" \
      --llm-api-key "$LLM_API_KEY" \
      --embed-base-url "$EMBED_BASE_URL" \
      --embed-model "$EMBED_MODEL" \
      --embed-api-key EMPTY \
      --embed-dim "$EMBED_DIM" \
      --resume
  ) > "$log_path" 2>&1 &
  pids+=("$!")
  echo "started MemoryOS LongMemEval shard $shard_id index[$start,$end) pid=${pids[-1]}"
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

if (( fail != 0 )); then
  echo "One or more MemoryOS LongMemEval shards failed; see $RUN_ROOT/logs/memoryos_longmemeval" >&2
  exit 1
fi

echo "MemoryOS LongMemEval completed: $RUN_ROOT/memoryos_longmemeval"
