#!/usr/bin/env bash
set -euo pipefail

MEMORYOS_ROOT="${MEMORYOS_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ROOT="${RUN_ROOT:-$PWD/runs/memoryos_locomo_$(date +%Y%m%d_%H%M%S)}"

LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8001/v1}"
LLM_MODEL="${LLM_MODEL:-google/gemma-4-12B-it}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
EMBED_BASE_URL="${EMBED_BASE_URL:-http://127.0.0.1:8003/v1}"
EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
LOCOMO_DATA="${LOCOMO_DATA:-$MEMORYOS_ROOT/eval/locomo10.json}"

FROM_SAMPLE="${FROM_SAMPLE:-0}"
TO_SAMPLE="${TO_SAMPLE:-10}"
MEMORYOS_LOCOMO_SHARDS="${MEMORYOS_LOCOMO_SHARDS:-64}"
MEMORYOS_LOCOMO_MAX_PARALLEL="${MEMORYOS_LOCOMO_MAX_PARALLEL:-64}"

mkdir -p "$RUN_ROOT/logs/memoryos_locomo" "$RUN_ROOT/memoryos_locomo/shards"

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

total=$((TO_SAMPLE - FROM_SAMPLE))
if (( total <= 0 )); then
  echo "Invalid range: FROM_SAMPLE=$FROM_SAMPLE TO_SAMPLE=$TO_SAMPLE" >&2
  exit 2
fi

{
  printf '{\n'
  printf '  "run_root": "%s",\n' "$RUN_ROOT"
  printf '  "dataset": "%s",\n' "$LOCOMO_DATA"
  printf '  "llm_base_url": "%s",\n' "$LLM_BASE_URL"
  printf '  "llm_model": "%s",\n' "$LLM_MODEL"
  printf '  "embedding_base_url": "%s",\n' "$EMBED_BASE_URL"
  printf '  "embedding_model": "%s",\n' "$EMBED_MODEL"
  printf '  "from_sample": %s,\n' "$FROM_SAMPLE"
  printf '  "to_sample": %s,\n' "$TO_SAMPLE"
  printf '  "shards": %s,\n' "$MEMORYOS_LOCOMO_SHARDS"
  printf '  "max_parallel": %s,\n' "$MEMORYOS_LOCOMO_MAX_PARALLEL"
  printf '  "worker_policy": "one MemoryOS process per shard; each shard writes independent output JSON",\n'
  printf '  "created_at": "%s"\n' "$(date --iso-8601=seconds)"
  printf '}\n'
} > "$RUN_ROOT/memoryos_locomo/manifest.json"

pids=()
for shard in $(seq 0 $((MEMORYOS_LOCOMO_SHARDS - 1))); do
  start=$((FROM_SAMPLE + (total * shard) / MEMORYOS_LOCOMO_SHARDS))
  end=$((FROM_SAMPLE + (total * (shard + 1)) / MEMORYOS_LOCOMO_SHARDS))
  if (( start == end )); then
    continue
  fi
  while (( $(jobs -pr | wc -l) >= MEMORYOS_LOCOMO_MAX_PARALLEL )); do
    sleep 5
  done

  shard_id="$(printf '%03d' "$shard")"
  shard_root="$RUN_ROOT/memoryos_locomo/shards/shard_$shard_id"
  log_path="$RUN_ROOT/logs/memoryos_locomo/shard_$shard_id.log"
  mkdir -p "$shard_root"
  (
    cd "$MEMORYOS_ROOT/eval"
    MEMORYOS_API_KEY="$LLM_API_KEY" \
    MEMORYOS_BASE_URL="$LLM_BASE_URL" \
    MEMORYOS_LLM_MODEL="$LLM_MODEL" \
    MEMORYOS_EMBED_BASE_URL="$EMBED_BASE_URL" \
    MEMORYOS_EMBED_MODEL="$EMBED_MODEL" \
    MEMORYOS_EMBED_API_KEY=EMPTY \
    MEMORYOS_DATASET="$LOCOMO_DATA" \
    MEMORYOS_MEMORY_DIR="$shard_root/memory" \
    MEMORYOS_OUTPUT_FILE="$shard_root/output/all_loco_results_gemma.json" \
    MEMORYOS_SAMPLE_OFFSET="$start" \
    MEMORYOS_SAMPLE_LIMIT="$((end - start))" \
    MEMORYOS_RESUME=1 \
    MEMORYOS_API_RETRIES=5 \
    PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" main_loco_parse.py
  ) > "$log_path" 2>&1 &
  pids+=("$!")
  echo "started MemoryOS LoCoMo shard $shard_id sample[$start,$end) pid=${pids[-1]}"
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

if (( fail != 0 )); then
  echo "One or more MemoryOS LoCoMo shards failed; see $RUN_ROOT/logs/memoryos_locomo" >&2
  exit 1
fi

echo "MemoryOS LoCoMo completed: $RUN_ROOT/memoryos_locomo"
