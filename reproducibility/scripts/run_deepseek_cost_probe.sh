#!/usr/bin/env bash
set -euo pipefail

ROOT=/ssd2/chenhan/MemBench
STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$ROOT/runs/revision_deepseek_cost_probe_${STAMP}"
PYTHON=/home/chenhan/miniconda3/envs/agent/bin/python
VLLM_PYTHON=/home/chenhan/miniconda3/envs/vllm/bin/python
MODEL=deepseek-v4-flash
EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBED_PATH=/home/chenhan/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
PROBE_MESSAGES=20
PROBE_CONV_INDEX="${PROBE_CONV_INDEX:-0}"
PROXY_PORT="${PROXY_PORT:-18001}"
EMBED_PORT="${EMBED_PORT:-18002}"
MEM0_PORT="${MEM0_PORT:-18931}"
NEO4J_HTTP_PORT="${NEO4J_HTTP_PORT:-17474}"
NEO4J_BOLT_PORT="${NEO4J_BOLT_PORT:-17687}"
EMBED_GPU="${EMBED_GPU:-3}"
NEO4J_CONTAINER="ds-cost-neo4j-${STAMP//_/-}"
ISOLATION_PREFIX="${ISOLATION_PREFIX_OVERRIDE:-mf-cost-${STAMP//_/-}}"
PROXY_PID=""
EMBED_PID=""
HEARTBEAT_PID=""

: "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY is required}"
mkdir -p "$RUN_DIR"/{services,proxy,memforest,evermemos,mem0,neo4j/data,neo4j/logs}
chmod 0777 "$RUN_DIR/neo4j/data" "$RUN_DIR/neo4j/logs"

cleanup() {
  local pid
  for pid in "$HEARTBEAT_PID" "$EMBED_PID" "$PROXY_PID"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  docker stop "$NEO4J_CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for port in "$PROXY_PORT" "$EMBED_PORT" "$MEM0_PORT" "$NEO4J_HTTP_PORT" "$NEO4J_BOLT_PORT"; do
  if ss -ltn "sport = :$port" | tail -n +2 | grep -q .; then
    printf 'required port %s is already in use\n' "$port" >&2
    exit 2
  fi
done

wait_url() {
  local url="$1"
  local label="$2"
  for _ in $(seq 1 600); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      printf '%s ready\n' "$label"
      return 0
    fi
    sleep 1
  done
  printf '%s failed to start\n' "$label" >&2
  return 1
}

"$PYTHON" "$ROOT/MemForest_Sigmod/artifact/scripts/deepseek_cost_usage_proxy.py" \
  --log-path "$RUN_DIR/proxy/llm_usage.jsonl" \
  --model "$MODEL" \
  --isolation-prefix "$ISOLATION_PREFIX" \
  --system-prompt-log "$RUN_DIR/proxy/system_prompts.jsonl" \
  --port "$PROXY_PORT" \
  > "$RUN_DIR/services/proxy.log" 2>&1 &
PROXY_PID=$!

(
  cd "$ROOT/MemoryForest"
  exec env CUDA_VISIBLE_DEVICES="$EMBED_GPU" "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$EMBED_PATH" \
    --served-model-name "$EMBED_MODEL" \
    --host 127.0.0.1 \
    --port "$EMBED_PORT" \
    --trust-remote-code \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.1 \
    --hf-overrides '{"is_matryoshka": true}'
) > "$RUN_DIR/services/embedding.log" 2>&1 &
EMBED_PID=$!

wait_url "http://127.0.0.1:$PROXY_PORT/control/status" proxy
wait_url "http://127.0.0.1:$EMBED_PORT/v1/models" embedding

"$PYTHON" "$ROOT/scripts/embedding_keepalive.py" \
  --base-url "http://127.0.0.1:$EMBED_PORT/v1" \
  --model "$EMBED_MODEL" \
  --interval 30 \
  --batch-size 4096 \
  --text-words 256 \
  > "$RUN_DIR/services/embedding_heartbeat.log" 2>&1 &
HEARTBEAT_PID=$!

read -r PROBE_SOURCE_ID QIDS < <("$PYTHON" - \
  "$ROOT/MemForest_Sigmod/artifact/datasets/performance_locomo_flat_20.json" \
  "$RUN_DIR/mini_locomo_flat.json" \
  "$RUN_DIR/zep_data/locomo10_real.json" \
  "$PROBE_MESSAGES" \
  "$PROBE_CONV_INDEX" <<'PY'
import copy
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
zep_output = pathlib.Path(sys.argv[3])
limit = int(sys.argv[4])
group_index = int(sys.argv[5])
rows = json.loads(source.read_text(encoding="utf-8"))
group_ids = list(dict.fromkeys(str(row["locomo_sample_id"]) for row in rows))
if not 0 <= group_index < len(group_ids):
    raise SystemExit(f"conversation index {group_index} outside [0, {len(group_ids)})")
source_id = group_ids[group_index]
row = copy.deepcopy(next(row for row in rows if str(row["locomo_sample_id"]) == source_id))
sessions, dates, session_ids = [], [], []
remaining = limit
for session, date, session_id in zip(
    row["haystack_sessions"], row["haystack_dates"], row["haystack_session_ids"]
):
    if remaining <= 0:
        break
    take = min(remaining, len(session))
    sessions.append(session[:take])
    dates.append(date)
    session_ids.append(session_id)
    remaining -= take
row["haystack_sessions"] = sessions
row["haystack_dates"] = dates
row["haystack_session_ids"] = session_ids
row["probe_message_limit"] = limit
row["probe_conversation_index"] = group_index
if remaining:
    raise SystemExit(f"{source_id} has only {limit - remaining} messages; expected {limit}")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps([row], indent=2) + "\n", encoding="utf-8")
zep_output.parent.mkdir(parents=True, exist_ok=True)
zep_output.write_text(json.dumps([row], indent=2) + "\n", encoding="utf-8")
print(source_id, row["question_id"])
PY
)
cp "$ROOT/runs/revision_prefix_cache_probe_extended_20260730/zep_data/locomo10.json" \
  "$RUN_DIR/zep_data/locomo10.json"

"$PYTHON" - \
  "$ROOT/MemoryForest/src/config/default.yaml" \
  "$RUN_DIR/memforest_config.yaml" \
  "http://127.0.0.1:$PROXY_PORT/v1" \
  "http://127.0.0.1:$EMBED_PORT/v1" \
  "$MODEL" \
  "$EMBED_MODEL" <<'PY'
import pathlib
import sys
import yaml

source, output, llm_url, embedding_url, llm_model, embedding_model = sys.argv[1:]
config = yaml.safe_load(pathlib.Path(source).read_text(encoding="utf-8"))
config["model"]["llm"]["global"].update(
    {"url": llm_url, "model_name": llm_model, "key": "EMPTY", "topk": None}
)
config["model"]["embedding"]["global"].update(
    {"url": embedding_url, "model_name": embedding_model, "key": "EMPTY"}
)
pathlib.Path(output).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

set_phase() {
  curl -fsS -X POST "http://127.0.0.1:$PROXY_PORT/control/method/$1" >/dev/null
  sleep 2
}

set_phase memforest
(
  cd "$ROOT/MemoryForest"
  PYTHONPATH=. PYTHONUNBUFFERED=1 "$PYTHON" scripts/bench_memforest_performance.py \
    --dataset "$RUN_DIR/mini_locomo_flat.json" \
    --config "$RUN_DIR/memforest_config.yaml" \
    --model-label 30b \
    --qids "$QIDS" \
    --out-json "$RUN_DIR/memforest/result.json" \
    --work-dir "$RUN_DIR/memforest/workdir" \
    --local-chat-url "http://127.0.0.1:$PROXY_PORT/v1" \
    --local-chat-model "$MODEL" \
    --answer-prompt default \
    --search-mode recommended \
    --group-by-field locomo_sample_id \
    --group-query-workers 128 \
    > "$RUN_DIR/memforest/run.log" 2>&1
)

set_phase evermemos
(
  cd "$ROOT/EverMemOS"
  VECTORIZE_PROVIDER=vllm \
  VECTORIZE_API_KEY=EMPTY \
  VECTORIZE_BASE_URL="http://127.0.0.1:$EMBED_PORT/v1" \
  VECTORIZE_MODEL="$EMBED_MODEL" \
  VECTORIZE_DIMENSIONS=1024 \
  VECTORIZE_BATCH_SIZE=10 \
  VECTORIZE_MAX_CONCURRENT=128 \
  VECTORIZE_FALLBACK_PROVIDER=none \
  EVAL_LLM_BASE_URL="http://127.0.0.1:$PROXY_PORT/v1" \
  EVAL_LLM_MODEL="$MODEL" \
  EVAL_LLM_API_KEY=EMPTY \
  CORE_LLM_CONCURRENCY=128 \
  CORE_ADD_CONVERSATION_WORKERS=1 \
  CORE_EVENT_LOG_WORKERS_PER_CONVERSATION=128 \
  EVAL_NUM_WORKERS=128 \
  PYTHONPATH=.:src PYTHONUNBUFFERED=1 \
  "$PYTHON" -m evaluation.cli \
    --dataset performance_locomo_20 \
    --system evermemos_frontier_qwen30_c128 \
    --stages add search answer \
    --from-conv "$PROBE_CONV_INDEX" --to-conv "$((PROBE_CONV_INDEX + 1))" \
    --smoke --smoke-messages "$PROBE_MESSAGES" --smoke-questions 1 \
    --output-dir "$RUN_DIR/evermemos/output" \
    > "$RUN_DIR/evermemos/run.log" 2>&1
)

set_phase mem0
(
  cd "$ROOT/EverMemOS"
  DATASET_NAME=performance_locomo_20 \
  MEM0_STAGES="add search answer" \
  RUN_NAME="deepseek_cost_probe_${STAMP}" \
  OUTPUT_DIR="$RUN_DIR/mem0/output" \
  MEM0_RUNTIME_ROOT="$RUN_DIR/mem0/runtime" \
  TRACE_DIR="$RUN_DIR/mem0/trace" \
  MEM0_SERVER_PORT="$MEM0_PORT" \
  LLM_BASE_URL="http://127.0.0.1:$PROXY_PORT/v1" \
  LLM_MODEL="$MODEL" \
  LLM_API_KEY=EMPTY \
  EMBED_BASE_URL="http://127.0.0.1:$EMBED_PORT/v1" \
  EMBED_MODEL="$EMBED_MODEL" \
  EMBED_DIM=1024 \
  BASELINE_NUM_WORKERS=1 \
  EVAL_NUM_WORKERS=4 \
  FROM_CONV="$PROBE_CONV_INDEX" TO_CONV="$((PROBE_CONV_INDEX + 1))" \
  PYTHONUNBUFFERED=1 \
  SMOKE_MESSAGES="$PROBE_MESSAGES" SMOKE_QUESTIONS=1 \
  ./run_mem0_local_qwen3.sh smoke \
    > "$RUN_DIR/mem0/run.log" 2>&1
)

set_phase memoryos
"$PYTHON" "$ROOT/MemForest_Sigmod/artifact/scripts/run_memoryos_prefix_probe.py" \
  --dataset "$RUN_DIR/mini_locomo_flat.json" \
  --output-dir "$RUN_DIR/memoryos" \
  --base-url "http://127.0.0.1:$PROXY_PORT/v1" \
  --embedding-base-url "http://127.0.0.1:$EMBED_PORT/v1" \
  --model "$MODEL" \
  --embedding-model "$EMBED_MODEL" \
  > "$RUN_DIR/memoryos.log" 2>&1

docker run -d \
  --name "$NEO4J_CONTAINER" \
  -p "127.0.0.1:$NEO4J_HTTP_PORT:7474" \
  -p "127.0.0.1:$NEO4J_BOLT_PORT:7687" \
  -v "$RUN_DIR/neo4j/data:/data" \
  -v "$RUN_DIR/neo4j/logs:/logs" \
  -e NEO4J_AUTH=neo4j/zep-local-revision \
  -e NEO4J_server_memory_heap_initial__size=1G \
  -e NEO4J_server_memory_heap_max__size=4G \
  neo4j:5.26.2 >/dev/null
wait_url "http://127.0.0.1:$NEO4J_HTTP_PORT" neo4j

set_phase zep_local
PYTHONPATH="$ROOT/external/graphiti-0.24.1:$ROOT/external/MemoryData" \
  "$ROOT/.venvs/zep-local/bin/python" \
  "$ROOT/MemoryForest/scripts/zep_local/run_benchmark.py" \
    --stage run \
    --benchmark locomo \
    --model-key qwen30b \
    --run-root "$RUN_DIR/zep" \
    --run-name deepseek_v4_flash \
    --data-root "$RUN_DIR/zep_data" \
    --llm-url "http://127.0.0.1:$PROXY_PORT/v1" \
    --embedding-url "http://127.0.0.1:$EMBED_PORT/v1" \
    --neo4j-uri "bolt://127.0.0.1:$NEO4J_BOLT_PORT" \
    --concurrency 1 \
    --source-id "$PROBE_SOURCE_ID" \
    --max-rows 1 \
    > "$RUN_DIR/zep.log" 2>&1

"$PYTHON" "$ROOT/MemForest_Sigmod/artifact/scripts/summarize_deepseek_cost_probe.py" \
  --trace "$RUN_DIR/proxy/llm_usage.jsonl" \
  --output "$RUN_DIR/deepseek_cost_summary.csv" \
  --validation "$RUN_DIR/validation.json"

"$PYTHON" - "$RUN_DIR" "$PROBE_SOURCE_ID" "$PROBE_CONV_INDEX" <<'PY'
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

run_dir = pathlib.Path(sys.argv[1])
source_id = sys.argv[2]
conversation_index = int(sys.argv[3])
dataset = run_dir / "mini_locomo_flat.json"
rows = json.loads(dataset.read_text(encoding="utf-8"))
question_id = str(rows[0]["question_id"])
manifest = {
    "protocol_id": "deepseek_v4_flash_cost_probe_locomo_20messages_v2",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "scope": f"first 20 messages of {source_id} plus one frozen retrieval and answer",
    "conversation_index": conversation_index,
    "source_id": source_id,
    "question_id": question_id,
    "model": "deepseek-v4-flash",
    "thinking": "disabled",
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
    "cache_isolation": "fresh DeepSeek user_id namespace per method",
    "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
    "methods": ["MemForest", "EverMemOS", "Mem0", "MemoryOS", "Zep Local"],
    "secrets_persisted": False,
}
(run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY

printf 'Completed DeepSeek cost probe: %s\n' "$RUN_DIR"
