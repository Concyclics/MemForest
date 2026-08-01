#!/usr/bin/env bash
set -euo pipefail

ROOT=/ssd2/chenhan/MemBench
STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
CACHE_NAMESPACE="${CACHE_NAMESPACE:-mf-cost-warm-${STAMP//_/-}}"
CACHE_SETTLE_SECONDS="${CACHE_SETTLE_SECONDS:-20}"
WARMUP_CONV_INDEX="${WARMUP_CONV_INDEX:-3}"
RUNS=()

: "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY is required}"
trap 'tmux set-environment -gu DEEPSEEK_API_KEY 2>/dev/null || true' EXIT

# Prime each native pipeline under the same provider user_id used by the
# measured runs. This conversation is excluded from aggregation.
warm_stamp="${STAMP}_warmup_conv${WARMUP_CONV_INDEX}"
PROBE_CONV_INDEX="$WARMUP_CONV_INDEX" EMBED_GPU=3 \
  ISOLATION_PREFIX_OVERRIDE="$CACHE_NAMESPACE" \
"$ROOT/scripts/run_deepseek_cost_probe.sh" "$warm_stamp"

/home/chenhan/miniconda3/envs/agent/bin/python \
  "$ROOT/MemForest_Sigmod/artifact/scripts/prime_deepseek_system_prompts.py" \
  --prompt-log "$ROOT/runs/revision_deepseek_cost_probe_${warm_stamp}/proxy/system_prompts.jsonl" \
  --output "$ROOT/runs/revision_deepseek_cost_probe_${warm_stamp}/proxy/system_prompt_prime_validation.json" \
  --isolation-prefix "$CACHE_NAMESPACE" \
  --settle-seconds "$CACHE_SETTLE_SECONDS"

for index in 0 1 2; do
  run_stamp="${STAMP}_conv${index}"
  PROBE_CONV_INDEX="$index" EMBED_GPU=3 \
    ISOLATION_PREFIX_OVERRIDE="$CACHE_NAMESPACE" \
    "$ROOT/scripts/run_deepseek_cost_probe.sh" "$run_stamp"
  RUNS+=("$ROOT/runs/revision_deepseek_cost_probe_${run_stamp}")
done

OUTPUT_ROOT="$ROOT/runs/revision_deepseek_cost_probe_${STAMP}_warm_aggregate"
mkdir -p "$OUTPUT_ROOT"
/home/chenhan/miniconda3/envs/agent/bin/python \
  "$ROOT/MemForest_Sigmod/artifact/scripts/aggregate_deepseek_cost_probes.py" \
  --run "${RUNS[0]}" \
  --run "${RUNS[1]}" \
  --run "${RUNS[2]}" \
  --detail-output "$OUTPUT_ROOT/deepseek_cost_probe_detail.csv" \
  --summary-output "$OUTPUT_ROOT/deepseek_cost_probe_summary.csv" \
  --validation-output "$OUTPUT_ROOT/validation.json" \
  --manifest-output "$OUTPUT_ROOT/manifest.json" \
  --warmup-source conv-47

/home/chenhan/miniconda3/envs/agent/bin/python - "$OUTPUT_ROOT/deepseek_cost_probe_detail.csv" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
missing = sorted(
    {row["method"] for row in rows if int(row["cache_hit_input_tokens"]) <= 0}
)
if missing:
    raise SystemExit(
        "warm-cache validation failed; zero provider cache hits for: "
        + ", ".join(missing)
    )
PY

printf 'Completed warm-cache three-conversation DeepSeek cost probe: %s\n' "$OUTPUT_ROOT"
