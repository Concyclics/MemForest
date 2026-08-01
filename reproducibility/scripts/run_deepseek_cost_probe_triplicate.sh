#!/usr/bin/env bash
set -euo pipefail

ROOT=/ssd2/chenhan/MemBench
STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RUNS=()

: "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY is required}"

for index in 0 1 2; do
  run_stamp="${STAMP}_conv${index}"
  PROBE_CONV_INDEX="$index" EMBED_GPU=3 \
    "$ROOT/scripts/run_deepseek_cost_probe.sh" "$run_stamp"
  RUNS+=("$ROOT/runs/revision_deepseek_cost_probe_${run_stamp}")
done

OUTPUT_ROOT="$ROOT/runs/revision_deepseek_cost_probe_${STAMP}_aggregate"
mkdir -p "$OUTPUT_ROOT"
/home/chenhan/miniconda3/envs/agent/bin/python \
  "$ROOT/MemForest_Sigmod/artifact/scripts/aggregate_deepseek_cost_probes.py" \
  --run "${RUNS[0]}" \
  --run "${RUNS[1]}" \
  --run "${RUNS[2]}" \
  --detail-output "$OUTPUT_ROOT/deepseek_cost_probe_detail.csv" \
  --summary-output "$OUTPUT_ROOT/deepseek_cost_probe_summary.csv" \
  --validation-output "$OUTPUT_ROOT/validation.json" \
  --manifest-output "$OUTPUT_ROOT/manifest.json"

printf 'Completed three-conversation DeepSeek cost probe: %s\n' "$OUTPUT_ROOT"
