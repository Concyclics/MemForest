#!/usr/bin/env bash
set -euo pipefail

ROOT=/ssd2/chenhan/MemBench
STAMP="${1:-20260801_warmcache_v2}"
CACHE_NAMESPACE="${CACHE_NAMESPACE:-mf-cost-warm-${STAMP//_/-}}"
RUNS=()

: "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY is required}"
trap 'tmux set-environment -gu DEEPSEEK_API_KEY 2>/dev/null || true' EXIT

# A second native warmup conversation ensures role-agnostic repeated prefixes
# (including systems that encode instructions in a user message) are exercised.
warm_stamp="${STAMP}_warmup_conv4"
PROBE_CONV_INDEX=4 EMBED_GPU=3 \
  ISOLATION_PREFIX_OVERRIDE="$CACHE_NAMESPACE" \
  "$ROOT/scripts/run_deepseek_cost_probe.sh" "$warm_stamp"

/home/chenhan/miniconda3/envs/agent/bin/python - \
  "$ROOT/runs/revision_deepseek_cost_probe_${warm_stamp}/deepseek_cost_summary.csv" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
missing = sorted(row["method"] for row in rows if int(row["cache_hit_input_tokens"]) <= 0)
if missing:
    raise SystemExit("second warmup has zero provider hits for: " + ", ".join(missing))
PY

sleep 20
for index in 0 1 2; do
  run_stamp="${STAMP}_measured_conv${index}"
  PROBE_CONV_INDEX="$index" EMBED_GPU=3 \
    ISOLATION_PREFIX_OVERRIDE="$CACHE_NAMESPACE" \
    "$ROOT/scripts/run_deepseek_cost_probe.sh" "$run_stamp"
  RUNS+=("$ROOT/runs/revision_deepseek_cost_probe_${run_stamp}")
done

OUTPUT_ROOT="$ROOT/runs/revision_deepseek_cost_probe_${STAMP}_verified_warm_aggregate"
mkdir -p "$OUTPUT_ROOT"
/home/chenhan/miniconda3/envs/agent/bin/python \
  "$ROOT/MemForest_Sigmod/artifact/scripts/aggregate_deepseek_cost_probes.py" \
  --run "${RUNS[0]}" --run "${RUNS[1]}" --run "${RUNS[2]}" \
  --detail-output "$OUTPUT_ROOT/deepseek_cost_probe_detail.csv" \
  --summary-output "$OUTPUT_ROOT/deepseek_cost_probe_summary.csv" \
  --validation-output "$OUTPUT_ROOT/validation.json" \
  --manifest-output "$OUTPUT_ROOT/manifest.json" \
  --warmup-source conv-47 --warmup-source conv-48

/home/chenhan/miniconda3/envs/agent/bin/python - "$OUTPUT_ROOT/deepseek_cost_probe_detail.csv" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
missing = sorted({row["method"] for row in rows if int(row["cache_hit_input_tokens"]) <= 0})
if missing:
    raise SystemExit("measured warm-cache validation failed for: " + ", ".join(missing))
PY

printf 'Completed verified warm-cache DeepSeek probe: %s\n' "$OUTPUT_ROOT"
