#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EVERMEMOS_ROOT="${EVERMEMOS_ROOT:-$(pwd)}"

test -f "$EVERMEMOS_ROOT/evaluation/cli.py" || {
  echo "Set EVERMEMOS_ROOT to the pinned EverMemOS checkout" >&2
  exit 2
}

if git -C "$EVERMEMOS_ROOT" apply --check "$BASELINE_DIR/upstream.patch"; then
  git -C "$EVERMEMOS_ROOT" apply "$BASELINE_DIR/upstream.patch"
elif ! git -C "$EVERMEMOS_ROOT" apply --reverse --check "$BASELINE_DIR/upstream.patch"; then
  echo "EverMemOS checkout is neither clean nor already patched" >&2
  exit 3
fi

cp "$BASELINE_DIR/configs/locomo_all.yaml" \
  "$EVERMEMOS_ROOT/evaluation/config/datasets/locomo_all.yaml"
for config in "$BASELINE_DIR"/configs/evermemos_*.yaml; do
  cp "$config" "$EVERMEMOS_ROOT/evaluation/config/systems/$(basename "$config")"
done
