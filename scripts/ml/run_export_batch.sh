#!/usr/bin/env bash
# Batch export player crops per M1 plan (run after benchmarks are ready).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
EXPORT="${ROOT}/scripts/ml/export_player_crops.py"

run_export() {
  local session="$1"
  local max="$2"
  echo "=== Export ${session} (max=${max}) ==="
  "$PY" "$EXPORT" --session "$session" --max-samples "$max"
}

run_export 7252 450
run_export 7559 400
run_export 7255 300
run_export 7515 600
run_export 7521 600
run_export 7125_7126 800

"$PY" "$EXPORT" --merge-split train --merge-only
"$PY" "$EXPORT" --merge-split val --merge-only
"$PY" "$EXPORT" --merge-split test --merge-only

echo "Batch export complete."
