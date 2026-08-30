#!/usr/bin/env bash
# 统一评测入口：bash scripts/eval.sh [config]
set -euo pipefail

CONFIG="${1:-configs/eval/industrial_benchmark.yaml}"

if [[ ! -f "$CONFIG" ]]; then
  echo "config not found: $CONFIG" >&2
  exit 1
fi

python -m industrial_ft.eval.run_eval --config "$CONFIG"
