#!/usr/bin/env bash
# 统一训练入口：bash scripts/train.sh [config]
set -euo pipefail

CONFIG="${1:-configs/train/lora_qwen7b_industrial.yaml}"

if [[ ! -f "$CONFIG" ]]; then
  echo "config not found: $CONFIG" >&2
  exit 1
fi

python -m industrial_ft.train.sft --config "$CONFIG"
