#!/usr/bin/env bash
# 统一训练入口：
#   bash scripts/train.sh [config]            # 正常训练
#   bash scripts/train.sh [config] --resume    # 断点续训
set -euo pipefail

CONFIG="${1:-configs/train/lora_qwen7b_industrial.yaml}"
RESUME_FLAG="${2:-}"

if [[ ! -f "$CONFIG" ]]; then
  echo "config not found: $CONFIG" >&2
  exit 1
fi

if [[ "$RESUME_FLAG" == "--resume" ]]; then
  python -m industrial_ft.train.sft --config "$CONFIG" --resume
else
  python -m industrial_ft.train.sft --config "$CONFIG"
fi
