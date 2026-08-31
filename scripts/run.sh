#!/usr/bin/env bash
# 兼容入口（即将废弃）：委托给代码级实验脚本。
# 推荐使用：
#   bash scripts/experiments/industrial_ft/train.sh <配置名> [args...]
#   bash scripts/experiments/industrial_ft/eval.sh <配置名> [args...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 2 ]]; then
  echo "用法: bash scripts/run.sh <配置名> <train|eval> [args...]" >&2
  echo "推荐: bash scripts/experiments/industrial_ft/<train|eval>.sh <配置名> [args...]" >&2
  exit 1
fi

CONFIG="$1"
MODE="$2"
shift 2

if [[ "$MODE" != "train" && "$MODE" != "eval" ]]; then
  echo "未知模式: $MODE（可选 train / eval）" >&2
  exit 1
fi

echo "[run.sh] 警告: run.sh 即将废弃，请改用 scripts/experiments/industrial_ft/${MODE}.sh" >&2
exec bash "$SCRIPT_DIR/experiments/industrial_ft/$MODE.sh" "$CONFIG" "$@"
