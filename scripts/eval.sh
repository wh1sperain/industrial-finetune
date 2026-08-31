#!/usr/bin/env bash
# 统一评测入口：
#   bash scripts/eval.sh                    # 完整 Eval（301 条）
#   bash scripts/eval.sh --probe            # Probe 快速验证（50 条）
#   bash scripts/eval.sh --baseline         # 基座 zero-shot 对照
#   bash scripts/eval.sh --probe --baseline # Probe + 基座对照
set -euo pipefail

CONFIG="configs/eval/industrial_benchmark.yaml"
ARGS=()

for arg in "$@"; do
  if [[ "$arg" == *.yaml ]]; then
    CONFIG="$arg"
  else
    ARGS+=("$arg")
  fi
done

if [[ ! -f "$CONFIG" ]]; then
  echo "config not found: $CONFIG" >&2
  exit 1
fi

# Xet 下载协议在部分网络下有 401 bug（github.com/huggingface/xet-core#404），强制回退传统 HTTP
export HF_HUB_DISABLE_XET=1
# 若 HF 直连慢/失败，可启用国内镜像（二选一，勿同时用）：
# export HF_ENDPOINT=https://hf-mirror.com

python -m industrial_ft.eval.run_eval --config "$CONFIG" "${ARGS[@]}"
