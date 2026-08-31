#!/usr/bin/env bash
# 下载全部所需数据集（Mining-Engineering 系列，见 data/README.md）
# 用法：
#   bash scripts/download_data.sh           # 下载全部 4 个数据集（训练/CoT/评测/探针）
#   bash scripts/download_data.sh sft       # 只下载指定数据集（可选：sft / cot / eval / probe）
set -euo pipefail

# Xet 下载协议在部分网络下有 401 bug（github.com/huggingface/xet-core#404），强制回退传统 HTTP
export HF_HUB_DISABLE_XET=1
# 若 HF 直连慢/失败，可启用国内镜像（二选一，勿同时用）：
# export HF_ENDPOINT=https://hf-mirror.com

# 数据集清单：<key>:<HF 路径>
DATASETS=(
  "sft:acnul/Mining-Engineering-SFT"
  "cot:acnul/Mining-Engineering-SFT-CoT"
  "eval:acnul/Mining-Engineering-Eval"
  "probe:acnul/Mining-Engineering-Probe"
)

download_one() {
  local key="$1"
  for entry in "${DATASETS[@]}"; do
    local k="${entry%%:*}"
    local path="${entry#*:}"
    if [[ "$k" == "$key" ]]; then
      echo "[download_data] 下载 $path ..."
      python -m industrial_ft.data.loader --dataset "$path"
      echo "[download_data] $key 完成"
      return 0
    fi
  done
  echo "[download_data] 未知数据集: $key（可选: sft / cot / eval / probe）" >&2
  return 1
}

if [[ $# -eq 0 ]]; then
  # 默认：全部下载
  for entry in "${DATASETS[@]}"; do
    download_one "${entry%%:*}"
  done
else
  download_one "$1"
fi

echo "[download_data] 全部完成"
