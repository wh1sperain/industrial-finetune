#!/usr/bin/env bash
# 下载数据集（Mining-Engineering 系列）
# 用法：bash scripts/download_data.sh [sft|cot|eval|probe]，缺省下载全部
set -euo pipefail

# Xet 下载协议在部分网络下有 401 bug（github.com/huggingface/xet-core#404），回退传统 HTTP
export HF_HUB_DISABLE_XET=1
# 直连 HF 慢时可启用镜像：export HF_ENDPOINT=https://hf-mirror.com

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
  for entry in "${DATASETS[@]}"; do
    download_one "${entry%%:*}"
  done
else
  download_one "$1"
fi

echo "[download_data] 全部完成"
