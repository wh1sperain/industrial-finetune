#!/usr/bin/env bash
# 下载公开训练数据集：bash scripts/download_data.sh [dataset]
# 默认：Mining-Engineering-SFT（M1 定案数据，见 data/README.md）
set -euo pipefail

DATASET="${1:-acnul/Mining-Engineering-SFT}"

# Xet 下载协议在部分网络下有 401 bug（github.com/huggingface/xet-core#404），强制回退传统 HTTP
export HF_HUB_DISABLE_XET=1
# 若 HF 直连慢/失败，可启用国内镜像（二选一，勿同时用）：
# export HF_ENDPOINT=https://hf-mirror.com

python -m industrial_ft.data.loader --dataset "$DATASET"
