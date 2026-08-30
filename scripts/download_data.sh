#!/usr/bin/env bash
# 下载公开数据集：bash scripts/download_data.sh [dataset]
set -euo pipefail

DATASET="${1:-gsm8k}"
python -m industrial_ft.data.loader --dataset "$DATASET"
