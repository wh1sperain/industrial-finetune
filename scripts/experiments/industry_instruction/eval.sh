#!/usr/bin/env bash
# industry_instruction 代码级实验：评测入口
# 用法: bash scripts/experiments/industry_instruction/eval.sh <配置名> [args...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../common.sh
source "$SCRIPT_DIR/../../common.sh"

CODE_EXPERIMENT=industry_instruction
export INDUSTRIAL_FT_CODE_EXPERIMENT="$CODE_EXPERIMENT"

resolve_repo_root
resolve_python
load_project_env

if [[ $# -lt 1 ]]; then
  echo "用法: $0 <配置名> [args...]" >&2
  echo "可用配置:" >&2
  list_experiments "$CODE_EXPERIMENT" >&2 || true
  exit 1
fi

CONFIG="$1"
shift
ensure_experiment_yaml "$CONFIG" "$CODE_EXPERIMENT"

cd "$REPO_ROOT"
exec "$PYTHON" -m industry_instruction.eval.run_eval --experiment "$CONFIG" "$@"
