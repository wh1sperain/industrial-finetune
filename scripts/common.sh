#!/usr/bin/env bash
# 公共脚本：仓库根目录、Python 解释器、项目环境变量、实验配置校验。
# 仅供其他脚本 source，不直接执行；本文件不要设置 set -euo pipefail（会污染调用者 shell）。

# 解析仓库根目录（兼容 symlink；优先 readlink -f，缺失时用 python 兜底）
resolve_repo_root() {
  local script_dir root
  if [[ -n "${INDUSTRIAL_FT_ROOT:-}" ]]; then
    REPO_ROOT="$INDUSTRIAL_FT_ROOT"
    return 0
  fi
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  root="$(cd "$script_dir/.." && pwd)"
  if command -v readlink >/dev/null 2>&1 && readlink -f "$root" >/dev/null 2>&1; then
    root="$(readlink -f "$root")"
  elif command -v python3 >/dev/null 2>&1; then
    root="$(python3 -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$root")"
  elif command -v python >/dev/null 2>&1; then
    root="$(python -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$root")"
  fi
  REPO_ROOT="$root"
}

# 选择 Python：INDUSTRIAL_FT_PYTHON > .venv/bin/python > python3 > python
resolve_python() {
  if [[ -n "${INDUSTRIAL_FT_PYTHON:-}" ]]; then
    PYTHON="$INDUSTRIAL_FT_PYTHON"
  elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
  else
    echo "[common] 错误: 未找到 Python（可设置 INDUSTRIAL_FT_PYTHON 或先激活 venv）" >&2
    return 1
  fi
}

# 加载项目环境：.env + 默认 HF 网络策略
load_project_env() {
  if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
  fi
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  export HF_TOKEN="${HF_TOKEN:-}"
  export WANDB_API_KEY="${WANDB_API_KEY:-}"
}

# 返回某代码级实验的配置目录
experiments_dir() {
  local code_experiment="${1:-${INDUSTRIAL_FT_CODE_EXPERIMENT:-}}"
  if [[ -n "${INDUSTRIAL_FT_EXPERIMENTS_DIR:-}" ]]; then
    echo "$INDUSTRIAL_FT_EXPERIMENTS_DIR"
    return 0
  fi
  if [[ -z "$code_experiment" ]]; then
    echo "[common] 错误: 未指定代码级实验名（设置 INDUSTRIAL_FT_CODE_EXPERIMENT 或传入参数）" >&2
    return 1
  fi
  echo "$REPO_ROOT/configs/experiments/$code_experiment"
}

# 校验实验配置存在：ensure_experiment_yaml <配置名> [代码级实验名]
ensure_experiment_yaml() {
  local name="${1:-}"
  local code_experiment="${2:-${INDUSTRIAL_FT_CODE_EXPERIMENT:-}}"
  local dir
  if [[ -z "$name" ]]; then
    echo "[common] 错误: 缺少实验配置名" >&2
    return 1
  fi
  dir="$(experiments_dir "$code_experiment")" || return 1
  if [[ ! -f "$dir/$name.yaml" ]]; then
    echo "[common] 错误: 实验配置不存在: $dir/$name.yaml" >&2
    echo "可用配置:" >&2
    list_experiments "$code_experiment" >&2 || true
    return 1
  fi
}

# 列出某代码级实验的配置名：list_experiments [代码级实验名]
list_experiments() {
  local code_experiment="${1:-${INDUSTRIAL_FT_CODE_EXPERIMENT:-}}"
  local dir
  dir="$(experiments_dir "$code_experiment")" || return 1
  if [[ -d "$dir" ]]; then
    find "$dir" -maxdepth 1 -name '*.yaml' -print0 2>/dev/null \
      | xargs -0 -n1 basename 2>/dev/null \
      | sed 's/\.yaml$//' \
      | sort
  fi
}
