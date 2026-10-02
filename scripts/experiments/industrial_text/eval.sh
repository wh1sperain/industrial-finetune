#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../common.sh"
resolve_repo_root
resolve_python
load_project_env
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_XET=1
cd "$REPO_ROOT"
exec "$PYTHON" -m industrial_text.evaluate --experiment "${1:?Usage: eval.sh <experiment> [--probe|--audit|--score-only]}" "${@:2}"
