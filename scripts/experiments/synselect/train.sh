#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../common.sh"
CODE_EXPERIMENT=synselect
resolve_repo_root
resolve_python
load_project_env
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <experiment> [--dry-run|--check-env|--benchmark-steps N|--resume]" >&2
  exit 1
fi
CONFIG="$1"
shift
ensure_experiment_yaml "$CONFIG" "$CODE_EXPERIMENT"
cd "$REPO_ROOT"
exec "$PYTHON" -m synselect.train --experiment "$CONFIG" "$@"
