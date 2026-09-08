#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../common.sh"
resolve_repo_root
resolve_python
load_project_env
cd "$REPO_ROOT"
exec "$PYTHON" -m synselect.mathverse --experiment mathverse_1h "$@"
