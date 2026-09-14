#!/usr/bin/env bash
# Run in the existing DrivoR Python environment. Extra arguments are forwarded.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"
cd "$REPO_ROOT"
exec "${PYTHON_BIN:-python}" -m navsim.offline_search.run "$@"
