#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/nuplan-devkit${PYTHONPATH:+:${PYTHONPATH}}"
cd "$REPO_ROOT"
"${PYTHON_BIN:-python}" -m pytest tests/offline_search -q
"${PYTHON_BIN:-python}" -m navsim.offline_search.run demo --output "${1:?Specify a NEW demo output directory}"
