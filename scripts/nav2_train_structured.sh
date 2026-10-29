#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
export PYTHON_BIN="${PY:-${PYTHON_BIN:-/usr/local/bin/python3}}"
if ! PYTHON_BIN="$(command -v -- "$PYTHON_BIN")" || [[ ! -x "$PYTHON_BIN" ]]; then
    printf '[ERROR] Set PY or PYTHON_BIN to an executable Python interpreter.\n' >&2
    exit 1
fi
export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
FUSION="${STRUCTURED_FUSION:-both}"
ENABLED="${STRUCTURED_ENABLED:-true}"
case "$FUSION" in gen|scorer|both) ;; *) printf '[ERROR] STRUCTURED_FUSION must be gen, scorer or both.\n' >&2; exit 1 ;; esac
case "$ENABLED" in true|false) ;; *) printf '[ERROR] STRUCTURED_ENABLED must be true or false.\n' >&2; exit 1 ;; esac
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-nav2_structured_gt_${FUSION}}"
printf '[INFO] Python: %s\n' "$PYTHON_BIN"
printf '[INFO] Current-frame GT structured input: %s; fusion: %s. Privileged-input ablation.\n' "$ENABLED" "$FUSION"
exec bash "$SCRIPT_DIR/nav2_train_safety.sh" \
    agent=drivoR_structured \
    "agent.config.structured_input.enabled=$ENABLED" \
    "agent.config.structured_input.fusion=$FUSION" \
    "$@"