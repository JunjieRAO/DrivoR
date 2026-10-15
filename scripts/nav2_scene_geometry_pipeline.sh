#!/usr/bin/env bash
# 公共入口；通常使用 nav2_cache_two_stage.sh / nav2_eval_scene_geometry.sh / nav2_eval_scene_geometry_all.sh。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
export EVAL_REPO_ROOT="${EVAL_REPO_ROOT:-$(dirname -- "$REPO_ROOT")/navsim-v2}"
export PY="${PY:-${PYTHON_BIN:-/usr/local/bin/python3}}"
PY="$(command -v -- "$PY")"
[[ -x "$PY" ]] || { echo 'Python 不可执行' >&2; exit 1; }
export PATH="$(dirname -- "$PY"):$PATH"
# 必须优先导入评测仓库，避免训练仓库的 navsim 覆盖正式评测实现。
export PYTHONPATH="$EVAL_REPO_ROOT:$REPO_ROOT/nuplan-devkit"
export NAVSIM_DEVKIT_ROOT="$EVAL_REPO_ROOT"
export OPENSCENE_DATA_ROOT="${DATA_ROOT:-${OPENSCENE_DATA_ROOT:-/mnt/workspace/hru4sgh/NAVSIM/dataset}}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NAVSIM_EXP_ROOT="${EVAL_EXP_ROOT:-$EVAL_REPO_ROOT/exp}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0,1,2,3}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
STAGE="${1:-all}"
case "$STAGE" in cache|eval|all) ;; *) echo '阶段必须为 cache / eval / all' >&2; exit 2 ;; esac
[[ $# -le 1 ]] || { echo '请通过环境变量配置参数，脚本不接受额外位置参数。' >&2; exit 2; }
cd "$EVAL_REPO_ROOT"
COMMAND=("$PY" "$SCRIPT_DIR/scene_geometry_evaluation.py" "$STAGE")
if [[ "${DRY_RUN:-0}" == 1 || "${CHECK_ONLY:-0}" == 1 || "${BACKGROUND:-1}" == 0 ]]; then
  exec "${COMMAND[@]}"
fi
LOG_FILE="${LOG_FILE:-$NAVSIM_EXP_ROOT/logs/scene_geometry_${STAGE}_$(date +%Y%m%d_%H%M%S)_$$.log}"
mkdir -p -- "$(dirname -- "$LOG_FILE")"
nohup "${COMMAND[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "${LOG_FILE}.pid"
printf 'PID: %s\n日志: %s\n查看: tail -f %q\n' "$PID" "$LOG_FILE" "$LOG_FILE"
