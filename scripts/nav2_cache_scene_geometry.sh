#!/usr/bin/env bash
# 当前帧几何标签缓存：配置环境后，默认通过 nohup 启动 72 个 worker。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
REPO_ROOT="$(cd -- "$REPO_ROOT" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3}"
DATA_ROOT="${DATA_ROOT:-/mnt/workspace/hru4sgh/NAVSIM/dataset}"
CACHE_WORKERS="${CACHE_WORKERS:-72}"
CACHE_WORKER_THREADS="${CACHE_WORKER_THREADS:-1}"
GEOMETRY_CACHE_ROOT="${GEOMETRY_CACHE_ROOT:-$REPO_ROOT/dataset/scene_geometry_v1}"
BACKGROUND="${BACKGROUND:-1}"

if [[ ! "$CACHE_WORKERS" =~ ^[1-9][0-9]*$ || ! "$CACHE_WORKER_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  printf '[ERROR] CACHE_WORKERS 和 CACHE_WORKER_THREADS 必须为正整数。\n' >&2
  exit 1
fi
if ! PYTHON_EXECUTABLE="$(command -v -- "$PYTHON_BIN")"; then
  printf '[ERROR] 找不到 Python: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi

cd "$REPO_ROOT"
REPO_ROOT="$PWD"
# 直接调用环境内的 Python，无需手动 conda activate。
export PATH="$(dirname -- "$PYTHON_EXECUTABLE"):$PATH"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/nuplan-devkit:${PYTHONPATH:-}"
export NAVSIM_DEVKIT_ROOT="$REPO_ROOT"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$REPO_ROOT/exp}"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DATA_ROOT/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
# 此任务仅生成 CPU 标签；限制每个进程的计算线程，避免线程数成倍增加。
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="$CACHE_WORKER_THREADS"
export OPENBLAS_NUM_THREADS="$CACHE_WORKER_THREADS"
export MKL_NUM_THREADS="$CACHE_WORKER_THREADS"
export NUMEXPR_NUM_THREADS="$CACHE_WORKER_THREADS"

for required_path in "$NUPLAN_MAPS_ROOT" "$OPENSCENE_DATA_ROOT/navsim_logs/trainval"; do
  if [[ ! -d "$required_path" ]]; then
    printf '[ERROR] 必需的数据目录不存在: %s\n' "$required_path" >&2
    exit 1
  fi
done

COMMAND=("$PYTHON_EXECUTABLE" "$REPO_ROOT/navsim/planning/script/run_scene_geometry_caching.py"
  agent=drivoR_scene_geometry
  experiment_name=scene_geometry_cache
  train_test_split=navtrain
  "agent.config.scene_geometry.cache_root=$GEOMETRY_CACHE_ROOT"
  "agent.config.scene_geometry.cache_num_workers=$CACHE_WORKERS"
  "agent.config.scene_geometry.cache_worker_threads=$CACHE_WORKER_THREADS"
  "$@")

# 配置查看等操作直接输出到当前终端。
for argument in "$@"; do
  case "$argument" in
    --cfg|--cfg=*|--help|-h|--hydra-help|--info|--info=*|--version) BACKGROUND=0 ;;
  esac
done
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
LOG_FILE="${LOG_FILE:-$NAVSIM_EXP_ROOT/logs/scene_geometry_cache_${RUN_ID}.log}"
PID_FILE="${PID_FILE:-${LOG_FILE%.log}.pid}"

print_environment() {
  printf '[INFO] Python: %s\n[INFO] 仓库: %s\n' "$PYTHON_EXECUTABLE" "$REPO_ROOT"
  printf '[INFO] 数据: %s\n[INFO] 地图: %s\n' "$OPENSCENE_DATA_ROOT" "$NUPLAN_MAPS_ROOT"
  printf '[INFO] 缓存: %s\n[INFO] Workers: %s，每个 worker 线程数: %s\n' \
    "$GEOMETRY_CACHE_ROOT" "$CACHE_WORKERS" "$CACHE_WORKER_THREADS"
}
print_environment
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  if [[ "$BACKGROUND" == 1 ]]; then printf 'nohup '; fi
  printf '%q ' "${COMMAND[@]}"
  if [[ "$BACKGROUND" == 1 ]]; then
    printf '>> %q 2>&1 < /dev/null &\n' "$LOG_FILE"
    printf '[INFO] PID 文件: %s\n' "$PID_FILE"
  else
    printf '\n'
  fi
  exit 0
fi
if [[ "$BACKGROUND" != 1 ]]; then
  exec "${COMMAND[@]}"
fi

mkdir -p -- "$(dirname -- "$LOG_FILE")" "$(dirname -- "$PID_FILE")"
{
  printf '\n[INFO] 启动时间: %s\n' "$(date -Is)"
  print_environment
  printf '[INFO] 命令: '; printf '%q ' "${COMMAND[@]}"; printf '\n'
} >> "$LOG_FILE"
nohup "${COMMAND[@]}" >> "$LOG_FILE" 2>&1 < /dev/null &
CACHE_PID=$!
printf '%s\n' "$CACHE_PID" > "$PID_FILE"
printf '[INFO] 已提交，PID: %s\n[INFO] 日志: %s\n[INFO] PID 文件: %s\n' \
  "$CACHE_PID" "$LOG_FILE" "$PID_FILE"
printf '[INFO] 查看进度: tail -f %q\n' "$LOG_FILE"
