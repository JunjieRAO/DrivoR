#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
# 可直接启动，无需预先激活 Conda；PY 优先，兼容原有 PYTHON_BIN 配置。
# 原有 metric cache 包含 NumPy 2.x 序列化路径，使用已验证的系统环境。
export PY="${PY:-${PYTHON_BIN:-/usr/local/bin/python3}}"
if ! PY="$(command -v -- "$PY")" || [[ ! -x "$PY" ]]; then
  echo '[ERROR] Python 不存在或不可执行，请通过 PY 指定解释器。' >&2
  exit 1
fi
export PYTHON_BIN="$PY"
export PATH="$(dirname -- "$PY"):$PATH"
GEOMETRY_CACHE_ROOT="${GEOMETRY_CACHE_ROOT:-$REPO_ROOT/dataset/scene_geometry_v1}"
MODE="${GEOMETRY_MODE:-gen}"
case "$MODE" in off|aux|gen|gen_scorer) ;; *) echo 'GEOMETRY_MODE must be off, aux, gen or gen_scorer' >&2; exit 1 ;; esac
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-drivor_sg_${MODE}}"
export DAC_WEIGHT=0.0 COLLISION_WEIGHT=0.0
printf '[INFO] Python: %s\n[INFO] Geometry cache: %s\n' "$PY" "$GEOMETRY_CACHE_ROOT"
# Inherit B0 data, initialization, seed, batch size, LR and DDP settings.
exec bash "$SCRIPT_DIR/nav2_train_safety.sh" \
  agent=drivoR_scene_geometry \
  "agent.config.scene_geometry.mode=$MODE" \
  "agent.config.scene_geometry.cache_root=$GEOMETRY_CACHE_ROOT" \
  "agent.config.scene_geometry.cache_mode=${GEOMETRY_CACHE_MODE:-strict}" \
  "$@"
