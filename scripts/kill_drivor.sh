#!/usr/bin/env bash
set -u

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PID_FILE="${PID_FILE:-$REPO_ROOT/run/drivor.pid}"
GRACE_SECONDS="${1:-10}"
USER_ID="$(id -u)"

stop_process_group() {
  [[ -f "$PID_FILE" ]] || return 0

  local root_pid pgid command
  root_pid="$(tr -d '[:space:]' < "$PID_FILE")"

  if [[ ! "$root_pid" =~ ^[0-9]+$ ]] || ! kill -0 "$root_pid" 2>/dev/null; then
    echo "[WARN] Stale PID file: $PID_FILE"
    rm -f "$PID_FILE"
    return 0
  fi

  command="$(ps -p "$root_pid" -o args= 2>/dev/null || true)"
  if [[ "$command" != *"nav1_train_frozen_backbones.sh"* &&
        "$command" != *"run_training_full.py"* ]]; then
    echo "[ERROR] PID $root_pid no longer belongs to DrivoR: $command" >&2
    return 1
  fi

  pgid="$(ps -p "$root_pid" -o pgid= | tr -d '[:space:]')"
  echo "[INFO] Sending TERM to DrivoR process group $pgid"
  kill -TERM -- "-$pgid" 2>/dev/null || true

  for ((second = 0; second < GRACE_SECONDS; second++)); do
    if ! pgrep -g "$pgid" >/dev/null 2>&1; then
      rm -f "$PID_FILE"
      return 0
    fi
    sleep 1
  done

  echo "[WARN] Force-killing DrivoR process group $pgid"
  kill -KILL -- "-$pgid" 2>/dev/null || true
  rm -f "$PID_FILE"
}

kill_known_orphans() {
  local patterns=(
    '[r]un_training_full.py'
    '[n]av1_train_frozen_backbones.sh'
    '[r]un_pdm_score_multi_gpu.py'
    '[p]t_data_worker'
  )

  echo "[INFO] Cleaning known orphan processes"
  for pattern in "${patterns[@]}"; do
    pkill -TERM -u "$USER_ID" -f "$pattern" 2>/dev/null || true
  done

  sleep 2

  for pattern in "${patterns[@]}"; do
    pkill -KILL -u "$USER_ID" -f "$pattern" 2>/dev/null || true
  done
}

stop_ray() {
  local ray_bin=""

  if command -v ray >/dev/null 2>&1; then
    ray_bin="$(command -v ray)"
  elif [[ -x "$HOME/.conda/envs/drivoR/bin/ray" ]]; then
    ray_bin="$HOME/.conda/envs/drivoR/bin/ray"
  fi

  if [[ -n "$ray_bin" ]]; then
    echo "[INFO] Stopping Ray with $ray_bin"
    "$ray_bin" stop --force >/dev/null 2>&1 || true
  fi

  pkill -TERM -u "$USER_ID" -f \
    '[r]aylet|[g]cs_server|[d]efault_worker.py|[p]lasma_store|[r]untime_env_agent' \
    2>/dev/null || true

  sleep 2

  pkill -KILL -u "$USER_ID" -f \
    '[r]aylet|[g]cs_server|[d]efault_worker.py|[p]lasma_store|[r]untime_env_agent' \
    2>/dev/null || true
}

show_remaining_processes() {
  echo "[INFO] Remaining related processes:"
  pgrep -afu "$USER_ID" \
    'nav1_train_frozen_backbones|run_training_full|run_pdm_score_multi_gpu|pt_data_worker|raylet|gcs_server|default_worker.py|plasma_store|runtime_env_agent' \
    || true

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[INFO] Remaining GPU compute processes:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader 2>/dev/null || true
  fi
}

echo "[INFO] Stopping DrivoR..."
stop_process_group
kill_known_orphans
stop_ray
show_remaining_processes
echo "[INFO] Cleanup completed."