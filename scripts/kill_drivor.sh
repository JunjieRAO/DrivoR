#!/usr/bin/env bash

kill_drivor() {
  local grace_seconds="${1:-10}"

  echo "[INFO] Stopping DrivoR training and evaluation processes..."
  pkill -TERM -f '[n]av1_train_frozen_backbones.sh' 2>/dev/null || true
  pkill -TERM -f '[r]un_training_full.py' 2>/dev/null || true
  pkill -TERM -f '[n]av1_eval.sh' 2>/dev/null || true
  pkill -TERM -f '[r]un_pdm_score_multi_gpu.py' 2>/dev/null || true
  pkill -TERM -f '[p]t_data_worker' 2>/dev/null || true

  if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  fi

  echo "[INFO] Waiting ${grace_seconds}s before force cleanup..."
  sleep "$grace_seconds"

  pkill -KILL -f '[n]av1_train_frozen_backbones.sh' 2>/dev/null || true
  pkill -KILL -f '[r]un_training_full.py' 2>/dev/null || true
  pkill -KILL -f '[n]av1_eval.sh' 2>/dev/null || true
  pkill -KILL -f '[r]un_pdm_score_multi_gpu.py' 2>/dev/null || true
  pkill -KILL -f '[p]t_data_worker' 2>/dev/null || true
  pkill -KILL -f '[r]aylet' 2>/dev/null || true
  pkill -KILL -f '[g]cs_server' 2>/dev/null || true

  echo "[INFO] Remaining DrivoR/Ray processes:"
  pgrep -af 'nav1_train_frozen_backbones|run_training_full|nav1_eval|run_pdm_score_multi_gpu|pt_data_worker|raylet|gcs_server' || true

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[INFO] Remaining GPU compute processes:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader 2>/dev/null || true
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  kill_drivor "${1:-10}"
fi