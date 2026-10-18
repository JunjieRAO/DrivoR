#!/usr/bin/env bash
set -euo pipefail

# NAVSIM-v2 README recipe, with the environment/nohup setup from nav2's
# scripts/nav2_train_safety.sh. No safety/geometry-only agent arguments.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/mnt/workspace/hru4sgh/NAVSIM/dataset}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_drivoR_Nav2_10epochs}"
BACKGROUND="${BACKGROUND:-1}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/nuplan-devkit:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export NAVSIM_DEVKIT_ROOT="$REPO_ROOT"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$REPO_ROOT/exp}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DATA_ROOT/maps}"
export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARRAY[@]}"
TRAINER_STRATEGY=auto
if (( NUM_GPUS > 1 )); then TRAINER_STRATEGY=ddp; fi

COMMAND=("$PYTHON_BIN" "$REPO_ROOT/navsim/planning/script/run_training.py"
  agent=drivoR
  "experiment_name=$EXPERIMENT_NAME"
  train_test_split=navtrain
  cache_path=null
  use_cache_without_dataset=false
  "trainer.params.max_epochs=${MAX_EPOCHS:-10}"
  "+trainer.params.devices=$NUM_GPUS"
  "trainer.params.strategy=$TRAINER_STRATEGY"
  dataloader.params.prefetch_factor=1
  "dataloader.params.batch_size=${BATCH_SIZE:-16}"
  "dataloader.params.num_workers=${DATALOADER_WORKERS:-16}"
  agent.lr_args.name=AdamW
  "agent.lr_args.base_lr=${BASE_LR:-0.0002}"
  "agent.num_gpus=$NUM_GPUS"
  agent.progress_bar=false
  agent.config.refiner_ls_values=0.0
  agent.config.image_backbone.focus_front_cam=false
  agent.config.one_token_per_traj=true
  agent.config.refiner_num_heads=1
  agent.config.tf_d_model=256
  agent.config.tf_d_ffn=1024
  agent.config.area_pred=false
  agent.config.agent_pred=false
  agent.config.ref_num=4
  agent.loss.prev_weight=0.0
  "seed=${SEED:-2}"
  "$@")

# README omits the continuation after prev_weight; seed=2 is intentionally
# passed to Hydra here. Other model/loss/precision settings use branch defaults.
# This branch automatically resumes the latest checkpoint of the same
# experiment. Use a new EXPERIMENT_NAME when you want a fresh training run.
INSPECT_ONLY=0
for argument in "$@"; do
  case "$argument" in
    --cfg|--cfg=*|--help|-h|--hydra-help|--info|--info=*|--version)
      INSPECT_ONLY=1; BACKGROUND=0 ;;
  esac
done
LOG_FILE="${LOG_FILE:-$NAVSIM_EXP_ROOT/logs/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)_$$.log}"
printf '[INFO] Python: %s\n[INFO] GPUs: %s\n' "$PYTHON_BIN" "$GPU_IDS"
printf '[INFO] Same experiment name retains the original automatic-resume behavior.\n'
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n[INFO] Background=%s; log=%s\n' "$BACKGROUND" "$LOG_FILE"
  exit 0
fi
if [[ "$INSPECT_ONLY" == 1 ]]; then exec "${COMMAND[@]}"; fi

command -v "$PYTHON_BIN" >/dev/null || { printf '[ERROR] Python not found: %s\n' "$PYTHON_BIN" >&2; exit 1; }
for path in "$NUPLAN_MAPS_ROOT" "$DATA_ROOT/navsim_logs/trainval" \
  "$DATA_ROOT/sensor_blobs/trainval" "$REPO_ROOT/nuplan-devkit" \
  "$REPO_ROOT/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"; do
  [[ -e "$path" ]] || { printf '[ERROR] Required path not found: %s\n' "$path" >&2; exit 1; }
done

# nav2-base hardcodes this cache location; do not overwrite an existing entry.
METRIC_CACHE="$NAVSIM_EXP_ROOT/train_metric_cache"
if [[ ! -e "$METRIC_CACHE" && ! -L "$METRIC_CACHE" ]]; then
  CACHE_SOURCE="${TRAIN_METRIC_CACHE_PATH:-$REPO_ROOT/dataset/train_metric_cache_navtrain}"
  [[ -d "$CACHE_SOURCE" ]] || { printf '[ERROR] Metric cache not found: %s\n' "$CACHE_SOURCE" >&2; exit 1; }
  CACHE_SOURCE="$(cd -- "$CACHE_SOURCE" && pwd)"
  mkdir -p -- "$NAVSIM_EXP_ROOT"
  ln -s -- "$CACHE_SOURCE" "$METRIC_CACHE"
fi
[[ -d "$METRIC_CACHE" ]] || { printf '[ERROR] Invalid metric cache: %s\n' "$METRIC_CACHE" >&2; exit 1; }
printf '[INFO] Metric cache: %s\n' "$METRIC_CACHE"

if [[ "$BACKGROUND" != 1 ]]; then exec "${COMMAND[@]}"; fi
mkdir -p -- "$(dirname -- "$LOG_FILE")"
nohup "${COMMAND[@]}" >"$LOG_FILE" 2>&1 < /dev/null &
printf '[INFO] Submitted PID: %s\n[INFO] Log: %s\n' "$!" "$LOG_FILE"
