#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/mnt/workspace/hru4sgh/NAVSIM/dataset}"
TRAIN_METRIC_CACHE_PATH="${TRAIN_METRIC_CACHE_PATH:-$REPO_ROOT/dataset/train_metric_cache_navtrain}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-nav2_safety_b0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/nuplan-devkit:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DATA_ROOT/maps}"
export NAVSIM_DEVKIT_ROOT="$REPO_ROOT"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$REPO_ROOT/exp}"
export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"
export HYDRA_FULL_ERROR=1
export DRIVOR_NO_RESUME=1
export PYTHONUNBUFFERED=1

IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARRAY[@]}"
TRAINER_STRATEGY=auto
if (( NUM_GPUS > 1 )); then
  TRAINER_STRATEGY=ddp
fi

for path in "$NUPLAN_MAPS_ROOT" \
  "$OPENSCENE_DATA_ROOT/navsim_logs/trainval" \
  "$OPENSCENE_DATA_ROOT/sensor_blobs/trainval" "$TRAIN_METRIC_CACHE_PATH"
do
  if [[ ! -e "$path" ]]; then
    printf '[ERROR] Required training path not found: %s\n' "$path" >&2
    exit 1
  fi
done
if (( DATALOADER_WORKERS < 1 )); then
  printf '[ERROR] DATALOADER_WORKERS must be at least 1.\n' >&2
  exit 1
fi

COMMAND=("$PYTHON_BIN" "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py"
  agent=drivoR
  "experiment_name=$EXPERIMENT_NAME"
  train_test_split=navtrain
  cache_path=null
  use_cache_without_dataset=false
  train_ckpt_path=null
  "trainer.params.max_epochs=$MAX_EPOCHS"
  "+trainer.params.devices=$NUM_GPUS"
  "trainer.params.strategy=$TRAINER_STRATEGY"
  trainer.params.gradient_clip_val=1.0
  "dataloader.params.batch_size=$BATCH_SIZE"
  "dataloader.params.num_workers=$DATALOADER_WORKERS"
  dataloader.params.prefetch_factor=1
  +dataloader.params.persistent_workers=true
  "agent.checkpoint_path=''"
  agent.checkpoint_perception_only=false
  "agent.train_metric_cache_path=$TRAIN_METRIC_CACHE_PATH"
  "agent.perception_lr_scale=${PERCEPTION_LR_SCALE:-1.0}"
  "agent.use_ray=${USE_RAY:-true}"
  "agent.num_gpus=$NUM_GPUS"
  agent.progress_bar=false
  agent.lr_args.name=AdamW
  "agent.lr_args.base_lr=${BASE_LR:-0.0002}"
  agent.config.image_backbone.use_lora=true
  agent.config.image_backbone.finetune=false
  agent.config.image_backbone.focus_front_cam=false
  'agent.config.lidar_pc=[]'
  agent.config.refiner_ls_values=0.0
  agent.config.one_token_per_traj=true
  agent.config.refiner_num_heads=1
  agent.config.tf_d_model=256
  agent.config.tf_d_ffn=1024
  agent.config.area_pred=false
  agent.config.agent_pred=false
  agent.config.bev_map=false
  agent.config.ref_num=4
  agent.config.long_trajectory_additional_poses=-1
  agent.loss.prev_weight=0.0
  "agent.loss.dac_weight=${DAC_WEIGHT:-0.0}"
  "agent.loss.collision_weight=${COLLISION_WEIGHT:-0.0}"
  "agent.loss.dac_margin=${DAC_MARGIN:-0.3}"
  "agent.loss.collision_margin=${COLLISION_MARGIN:-0.3}"
  "agent.loss.sdf_resolution=${SDF_RESOLUTION:-0.5}"
  "seed=${SEED:-2}"
  "$@")

BACKGROUND="${BACKGROUND:-1}"
for argument in "$@"; do
  case "$argument" in
    --cfg|--cfg=*|--help|-h|--hydra-help|--info|--info=*|--version)
      BACKGROUND=0
      ;;
  esac
done
LOG_FILE="${LOG_FILE:-$NAVSIM_EXP_ROOT/logs/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log}"

printf '[INFO] Data: %s\n[INFO] GPUs: %s\n' "$DATA_ROOT" "$GPU_IDS"
printf '[INFO] B2: pretrained DINOv2 only; no DrivoR checkpoint or automatic resume.\n'
printf '[INFO] LoRA, neck, scene tokens, ego encoder, trajectory generator and scorer start fresh.\n'
printf '[INFO] Outputs: %s/ke/%s\n' "$NAVSIM_EXP_ROOT" "$EXPERIMENT_NAME"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  if [[ "$BACKGROUND" == 1 ]]; then
    printf 'nohup '
  fi
  printf '%q ' "${COMMAND[@]}"
  if [[ "$BACKGROUND" == 1 ]]; then
    printf '> %q 2>&1 < /dev/null &\n' "$LOG_FILE"
  else
    printf '\n'
  fi
  exit 0
fi
if [[ "$BACKGROUND" != 1 ]]; then
  exec "${COMMAND[@]}"
fi
mkdir -p -- "$(dirname -- "$LOG_FILE")"
nohup "${COMMAND[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
printf '[INFO] Submitted PID: %s\n[INFO] Log: %s\n' "$!" "$LOG_FILE"