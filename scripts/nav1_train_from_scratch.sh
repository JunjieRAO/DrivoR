#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/workspace/roa7sgh/DrivoR}"
DATA_ROOT="${DATA_ROOT:-/mnt/workspace/hru4sgh/NAVSIM/dataset}"
TRAIN_METRIC_CACHE_PATH="${TRAIN_METRIC_CACHE_PATH:-$DATA_ROOT/train_metric_cache_navtrain}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-16}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-25}"
BASE_LR="${BASE_LR:-0.0002}"
SEED="${SEED:-2}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-nav1_from_scratch}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DATA_ROOT/maps}"
export NAVSIM_DEVKIT_ROOT="$REPO_ROOT"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NAVSIM_EXP_ROOT="$REPO_ROOT/exp"
export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"
export HYDRA_FULL_ERROR=1
export DRIVOR_NO_RESUME=1

IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARRAY[@]}"
if (( NUM_GPUS > 1 )); then
  TRAINER_STRATEGY=ddp
else
  TRAINER_STRATEGY=auto
fi

for path in \
  "$NUPLAN_MAPS_ROOT" \
  "$OPENSCENE_DATA_ROOT/navsim_logs/trainval" \
  "$OPENSCENE_DATA_ROOT/sensor_blobs/trainval" \
  "$TRAIN_METRIC_CACHE_PATH"
do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Required training data/cache not found: $path" >&2
    exit 1
  fi
done

echo "[INFO] Training DrivoR from scratch without loading a DrivoR checkpoint."
echo "[INFO] Image encoder uses DINOv2 initialization with trainable LoRA adapters, as in DrivoR."
echo "[INFO] Scene embeddings, trajectory generator, scorer, neck, and LoRA adapters are trainable."
echo "[INFO] GPUs: $GPU_IDS ($NUM_GPUS device(s))"
echo "[INFO] Per-GPU batch size: $BATCH_SIZE"
echo "[INFO] DataLoader workers: $DATALOADER_WORKERS"
echo "[INFO] Base learning rate: $BASE_LR"
echo "[INFO] Training metric cache: $TRAIN_METRIC_CACHE_PATH"
echo "[INFO] Training outputs: $NAVSIM_EXP_ROOT/ke/$EXPERIMENT_NAME"

python3 "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py" \
  agent=drivoR \
  experiment_name="$EXPERIMENT_NAME" \
  train_test_split=navtrain \
  cache_path=null \
  use_cache_without_dataset=false \
  trainer.params.max_epochs="$MAX_EPOCHS" \
  +trainer.params.devices="$NUM_GPUS" \
  trainer.params.strategy="$TRAINER_STRATEGY" \
  dataloader.params.batch_size="$BATCH_SIZE" \
  dataloader.params.num_workers="$DATALOADER_WORKERS" \
  dataloader.params.prefetch_factor=1 \
  +dataloader.params.persistent_workers=true \
  agent.checkpoint_path='' \
  agent.image_backbone_checkpoint_path='' \
  agent.lidar_backbone_checkpoint_path='' \
  agent.freeze_image_backbone=false \
  agent.freeze_lidar_backbone=false \
  agent.train_metric_cache_path="$TRAIN_METRIC_CACHE_PATH" \
  agent.num_gpus="$NUM_GPUS" \
  agent.progress_bar=false \
  agent.lr_args.name=AdamW \
  agent.lr_args.base_lr="$BASE_LR" \
  agent.config.lidar_pc='[]' \
  agent.config.image_backbone.use_lora=true \
  agent.config.image_backbone.finetune=false \
  agent.config.refiner_ls_values=0.0 \
  agent.config.image_backbone.focus_front_cam=false \
  agent.config.one_token_per_traj=true \
  agent.config.refiner_num_heads=1 \
  agent.config.tf_d_model=256 \
  agent.config.tf_d_ffn=1024 \
  agent.config.area_pred=false \
  agent.config.agent_pred=false \
  agent.config.ref_num=4 \
  agent.loss.prev_weight=0.0 \
  agent.config.long_trajectory_additional_poses=2 \
  seed="$SEED"

echo "[DONE] Training outputs: $NAVSIM_EXP_ROOT/ke/$EXPERIMENT_NAME"
