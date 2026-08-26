#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/workspace/roa7sgh/DrivoR}"
DATA_ROOT="${DATA_ROOT:-/mnt/workspace/hru4sgh/NAVSIM/dataset}"
IMAGE_BACKBONE_CHECKPOINT="${IMAGE_BACKBONE_CHECKPOINT:-/mnt/workspace/roa7sgh/DrivoR/weights/drivor_Nav1_25epochs.pth}"
LIDAR_BACKBONE_CHECKPOINT="${LIDAR_BACKBONE_CHECKPOINT:-}"
TRAIN_METRIC_CACHE_PATH="${TRAIN_METRIC_CACHE_PATH:-/mnt/workspace/hru4sgh/NAVSIM/dataset/matric_cache/metric_cache_navtrain}"
GPU_IDS="${GPU_IDS:-0}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-16}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-25}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-drivoR_nav1_frozen_backbones}"

if [[ ! -f "$IMAGE_BACKBONE_CHECKPOINT" ]]; then
  echo "[ERROR] Image backbone checkpoint not found: $IMAGE_BACKBONE_CHECKPOINT" >&2
  exit 1
fi
if [[ -n "$LIDAR_BACKBONE_CHECKPOINT" && ! -f "$LIDAR_BACKBONE_CHECKPOINT" ]]; then
  echo "[ERROR] Lidar backbone checkpoint not found: $LIDAR_BACKBONE_CHECKPOINT" >&2
  exit 1
fi

IMAGE_BACKBONE_CHECKPOINT="$IMAGE_BACKBONE_CHECKPOINT" \
LIDAR_BACKBONE_CHECKPOINT="$LIDAR_BACKBONE_CHECKPOINT" \
python3 - <<'PY'
import os
import torch

branches = [("image_backbone", "scene_embeds", "IMAGE_BACKBONE_CHECKPOINT")]
if os.environ["LIDAR_BACKBONE_CHECKPOINT"]:
  branches.append(("lidar_backbone", "lidar_scene_embeds", "LIDAR_BACKBONE_CHECKPOINT"))

for branch, scene_embed, variable in branches:
  path = os.environ[variable]
  checkpoint = torch.load(path, map_location="cpu")
  state_dict = checkpoint.get("state_dict", checkpoint)
  count = sum(f"{branch}." in key for key in state_dict)
  if count == 0:
    raise RuntimeError(f"{path} contains no {branch} weights")
  embed_count = sum(key.split(".")[-1] == scene_embed for key in state_dict)
  if embed_count != 1:
    raise RuntimeError(f"{path} must contain exactly one {scene_embed}; found {embed_count}")
  print(f"[OK] Found {count} {branch} tensors and {scene_embed} in {path}")
PY

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
export DRIVOR_NO_RESUME="${DRIVOR_NO_RESUME:-1}"

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

echo "[INFO] GPUs: $GPU_IDS ($NUM_GPUS device(s))"
echo "[INFO] DataLoader workers: $DATALOADER_WORKERS"
echo "[INFO] Training metric cache: $TRAIN_METRIC_CACHE_PATH"
echo "[INFO] Training outputs: $NAVSIM_EXP_ROOT"
echo "[INFO] Image backbone and scene_embeds will be loaded and frozen."
if [[ -n "$LIDAR_BACKBONE_CHECKPOINT" ]]; then
  echo "[INFO] Lidar backbone and lidar_scene_embeds will be loaded and frozen."
else
  echo "[INFO] Image-only initialization; lidar branch is disabled."
fi
echo "[INFO] Trajectory generation and scorer modules will train from scratch."

LIDAR_SENSOR_CONFIG='[]'
FREEZE_LIDAR=false
if [[ -n "$LIDAR_BACKBONE_CHECKPOINT" ]]; then
  LIDAR_SENSOR_CONFIG='[3]'
  FREEZE_LIDAR=true
fi

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
  agent.image_backbone_checkpoint_path="$IMAGE_BACKBONE_CHECKPOINT" \
  agent.lidar_backbone_checkpoint_path="$LIDAR_BACKBONE_CHECKPOINT" \
  agent.freeze_image_backbone=true \
  agent.freeze_lidar_backbone="$FREEZE_LIDAR" \
  agent.train_metric_cache_path="$TRAIN_METRIC_CACHE_PATH" \
  agent.num_gpus="$NUM_GPUS" \
  agent.progress_bar=false \
  agent.lr_args.name=AdamW \
  agent.lr_args.base_lr="${BASE_LR:-0.0002}" \
  agent.config.lidar_pc="$LIDAR_SENSOR_CONFIG" \
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
  seed="${SEED:-2}"

echo "[DONE] Training outputs: $NAVSIM_EXP_ROOT/ke/$EXPERIMENT_NAME"