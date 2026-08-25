#!/usr/bin/env bash
set -euo pipefail

source /home/roa7sgh/DrivoR/scripts/setup_drivor_env.sh

export HYDRA_FULL_ERROR=1
EXPERIMENT=training_drivoR_Nav1_traj_long_25epochs
AGENT=drivoR

# python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_train_metric_caching.py"

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py" \
  agent=$AGENT \
  experiment_name=$EXPERIMENT \
  train_test_split=navtrain \
  cache_path=null \
  use_cache_without_dataset=false \
  trainer.params.max_epochs=25 \
  dataloader.params.prefetch_factor=2 \
  dataloader.params.batch_size=${BATCH_SIZE:-16} \
  dataloader.params.num_workers=${DRIVOR_NUM_WORKERS:-16} \
  +dataloader.params.persistent_workers=true \
  dataloader.params.pin_memory=true \
  agent.lr_args.name=AdamW \
  agent.lr_args.base_lr=0.0002 \
  agent.num_gpus=${NUM_GPUS:-4} \
  agent.progress_bar=false \
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
  seed=2

echo "[DONE] Training completed. Check outputs under: $NAVSIM_EXP_ROOT"

# Ensure terminal stays open so user can view logs
read -p "[INFO] Press Enter to close terminal..." || true
