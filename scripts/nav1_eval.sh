#!/usr/bin/env bash

cd /mnt/workspace/roa7sgh/DrivoR
export PYTHONPATH=/mnt/workspace/roa7sgh/DrivoR:$PYTHONPATH
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/mnt/workspace/hru4sgh/NAVSIM/dataset/maps
export NAVSIM_DEVKIT_ROOT=/mnt/workspace/roa7sgh/DrivoR
export OPENSCENE_DATA_ROOT=/mnt/workspace/hru4sgh/NAVSIM/dataset
export NAVSIM_EXP_ROOT=/mnt/workspace/roa7sgh/DrivoR/exp
export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"

python3 "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_multi_gpu.py" \
  worker.threads_per_node=16 \
  train_test_split=navtest \
  agent=drivoR \
  agent.checkpoint_path="/mnt/workspace/roa7sgh/DrivoR/exp/ke/nav1_frozen_backbones_rank_weight/09.02_09.38/lightning_logs/version_0/checkpoints/best-epoch20-step33873.ckpt" \
  experiment_name=nav1_frozen_backbones_rank_weight_eval \
  evaluate_all_proposals=true \
  +trainer.params.devices=1 \
  trainer.params.strategy=auto \
  metric_cache_path="/mnt/workspace/hru4sgh/NAVSIM/dataset/matric_cache/metric_cache_navtest" \
  agent.config.proposal_num=64 \
  agent.config.refiner_ls_values=0.0 \
  agent.config.image_backbone.focus_front_cam=false \
  agent.config.one_token_per_traj=true \
  agent.config.refiner_num_heads=1 \
  agent.config.tf_d_model=256 \
  agent.config.tf_d_ffn=1024 \
  agent.config.area_pred=false \
  agent.config.agent_pred=false \
  agent.config.ref_num=4 \
  agent.config.noc=1 \
  agent.config.dac=1 \
  agent.config.ddc=0.0 \
  agent.config.ttc=5 \
  agent.config.ep=5 \
  agent.config.comfort=2