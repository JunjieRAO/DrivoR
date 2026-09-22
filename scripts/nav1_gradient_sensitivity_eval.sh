#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/workspace/roa7sgh/DrivoR}"
DATA_ROOT="${DATA_ROOT:-/mnt/workspace/hru4sgh/NAVSIM/dataset}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$REPO_ROOT/weights/drivor_Nav1_25epochs.pth}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-$DATA_ROOT/metric_cache/metric_cache_navtest}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-16}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"
SCORING_WORKERS="${SCORING_WORKERS:-60}"
SCENE_COUNT="${SCENE_COUNT:-2000}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
DISPLACEMENTS="${DISPLACEMENTS:-[0.1,0.3,0.5,0.7]}"
SMOOTH_SIGMA="${SMOOTH_SIGMA:-1.0}"
SMOOTH_KERNEL_SIZE="${SMOOTH_KERNEL_SIZE:-5}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-nav1_gradient_sensitivity}"

for path in \
  "$CHECKPOINT_PATH" \
  "$DATA_ROOT/maps" \
  "$DATA_ROOT/navsim_logs/test" \
  "$DATA_ROOT/sensor_blobs/test" \
  "$METRIC_CACHE_PATH"
do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Required path not found: $path" >&2
    exit 1
  fi
done

IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARRAY[@]}"
if (( NUM_GPUS > 1 )); then
  TRAINER_STRATEGY=ddp
else
  TRAINER_STRATEGY=auto
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"
export NAVSIM_DEVKIT_ROOT="$REPO_ROOT"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NAVSIM_EXP_ROOT="$REPO_ROOT/exp"
export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"
export HYDRA_FULL_ERROR=1

echo "[INFO] Checkpoint: $CHECKPOINT_PATH"
echo "[INFO] GPUs: $GPU_IDS ($NUM_GPUS device(s))"
echo "[INFO] Batch size: $GPU_BATCH_SIZE per GPU, $((GPU_BATCH_SIZE * NUM_GPUS)) global"
echo "[INFO] Scenes: $SCENE_COUNT; sample seed: $SAMPLE_SEED"
echo "[INFO] XY displacements: $DISPLACEMENTS m"
echo "[INFO] Smooth gradient: Gaussian sigma=$SMOOTH_SIGMA, kernel=$SMOOTH_KERNEL_SIZE"
echo "[INFO] Each displacement evaluates raw/smooth x up/down/random for NC/DAC/TTC."

python3 "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_multi_gpu.py" \
  worker.threads_per_node="$SCORING_WORKERS" \
  train_test_split=navtest \
  agent=drivoR \
  agent.checkpoint_path="$CHECKPOINT_PATH" \
  experiment_name="$EXPERIMENT_NAME" \
  evaluate_all_proposals=true \
  gradient_sensitivity.enabled=true \
  gradient_sensitivity.scene_count="$SCENE_COUNT" \
  gradient_sensitivity.sample_seed="$SAMPLE_SEED" \
  gradient_sensitivity.displacements="$DISPLACEMENTS" \
  gradient_sensitivity.smooth_sigma="$SMOOTH_SIGMA" \
  gradient_sensitivity.smooth_kernel_size="$SMOOTH_KERNEL_SIZE" \
  +trainer.params.devices="$NUM_GPUS" \
  trainer.params.strategy="$TRAINER_STRATEGY" \
  +trainer.params.inference_mode=false \
  dataloader.params.batch_size="$GPU_BATCH_SIZE" \
  dataloader.params.num_workers="$DATALOADER_WORKERS" \
  metric_cache_path="$METRIC_CACHE_PATH" \
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

echo "[DONE] Results: $NAVSIM_EXP_ROOT/ke/$EXPERIMENT_NAME"