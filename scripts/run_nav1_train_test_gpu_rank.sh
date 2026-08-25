#!/usr/bin/env bash
set -euo pipefail

# One-click NAVSIM-v1 training/evaluation with explicit GPU selection.
#
# Examples:
#   bash scripts/run_nav1_train_test_gpu_rank.sh --gpu-rank 0 --mode train
#   bash scripts/run_nav1_train_test_gpu_rank.sh --gpu-rank 0,1,2,3 --mode train_eval
#   bash scripts/run_nav1_train_test_gpu_rank.sh --gpu-rank 0 --mode eval --checkpoint /path/to/model.ckpt
#
# Options:
#   --gpu-rank <list>          GPU indices to use, comma-separated (required), e.g. 0 or 0,1,2,3
#   --mode <train|eval|train_eval>   Run mode (default: train_eval)
#   --checkpoint <path>        Checkpoint for eval mode; optional in train_eval (auto-detect latest)
#   --experiment <name>        Experiment name (default: training_drivoR_Nav1_traj_long_25epochs)
#
# Optional env vars:
#   SKIP_TRAIN_CACHE=1         Skip train metric caching
#   SKIP_EVAL_CACHE=1          Skip eval metric caching
#   TRAIN_EPOCHS=25            Override max epochs

GPU_RANK=""
MODE="train_eval"
CHECKPOINT_PATH=""
EXPERIMENT="training_drivoR_Nav1_traj_long_25epochs"
AGENT="drivoR"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-rank)
      GPU_RANK="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --experiment)
      EXPERIMENT="$2"
      shift 2
      ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$GPU_RANK" ]]; then
  echo "[ERROR] --gpu-rank is required, e.g. --gpu-rank 0 or --gpu-rank 0,1,2,3"
  exit 1
fi

if [[ "$MODE" != "train" && "$MODE" != "eval" && "$MODE" != "train_eval" ]]; then
  echo "[ERROR] --mode must be one of: train, eval, train_eval"
  exit 1
fi

source /home/roa7sgh/DrivoR/scripts/setup_drivor_env.sh

export CUDA_VISIBLE_DEVICES="$GPU_RANK"
NUM_GPUS=$(awk -F',' '{print NF}' <<< "$GPU_RANK")
export HYDRA_FULL_ERROR=1
export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-25}"

echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] NUM_GPUS=$NUM_GPUS"
echo "[INFO] MODE=$MODE"

run_train() {
  # if [[ "${SKIP_TRAIN_CACHE:-0}" != "1" ]]; then
  #   echo "[INFO] Running train metric caching..."
  #   python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_train_metric_caching.py"
  # else
  #   echo "[INFO] Skipping train metric caching (SKIP_TRAIN_CACHE=1)."
  # fi

  echo "[INFO] Starting NAVSIM-v1 training..."
  python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py" \
    agent=$AGENT \
    experiment_name=$EXPERIMENT \
    train_test_split=navtrain \
    cache_path=null \
    use_cache_without_dataset=false \
    trainer.params.max_epochs=$TRAIN_EPOCHS \
    dataloader.params.prefetch_factor=1 \
    dataloader.params.batch_size=4 \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=0.0002 \
    agent.num_gpus=$NUM_GPUS \
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
}

find_latest_checkpoint() {
  local search_root="$NAVSIM_EXP_ROOT/$EXPERIMENT"
  local latest
  latest=$(find "$search_root" -type f \( -name '*.ckpt' -o -name '*.pth' \) -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | awk '{print $2}')
  echo "$latest"
}

run_eval() {
  local ckpt="$CHECKPOINT_PATH"

  if [[ -z "$ckpt" ]]; then
    ckpt=$(find_latest_checkpoint)
    if [[ -z "$ckpt" ]]; then
      echo "[ERROR] No checkpoint found under $NAVSIM_EXP_ROOT/$EXPERIMENT"
      echo "        Please pass --checkpoint /path/to/model.ckpt"
      exit 1
    fi
    echo "[INFO] Auto-selected checkpoint: $ckpt"
  fi

  if [[ ! -f "$ckpt" ]]; then
    echo "[ERROR] Checkpoint not found: $ckpt"
    exit 1
  fi

  if [[ "${SKIP_EVAL_CACHE:-0}" != "1" ]]; then
    echo "[INFO] Running eval metric caching..."
    bash /home/roa7sgh/DrivoR/scripts/evaluation/run_metric_caching.sh
  else
    echo "[INFO] Skipping eval metric caching (SKIP_EVAL_CACHE=1)."
  fi

  echo "[INFO] Starting NAVSIM-v1 navtest evaluation..."
  python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_multi_gpu.py" \
    train_test_split=navtest \
    agent=$AGENT \
    agent.checkpoint_path="$ckpt" \
    experiment_name="${EXPERIMENT}_eval" \
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
}

case "$MODE" in
  train)
    run_train
    ;;
  eval)
    run_eval
    ;;
  train_eval)
    run_train
    run_eval
    ;;
esac

echo "[DONE] Mode '$MODE' completed."

# Ensure terminal stays open so user can view logs
read -p "[INFO] Press Enter to close terminal..." || true
