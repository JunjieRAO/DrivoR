#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_nav1_eval_oneclick.sh [path/to/checkpoint.pth]
# Optional env:
#   SKIP_METRIC_CACHE=1    # skip metric caching step
#   EXPERIMENT_NAME=name   # override experiment name (default: drivoR_nav1_eval)
#   EVAL_DEVICES=1         # number of GPUs visible to Lightning (default: 1)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKPOINT_PATH="${1:-$REPO_ROOT/weights/drivor_Nav1_25epochs.pth}"
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "[ERROR] Checkpoint not found: $CHECKPOINT_PATH"
  echo "        Download it with: bash $REPO_ROOT/download/download_nav1_weights.sh"
  exit 1
fi

source "$SCRIPT_DIR/setup_drivor_env.sh"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-drivoR_nav1_eval}"
AGENT=drivoR

if [[ "${SKIP_METRIC_CACHE:-0}" != "1" ]]; then
  echo "[INFO] Running evaluation metric caching..."
  bash /home/roa7sgh/DrivoR/scripts/evaluation/run_metric_caching.sh
else
  echo "[INFO] Skipping evaluation metric caching (SKIP_METRIC_CACHE=1)."
fi

export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"

echo "[INFO] Running NAVSIM-v1 evaluation on navtest..."
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_multi_gpu.py" \
  train_test_split=navtest \
  agent=$AGENT \
  agent.checkpoint_path="$CHECKPOINT_PATH" \
  experiment_name=$EXPERIMENT_NAME \
  evaluate_all_proposals=true \
  trainer.params.devices="${EVAL_DEVICES:-1}" \
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

echo "[DONE] Evaluation finished. Selected and best-of-all-proposal scores are under:"
echo "       $NAVSIM_EXP_ROOT/ke/$EXPERIMENT_NAME"
