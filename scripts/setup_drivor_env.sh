#!/usr/bin/env bash

# Detect whether this script is being sourced. When sourced, strict-mode
# options (set -e/-u/pipefail) would otherwise leak into the caller's
# interactive shell, so that pressing Ctrl+C (exit code 130) triggers
# 'set -e' and closes the terminal. We save the caller's options here and
# restore them at the end.
if (return 0 2>/dev/null); then
  _DRIVOR_SOURCED=1
  _DRIVOR_OLD_SETOPTS="$(set +o)"
else
  _DRIVOR_SOURCED=0
fi

set -euo pipefail

# Ensure the module command is available in non-login shells.
if ! command -v module >/dev/null 2>&1; then
  [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
fi

# Load cluster modules (ignore optional module failures).
# NOTE: NVIDIA B200 GPUs (Blackwell, sm_100) require CUDA >= 12.6 and a matching
# PyTorch build (torch 2.1.0+cu121 has no compiled kernels for sm_100 and fails
# with "CUDA failure 'named symbol not found'").
module purge || true
module load cuda/12.6.0
module load cudnn/12.6_v9.4
module load gcc/11.4.0
module load nccl/2.23.4_cuda12.6 || true

# Activate conda environment.
# On this cluster, conda may only be available after loading a module.
if ! command -v conda >/dev/null 2>&1; then
  module load conda/4.9.2 >/dev/null 2>&1 || module load conda >/dev/null 2>&1 || true
fi

# Conda's shell hook and `conda activate` reference variables such as $PS1
# that are unbound in a non-interactive shell. Under `set -u` (nounset) that
# aborts the script, so relax nounset while initialising/activating conda and
# restore it afterwards.
set +u
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] conda not found. Please install conda or source conda.sh first."
  return 1 2>/dev/null || exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "drivoR"; then
  echo "[ERROR] Conda env 'drivoR' does not exist."
  echo "        Create it once with:"
  echo "        conda create -n drivoR python=3.9 -y"
  echo "        conda activate drivoR"
  echo "        pip install --upgrade pip"
  echo "        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121"
  echo "        pip install ninja"
  echo "        pip install -e ./nuplan-devkit"
  echo "        pip install -e ."
  return 1 2>/dev/null || exit 1
fi

conda activate drivoR
set -u

# Project environment variables.
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/datasets/public/navsim/dataset/maps
export OPENSCENE_DATA_ROOT=/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/datasets/public/navsim/dataset
export NAVSIM_DEVKIT_ROOT=/home/roa7sgh/DrivoR
export NAVSIM_EXP_ROOT=/home/roa7sgh/DrivoR/exp
export SUBSCORE_PATH=$NAVSIM_EXP_ROOT
mkdir -p "$NAVSIM_EXP_ROOT"

# NAVSIM v1 data readiness check.
missing=0
for p in \
  "$NUPLAN_MAPS_ROOT" \
  "$OPENSCENE_DATA_ROOT/navsim_logs/trainval" \
  "$OPENSCENE_DATA_ROOT/sensor_blobs/trainval" \
  "$OPENSCENE_DATA_ROOT/navsim_logs/test" \
  "$OPENSCENE_DATA_ROOT/sensor_blobs/test"
do
  if [ ! -d "$p" ]; then
    echo "[MISSING] $p"
    missing=1
  fi
done

if [ "$missing" -eq 1 ]; then
  echo "[WARN] NAVSIM v1 data is incomplete. Download missing parts before training."
else
  echo "[OK] NAVSIM v1 data is ready."
fi

# CUDA / PyTorch sanity check.
python - <<'PY'
import torch
print('[Torch]', torch.__version__)
print('[CUDA]', torch.version.cuda)
print('[CUDA available]', torch.cuda.is_available())
print('[GPU count]', torch.cuda.device_count())
PY

echo "[DONE] DrivoR environment loaded."

# Restore the caller's original shell options if this script was sourced,
# so that strict mode (set -e/-u/pipefail) does not leak into the
# interactive shell and Ctrl+C no longer closes the terminal.
if [[ "${_DRIVOR_SOURCED:-0}" -eq 1 ]]; then
  eval "$_DRIVOR_OLD_SETOPTS"
  unset _DRIVOR_SOURCED _DRIVOR_OLD_SETOPTS
fi
