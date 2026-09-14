#!/usr/bin/env bash
# Server defaults taken from scripts/nav1_train_frozen_backbones.sh and nav1_eval.sh.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/mnt/workspace/roa7sgh/DrivoR}"
DATA_ROOT="${DATA_ROOT:-/mnt/workspace/hru4sgh/NAVSIM/dataset}"
WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/exp/offline_search/nav1_engineering}"
# Reuse the original manifests/cache; choose a fresh result destination on reruns.
RESULT_ROOT="${RESULT_ROOT:-$WORK_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/nuplan-devkit${PYTHONPATH:+:$PYTHONPATH}"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$DATA_ROOT/maps}"
export NAVSIM_DEVKIT_ROOT="$REPO_ROOT"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NAVSIM_EXP_ROOT="$REPO_ROOT/exp"
export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"
export HYDRA_FULL_ERROR=1
cd "$REPO_ROOT"
case "${1:-help}" in
  verify)
    "$PYTHON_BIN" -m pytest tests/offline_search -q
    ;;
  prepare)
    "$PYTHON_BIN" -m navsim.offline_search.prepare_nav1 \
      --repo "$REPO_ROOT" --scene-root "$DATA_ROOT/navsim_logs/trainval" \
      --output "$WORK_ROOT" --count "${SCENE_COUNT:-16}" --seed "${SEARCH_SEED:-20260914}"
    ;;
  smoke|full)
    mode="$1"
    if [[ "$mode" == smoke ]]; then
      manifest="$WORK_ROOT/one_scene_manifest.jsonl"
      population=4
      generations=1
    else
      manifest="$WORK_ROOT/engineering_manifest.jsonl"
      population=32
      generations=5
    fi
    "$PYTHON_BIN" -m navsim.offline_search.run run \
      --manifest "$manifest" --map-root "$NUPLAN_MAPS_ROOT" \
      --population "$population" --generations "$generations" \
      --workers "${SEARCH_WORKERS:-1}" --worker-threads "${WORKER_THREADS:-1}" \
      --seed "${SEARCH_SEED:-20260914}" --output "$RESULT_ROOT/${mode}_results"
    ;;
  *)
    printf '%s\n' 'Usage: bash scripts/offline_search/nav1_engineering.sh {verify|prepare|smoke|full}'
    exit 1
    ;;
esac
