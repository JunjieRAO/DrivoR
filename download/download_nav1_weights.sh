#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_PATH="${1:-$REPO_ROOT/weights/drivor_Nav1_25epochs.pth}"
WEIGHT_URL="https://github.com/valeoai/DrivoR/releases/download/model_weights/drivor_Nav1_25epochs.pth"
EXPECTED_SHA256="e1a678f201e4f1ab93d117caad42782cd7ead293bdced2b5f80212bc92426ae3"

mkdir -p "$(dirname "$OUTPUT_PATH")"

if [[ -f "$OUTPUT_PATH" ]] && echo "$EXPECTED_SHA256  $OUTPUT_PATH" | sha256sum --check --status; then
  echo "[OK] NAVSIM-v1 checkpoint already exists and checksum matches: $OUTPUT_PATH"
  exit 0
fi

echo "[INFO] Downloading official NAVSIM-v1 checkpoint to $OUTPUT_PATH"
if command -v curl >/dev/null 2>&1; then
  curl --fail --location --retry 3 --continue-at - --output "$OUTPUT_PATH" "$WEIGHT_URL"
elif command -v wget >/dev/null 2>&1; then
  wget --continue --output-document="$OUTPUT_PATH" "$WEIGHT_URL"
else
  echo "[ERROR] curl or wget is required." >&2
  exit 1
fi

echo "$EXPECTED_SHA256  $OUTPUT_PATH" | sha256sum --check
echo "[DONE] Checkpoint ready: $OUTPUT_PATH"