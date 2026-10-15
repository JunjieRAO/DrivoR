#!/usr/bin/env bash
set -euo pipefail
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/nav2_scene_geometry_pipeline.sh" cache "$@"
