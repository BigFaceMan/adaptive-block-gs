#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CUDA_ID="${CUDA_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
USE_SHARED_MMAP_IMAGES="${USE_SHARED_MMAP_IMAGES:-1}"
REBUILD_IMAGE_MMAP_CACHE="${REBUILD_IMAGE_MMAP_CACHE:-0}"
CONFIGS="${CONFIGS:-config/mc_aerial_coarse_depth_pseudo_blockgs.yaml config/mc_aerial_coarse_depth_pseudo_maskdc_3w.yaml config/mc_aerial_coarse_depth_pseudo_late_3w.yaml config/mc_aerial_coarse_depth_pseudo_stable_3w.yaml config/mc_aerial_coarse_depth_pseudo_cheap_3w.yaml}"

export CUDA_ID
export PYTHON_BIN
export USE_SHARED_MMAP_IMAGES
export REBUILD_IMAGE_MMAP_CACHE

for config in $CONFIGS; do
    if [ ! -f "$config" ]; then
        echo "Config not found: $config" >&2
        exit 1
    fi
    echo "Running pseudo coarse experiment: $config"
    CONFIG="$config" bash scripts/matrix_city_aerial_train_coarse_depth_pseudo_blockgs.sh
done
