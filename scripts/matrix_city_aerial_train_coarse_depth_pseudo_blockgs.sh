#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-config/mc_aerial_coarse_depth_pseudo_blockgs.yaml}"
CUDA_ID="${CUDA_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
USE_SHARED_MMAP_IMAGES="${USE_SHARED_MMAP_IMAGES:-1}"
REBUILD_IMAGE_MMAP_CACHE="${REBUILD_IMAGE_MMAP_CACHE:-0}"

export CONFIG
export CUDA_ID
export PYTHON_BIN
export USE_SHARED_MMAP_IMAGES
export REBUILD_IMAGE_MMAP_CACHE

bash scripts/matrix_city_aerial_train_coarse_depth_8w_densify32k.sh
