#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-config/mc_aerial_coarse_normal_w10x_start20k.yaml}"
CUDA_ID="${CUDA_ID:-1}"
RENDER_CUDA_ID="${RENDER_CUDA_ID:-$CUDA_ID}"
METRICS_CUDA_ID="${METRICS_CUDA_ID:-$RENDER_CUDA_ID}"
PYTHON_BIN="${PYTHON_BIN:-/lfs1/users/spsong/anaconda3/envs/BlockGSNorm/bin/python}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_RENDER="${RUN_RENDER:-1}"
RUN_METRICS="${RUN_METRICS:-1}"
IMAGE_MMAP_CACHE_DIR="${IMAGE_MMAP_CACHE_DIR:-/lfs3/users/spsong/dataset/MatrixCity/small_city/aerial/train/block_all/.cache/images_input_r-1_normal}"

export CONFIG CUDA_ID RENDER_CUDA_ID METRICS_CUDA_ID PYTHON_BIN
export RUN_TRAIN RUN_RENDER RUN_METRICS IMAGE_MMAP_CACHE_DIR

bash scripts/matrix_city_aerial_coarse_normal.sh
