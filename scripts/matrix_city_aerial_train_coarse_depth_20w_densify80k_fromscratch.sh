#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-config/mc_aerial_coarse_depth_20w_densify80k_fromscratch.yaml}"
CUDA_ID="${CUDA_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_RENDER_TEST="${RUN_RENDER_TEST:-0}"
RUN_METRICS="${RUN_METRICS:-0}"
SKIP_TRAINED="${SKIP_TRAINED:-1}"
SKIP_RENDERED="${SKIP_RENDERED:-1}"
USE_SHARED_MMAP_IMAGES="${USE_SHARED_MMAP_IMAGES:-1}"
REBUILD_IMAGE_MMAP_CACHE="${REBUILD_IMAGE_MMAP_CACHE:-0}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_RENDER_ARGS="${EXTRA_RENDER_ARGS:-}"
EXTRA_METRICS_ARGS="${EXTRA_METRICS_ARGS:-}"

if [ ! -f "$CONFIG" ]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

eval "$(
    "$PYTHON_BIN" - "$CONFIG" <<'PY'
import os
import shlex
import sys

from utils.config_utils import load_yaml_config, stage_args_from_config

cfg = load_yaml_config(sys.argv[1])
train_args = stage_args_from_config(cfg, "train")

source_path = train_args["source_path"]
images = train_args.get("images", "images")
depths = train_args.get("depths", "")
resolution = train_args.get("resolution", -1)
safe_images = str(images).replace(os.sep, "_").replace("/", "_")
default_cache = os.path.join(source_path, ".cache", f"images_{safe_images}_r{resolution}")

def emit(name, value):
    print(f"{name}={shlex.quote(str(value))}")

emit("CFG_SOURCE_PATH", source_path)
emit("CFG_IMAGES", images)
emit("CFG_DEPTHS", depths)
emit("CFG_RESOLUTION", resolution)
emit("CFG_IMAGE_MMAP_CACHE_DIR", train_args.get("image_mmap_cache_dir", "") or default_cache)
PY
)"

IMAGE_MMAP_CACHE_DIR="${IMAGE_MMAP_CACHE_DIR:-$CFG_IMAGE_MMAP_CACHE_DIR}"

if [ "$USE_SHARED_MMAP_IMAGES" = "1" ]; then
    depth_cache_missing=0
    if [ -n "$CFG_DEPTHS" ] && [ ! -f "$IMAGE_MMAP_CACHE_DIR/depths.float32.bin" ]; then
        depth_cache_missing=1
    fi

    if [ "$REBUILD_IMAGE_MMAP_CACHE" = "1" ] || [ ! -f "$IMAGE_MMAP_CACHE_DIR/manifest.json" ] || [ ! -f "$IMAGE_MMAP_CACHE_DIR/images.uint8.bin" ] || [ "$depth_cache_missing" = "1" ]; then
        echo "Building shared image/depth mmap cache: $IMAGE_MMAP_CACHE_DIR"
        "$PYTHON_BIN" tools/build_image_mmap_cache.py \
            -s "$CFG_SOURCE_PATH" \
            --images "$CFG_IMAGES" \
            --depths "$CFG_DEPTHS" \
            -r "$CFG_RESOLUTION" \
            -o "$IMAGE_MMAP_CACHE_DIR"
    else
        echo "Using existing shared image/depth mmap cache: $IMAGE_MMAP_CACHE_DIR"
    fi

    shared_mmap_args=(
        --override "dataset.image_load_mode=shared_mmap"
        --override "dataset.image_mmap_cache_dir=$IMAGE_MMAP_CACHE_DIR"
    )
    EXTRA_TRAIN_ARGS="${shared_mmap_args[*]} ${EXTRA_TRAIN_ARGS:-}"
fi

export CONFIG
export CUDA_ID
export PYTHON_BIN
export RUN_TRAIN
export RUN_RENDER_TEST
export RUN_METRICS
export SKIP_TRAINED
export SKIP_RENDERED
export EXTRA_TRAIN_ARGS
export EXTRA_RENDER_ARGS
export EXTRA_METRICS_ARGS

bash scripts/matrix_city_aerial_train_coarse_depth.sh
