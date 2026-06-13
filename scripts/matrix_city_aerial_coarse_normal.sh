#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-config/mc_aerial_coarse_normal.yaml}"
CUDA_ID="${CUDA_ID:-9}"
RENDER_CUDA_ID="${RENDER_CUDA_ID:-$CUDA_ID}"
METRICS_CUDA_ID="${METRICS_CUDA_ID:-$RENDER_CUDA_ID}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_RENDER="${RUN_RENDER:-${RUN_RENDER_EVAL:-1}}"
RUN_METRICS="${RUN_METRICS:-${RUN_EVAL:-1}}"
SKIP_TRAINED="${SKIP_TRAINED:-1}"
FORCE_RENDER="${FORCE_RENDER:-0}"

USE_SHARED_MMAP_IMAGES="${USE_SHARED_MMAP_IMAGES:-1}"
REBUILD_IMAGE_MMAP_CACHE="${REBUILD_IMAGE_MMAP_CACHE:-0}"
IMAGE_MMAP_CACHE_DIR="${IMAGE_MMAP_CACHE_DIR:-}"

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

from utils.config_utils import get_in, load_yaml_config

cfg = load_yaml_config(sys.argv[1])
source_path = get_in(cfg, "dataset.source_path", "")
images = get_in(cfg, "dataset.images", "images")
normals = get_in(cfg, "dataset.normals", "")
resolution = get_in(cfg, "dataset.resolution", -1)
safe_images = str(images).replace("/", "_").replace(" ", "_")
default_cache = os.path.join(source_path, ".cache", f"images_{safe_images}_r{resolution}_normal")

def emit(name, value):
    print(f"{name}={shlex.quote(str(value))}")

emit("CFG_SOURCE_PATH", source_path)
emit("CFG_IMAGES", images)
emit("CFG_NORMALS", normals)
emit("CFG_NORMAL_DIR", os.path.join(source_path, normals) if normals else "")
emit("CFG_RESOLUTION", resolution)
emit("CFG_MODEL_PATH", get_in(cfg, "training.model_path", get_in(cfg, "experiment.output_root", "")))
emit("CFG_ITERATION", get_in(cfg, "render.iteration", get_in(cfg, "optimization.iterations", 30000)))
emit("CFG_IMAGE_MMAP_CACHE_DIR", get_in(cfg, "dataset.image_mmap_cache_dir", "") or default_cache)
emit("CFG_NORMAL_WEIGHT_INIT", get_in(cfg, "optimization.normal_weight_init", 0.0))
emit("CFG_NORMAL_WEIGHT_FINAL", get_in(cfg, "optimization.normal_weight_final", 0.0))
emit("CFG_NORMAL_START_ITER", get_in(cfg, "optimization.normal_start_iter", 0))
PY
)"

MODEL_PLY="$CFG_MODEL_PATH/point_cloud/iteration_${CFG_ITERATION}/point_cloud.ply"
TEST_RENDER_DIR="$CFG_MODEL_PATH/test/ours_${CFG_ITERATION}/renders"
TRAIN_RENDER_SET="$CFG_MODEL_PATH/train/ours_${CFG_ITERATION}"
IMAGE_MMAP_CACHE_DIR="${IMAGE_MMAP_CACHE_DIR:-$CFG_IMAGE_MMAP_CACHE_DIR}"

if [ "$RUN_TRAIN" = "1" ]; then
    if [ -z "$CFG_NORMALS" ]; then
        echo "Normal experiment requires dataset.normals in $CONFIG" >&2
        exit 1
    fi
    if [ ! -d "$CFG_NORMAL_DIR" ]; then
        echo "Normal directory not found: $CFG_NORMAL_DIR" >&2
        echo "Generate it with scripts/prepare_normal_regularization.sh before training." >&2
        exit 1
    fi
fi

echo "Running MatrixCity aerial coarse normal experiment"
echo "Config: $CONFIG"
echo "Model path: $CFG_MODEL_PATH"
echo "Iteration: $CFG_ITERATION"
echo "Normals: $CFG_NORMALS"
echo "Normal weights: $CFG_NORMAL_WEIGHT_INIT -> $CFG_NORMAL_WEIGHT_FINAL"
echo "Normal start iter: $CFG_NORMAL_START_ITER"
echo "CUDA_ID: $CUDA_ID"
echo "Render CUDA: $RENDER_CUDA_ID"
echo "Metrics CUDA: $METRICS_CUDA_ID"

train_overrides=()
if [ "$USE_SHARED_MMAP_IMAGES" = "1" ] && [ "$RUN_TRAIN" = "1" ]; then
    if [ "$REBUILD_IMAGE_MMAP_CACHE" = "1" ] || [ ! -f "$IMAGE_MMAP_CACHE_DIR/manifest.json" ] || [ ! -f "$IMAGE_MMAP_CACHE_DIR/normals.float32.bin" ]; then
        echo "Building shared image/normal mmap cache: $IMAGE_MMAP_CACHE_DIR"
        "$PYTHON_BIN" tools/build_image_mmap_cache.py \
            -s "$CFG_SOURCE_PATH" \
            --images "$CFG_IMAGES" \
            --normals "$CFG_NORMALS" \
            -r "$CFG_RESOLUTION" \
            -o "$IMAGE_MMAP_CACHE_DIR"
    else
        echo "Using existing shared image/normal mmap cache: $IMAGE_MMAP_CACHE_DIR"
    fi
    train_overrides+=(
        --override "dataset.image_load_mode=shared_mmap"
        --override "dataset.image_mmap_cache_dir=$IMAGE_MMAP_CACHE_DIR"
    )
fi

if [ "$RUN_TRAIN" = "1" ]; then
    if [ "$SKIP_TRAINED" = "1" ] && [ -f "$MODEL_PLY" ]; then
        echo "Model already exists at $MODEL_PLY. Skipping train."
    else
        extra_train_args=()
        if [ -n "$EXTRA_TRAIN_ARGS" ]; then
            # shellcheck disable=SC2206
            extra_train_args=($EXTRA_TRAIN_ARGS)
        fi

        echo "Training coarse normal model on GPU $CUDA_ID"
        CUDA_VISIBLE_DEVICES="$CUDA_ID" "$PYTHON_BIN" train.py \
            --config "$CONFIG" \
            "${train_overrides[@]}" \
            "${extra_train_args[@]}"
    fi
fi

if [ "$RUN_RENDER" = "1" ]; then
    if [ ! -f "$MODEL_PLY" ]; then
        echo "Cannot render; missing model: $MODEL_PLY" >&2
        exit 1
    fi
    if [ "$FORCE_RENDER" != "1" ] && [ -d "$TEST_RENDER_DIR" ]; then
        echo "Test renders already exist at $TEST_RENDER_DIR. Skipping render."
    else
        extra_render_args=()
        if [ -n "$EXTRA_RENDER_ARGS" ]; then
            # shellcheck disable=SC2206
            extra_render_args=($EXTRA_RENDER_ARGS)
        fi

        rm -rf "$TRAIN_RENDER_SET"
        if [ "$FORCE_RENDER" = "1" ]; then
            rm -rf "$CFG_MODEL_PATH/test/ours_${CFG_ITERATION}"
        fi

        echo "Rendering MatrixCity aerial test set on GPU $RENDER_CUDA_ID"
        CUDA_VISIBLE_DEVICES="$RENDER_CUDA_ID" "$PYTHON_BIN" render.py \
            --config "$CONFIG" \
            --override "render.iteration=$CFG_ITERATION" \
            "${extra_render_args[@]}"

        if [ ! -d "$TEST_RENDER_DIR" ] && [ -d "$TRAIN_RENDER_SET" ]; then
            mkdir -p "$CFG_MODEL_PATH/test"
            mv "$TRAIN_RENDER_SET" "$CFG_MODEL_PATH/test/"
            rmdir "$CFG_MODEL_PATH/train" 2>/dev/null || true
        fi
        if [ ! -d "$TEST_RENDER_DIR" ]; then
            echo "Render did not produce test renders at $TEST_RENDER_DIR" >&2
            exit 1
        fi
    fi
fi

if [ "$RUN_METRICS" = "1" ]; then
    if [ ! -d "$TEST_RENDER_DIR" ]; then
        echo "Cannot run metrics; missing test renders: $TEST_RENDER_DIR" >&2
        exit 1
    fi

    extra_metrics_args=()
    if [ -n "$EXTRA_METRICS_ARGS" ]; then
        # shellcheck disable=SC2206
        extra_metrics_args=($EXTRA_METRICS_ARGS)
    fi

    echo "Evaluating test renders under $CFG_MODEL_PATH"
    CUDA_VISIBLE_DEVICES="$METRICS_CUDA_ID" "$PYTHON_BIN" metrics.py \
        --config "$CONFIG" \
        "${extra_metrics_args[@]}"
    cp "$CFG_MODEL_PATH/results.json" "$CFG_MODEL_PATH/results_test.json"
    cp "$CFG_MODEL_PATH/per_view.json" "$CFG_MODEL_PATH/per_view_test.json"
fi
