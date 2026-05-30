#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-config/tandt_train_depth.yaml}"
CUDA_ID="${CUDA_ID:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_RENDER="${RUN_RENDER:-0}"
RUN_METRICS="${RUN_METRICS:-0}"
SKIP_TRAINED="${SKIP_TRAINED:-1}"
SKIP_RENDERED="${SKIP_RENDERED:-1}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_RENDER_ARGS="${EXTRA_RENDER_ARGS:-}"
EXTRA_METRICS_ARGS="${EXTRA_METRICS_ARGS:-}"

if [ ! -f "$CONFIG" ]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

eval "$(
    python - "$CONFIG" <<'PY'
import os
import shlex
import sys

from utils.config_utils import load_yaml_config, stage_args_from_config

cfg = load_yaml_config(sys.argv[1])
train_args = stage_args_from_config(cfg, "train")
render_args = stage_args_from_config(cfg, "render")

source_path = train_args["source_path"]
depths = train_args.get("depths", "")

def emit(name, value):
    print(f"{name}={shlex.quote(str(value))}")

emit("CFG_SOURCE_PATH", source_path)
emit("CFG_IMAGES", train_args.get("images", "images"))
emit("CFG_DEPTHS", depths)
emit("CFG_DEPTH_DIR", os.path.join(source_path, depths) if depths else "")
emit("CFG_DEPTH_PARAMS", os.path.join(source_path, "sparse/0/depth_params.json"))
emit("CFG_MODEL_PATH", train_args["model_path"])
emit("CFG_ITERATIONS", train_args["iterations"])
emit("CFG_SWANLAB_EXP_NAME", train_args["swanlab_experiment_name"])
emit("CFG_RENDER_MODEL_PATH", render_args["model_path"])
emit("CFG_RENDER_ITERATION", render_args["iteration"])
emit("CFG_RENDER_DEPTH", render_args.get("render_depth", False))
PY
)"

if [ -z "$CFG_DEPTHS" ]; then
    echo "Depth training requires dataset.depths in $CONFIG" >&2
    exit 1
fi
if [ ! -d "$CFG_DEPTH_DIR" ]; then
    echo "Depth directory not found: $CFG_DEPTH_DIR" >&2
    exit 1
fi
if [ ! -f "$CFG_DEPTH_PARAMS" ]; then
    echo "Depth alignment params not found: $CFG_DEPTH_PARAMS" >&2
    echo "Generate it with scripts/prepare_depth_regularization.sh before training." >&2
    exit 1
fi

MODEL_PLY="$CFG_MODEL_PATH/point_cloud/iteration_${CFG_ITERATIONS}/point_cloud.ply"
TEST_RENDER_DIR="$CFG_RENDER_MODEL_PATH/test/ours_${CFG_RENDER_ITERATION}/renders"
TEST_DEPTH_RENDER_DIR="$CFG_RENDER_MODEL_PATH/test/ours_${CFG_RENDER_ITERATION}/render_inv_depth_vis"

train_args=()
if [ -n "$EXTRA_TRAIN_ARGS" ]; then
    # shellcheck disable=SC2206
    train_args=($EXTRA_TRAIN_ARGS)
fi

render_args=()
if [ -n "$EXTRA_RENDER_ARGS" ]; then
    # shellcheck disable=SC2206
    render_args=($EXTRA_RENDER_ARGS)
fi

metrics_args=()
if [ -n "$EXTRA_METRICS_ARGS" ]; then
    # shellcheck disable=SC2206
    metrics_args=($EXTRA_METRICS_ARGS)
fi

if [ "$RUN_TRAIN" = "1" ]; then
    if [ "$SKIP_TRAINED" = "1" ] && [ -f "$MODEL_PLY" ]; then
        echo "Depth model already exists: $MODEL_PLY"
        echo "Skipping train. Set SKIP_TRAINED=0 to retrain."
    else
        echo "Training T&T with depth regularization"
        echo "  config: $CONFIG"
        echo "  source: $CFG_SOURCE_PATH"
        echo "  images: $CFG_IMAGES"
        echo "  depths: $CFG_DEPTHS"
        echo "  output: $CFG_MODEL_PATH"
        echo "  iterations: $CFG_ITERATIONS"
        echo "  cuda: $CUDA_ID"
        echo "  swanlab: $CFG_SWANLAB_EXP_NAME"
        CUDA_VISIBLE_DEVICES="$CUDA_ID" python train.py \
            --config "$CONFIG" \
            "${train_args[@]}"
    fi
else
    echo "RUN_TRAIN=0, skipping train."
fi

if [ "$RUN_RENDER" = "1" ]; then
    if [ ! -f "$MODEL_PLY" ]; then
        echo "Cannot render; missing model: $MODEL_PLY" >&2
        exit 1
    fi
    render_done=0
    if [ -d "$TEST_RENDER_DIR" ]; then
        if [ "$CFG_RENDER_DEPTH" = "True" ]; then
            if [ -d "$TEST_DEPTH_RENDER_DIR" ]; then
                render_done=1
            fi
        else
            render_done=1
        fi
    fi
    if [ "$SKIP_RENDERED" = "1" ] && [ "$render_done" = "1" ]; then
        echo "Test renders already exist: $TEST_RENDER_DIR"
        if [ "$CFG_RENDER_DEPTH" = "True" ]; then
            echo "Test depth renders already exist: $TEST_DEPTH_RENDER_DIR"
        fi
        echo "Skipping render. Set SKIP_RENDERED=0 to rerender."
    else
        echo "Rendering T&T depth model test views on GPU $CUDA_ID"
        CUDA_VISIBLE_DEVICES="$CUDA_ID" python render.py \
            --config "$CONFIG" \
            "${render_args[@]}"
    fi
else
    echo "RUN_RENDER=0, skipping render."
fi

if [ "$RUN_METRICS" = "1" ]; then
    if [ ! -d "$TEST_RENDER_DIR" ]; then
        echo "Cannot run metrics; missing test renders: $TEST_RENDER_DIR" >&2
        exit 1
    fi
    echo "Evaluating T&T depth renders on GPU $CUDA_ID"
    CUDA_VISIBLE_DEVICES="$CUDA_ID" python metrics.py \
        --config "$CONFIG" \
        "${metrics_args[@]}"
else
    echo "RUN_METRICS=0, skipping metrics."
fi

echo "Depth model:"
echo "  $MODEL_PLY"
