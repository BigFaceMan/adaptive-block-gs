#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-config/tandt_train.yaml}"
CUDA_ID="${CUDA_ID:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_RENDER="${RUN_RENDER:-1}"
RUN_METRICS="${RUN_METRICS:-1}"
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

def emit(name, value):
    print(f"{name}={shlex.quote(str(value))}")

emit("MODEL_PATH", train_args["model_path"])
emit("RENDER_MODEL_PATH", render_args["model_path"])
emit("ITERATION", render_args["iteration"])
PY
)"

MODEL_PLY="$MODEL_PATH/point_cloud/iteration_${ITERATION}/point_cloud.ply"
TEST_RENDER_DIR="$RENDER_MODEL_PATH/test/ours_${ITERATION}/renders"

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
        echo "Model already exists: $MODEL_PLY"
        echo "Skipping train. Set SKIP_TRAINED=0 to retrain."
    else
        echo "Training T&T on GPU $CUDA_ID"
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
    if [ "$SKIP_RENDERED" = "1" ] && [ -d "$TEST_RENDER_DIR" ]; then
        echo "Test renders already exist: $TEST_RENDER_DIR"
        echo "Skipping render. Set SKIP_RENDERED=0 to rerender."
    else
        echo "Rendering T&T test views on GPU $CUDA_ID"
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
    echo "Evaluating T&T renders on GPU $CUDA_ID"
    CUDA_VISIBLE_DEVICES="$CUDA_ID" python metrics.py \
        --config "$CONFIG" \
        "${metrics_args[@]}"
else
    echo "RUN_METRICS=0, skipping metrics."
fi
