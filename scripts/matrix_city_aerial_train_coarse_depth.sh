#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-config/mc_aerial_coarse_depth.yaml}"
CUDA_ID="${CUDA_ID:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_RENDER_TEST="${RUN_RENDER_TEST:-1}"
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
emit("CFG_TEST_SOURCE_PATH", train_args.get("test_source_path", ""))
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

train_model() {
    local extra_args=()
    if [ -n "$EXTRA_TRAIN_ARGS" ]; then
        # shellcheck disable=SC2206
        extra_args=($EXTRA_TRAIN_ARGS)
    fi

    echo "Training coarse GS with depth regularization"
    echo "  config: $CONFIG"
    echo "  source: $CFG_SOURCE_PATH"
    echo "  images: $CFG_IMAGES"
    echo "  depths: $CFG_DEPTHS"
    echo "  depth params: $CFG_DEPTH_PARAMS"
    echo "  output: $CFG_MODEL_PATH"
    echo "  iterations: $CFG_ITERATIONS"
    echo "  cuda: $CUDA_ID"
    echo "  swanlab: $CFG_SWANLAB_EXP_NAME"

    CUDA_VISIBLE_DEVICES="$CUDA_ID" python train.py \
        --config "$CONFIG" \
        "${extra_args[@]}"
}

render_test_set() {
    local extra_args=()
    if [ -n "$EXTRA_RENDER_ARGS" ]; then
        # shellcheck disable=SC2206
        extra_args=($EXTRA_RENDER_ARGS)
    fi

    echo "Rendering coarse depth GS test set"
    echo "  test source: $CFG_TEST_SOURCE_PATH"
    echo "  output: $TEST_RENDER_DIR"

    CUDA_VISIBLE_DEVICES="$CUDA_ID" python render.py \
        --config "$CONFIG" \
        "${extra_args[@]}"
}

run_metrics() {
    local extra_args=()
    if [ -n "$EXTRA_METRICS_ARGS" ]; then
        # shellcheck disable=SC2206
        extra_args=($EXTRA_METRICS_ARGS)
    fi

    echo "Evaluating coarse depth GS test renders"
    CUDA_VISIBLE_DEVICES="$CUDA_ID" python metrics.py \
        --config "$CONFIG" \
        "${extra_args[@]}"

    cp "$CFG_MODEL_PATH/results.json" "$CFG_MODEL_PATH/results_test.json"
    cp "$CFG_MODEL_PATH/per_view.json" "$CFG_MODEL_PATH/per_view_test.json"
}

if [ "$RUN_TRAIN" = "1" ]; then
    if [ "$SKIP_TRAINED" = "1" ] && [ -f "$MODEL_PLY" ]; then
        echo "Coarse depth model already exists: $MODEL_PLY"
        echo "Skipping train. Set SKIP_TRAINED=0 to retrain."
    else
        train_model
    fi
else
    echo "RUN_TRAIN=0, skipping train."
fi

if [ ! -f "$MODEL_PLY" ]; then
    echo "Coarse depth model not found: $MODEL_PLY" >&2
    exit 1
fi

if [ "$RUN_RENDER_TEST" = "1" ]; then
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
        render_test_set
    fi
else
    echo "RUN_RENDER_TEST=0, skipping test render."
fi

if [ "$RUN_METRICS" = "1" ]; then
    if [ ! -d "$TEST_RENDER_DIR" ]; then
        echo "Test render directory not found: $TEST_RENDER_DIR" >&2
        exit 1
    fi
    run_metrics
else
    echo "RUN_METRICS=0, skipping metrics."
fi

echo "Coarse depth model:"
echo "  $MODEL_PLY"
