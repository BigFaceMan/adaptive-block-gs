#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-config/mc_aerial_recursive_d5_c32_post.yaml}"
CUDA_ID="${CUDA_ID:-${CUDA_IDS:-0}}"
CUDA_ID="${CUDA_ID%%,*}"
RENDER_CUDA_ID="${RENDER_CUDA_ID:-$CUDA_ID}"

RUN_POST_TRAIN="${RUN_POST_TRAIN:-1}"
RUN_RENDER_EVAL="${RUN_RENDER_EVAL:-1}"
RUN_METRICS="${RUN_METRICS:-1}"
SKIP_POST_TRAINED="${SKIP_POST_TRAINED:-1}"
SKIP_RENDERED="${SKIP_RENDERED:-1}"

EXTRA_POST_TRAIN_ARGS="${EXTRA_POST_TRAIN_ARGS:-}"
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
post = stage_args_from_config(cfg, "post_train")
render = stage_args_from_config(cfg, "render")

def emit(name, value):
    print(f"{name}={shlex.quote(str(value))}")

emit("CFG_POST_OUTPUT_PATH", post["output_path"])
emit("CFG_POST_ITERATION", post["iterations"])
emit("CFG_RENDER_ITERATION", render["iteration"])
emit(
    "CFG_INPUT_PLY",
    os.path.join(
        post["input_model_path"],
        "point_cloud",
        f"iteration_{post['input_iteration']}",
        "point_cloud.ply",
    ),
)
PY
)"

POST_OUTPUT_PATH="${POST_OUTPUT_PATH:-$CFG_POST_OUTPUT_PATH}"
POST_ITERATION="${POST_ITERATION:-$CFG_POST_ITERATION}"
RENDER_ITERATION="${RENDER_ITERATION:-$CFG_RENDER_ITERATION}"
POST_PLY="$POST_OUTPUT_PATH/point_cloud/iteration_${POST_ITERATION}/point_cloud.ply"
TEST_RENDER_DIR="$POST_OUTPUT_PATH/test/ours_${RENDER_ITERATION}/renders"

if [ "$RUN_POST_TRAIN" = "1" ]; then
    if [ "$SKIP_POST_TRAINED" = "1" ] && [ -f "$POST_PLY" ]; then
        echo "Post-trained model already exists: $POST_PLY"
        echo "Skipping post-train. Set SKIP_POST_TRAINED=0 to rerun it."
    else
        if [ ! -f "$CFG_INPUT_PLY" ]; then
            echo "Input merged PLY not found: $CFG_INPUT_PLY" >&2
            exit 1
        fi

        post_extra_args=()
        if [ -n "$EXTRA_POST_TRAIN_ARGS" ]; then
            # shellcheck disable=SC2206
            post_extra_args=($EXTRA_POST_TRAIN_ARGS)
        fi
        echo "Running post-train on GPU $CUDA_ID with config $CONFIG"
        CUDA_VISIBLE_DEVICES="$CUDA_ID" python post_train.py \
            --config "$CONFIG" \
            "${post_extra_args[@]}"
    fi
else
    echo "RUN_POST_TRAIN=0, skipping post-train."
fi

if [ "$RUN_RENDER_EVAL" = "1" ]; then
    if [ "$SKIP_RENDERED" = "1" ] && [ -d "$TEST_RENDER_DIR" ]; then
        echo "Post-train test render already exists: $TEST_RENDER_DIR"
        echo "Skipping render. Set SKIP_RENDERED=0 to rerun it."
    else
        if [ ! -f "$POST_PLY" ]; then
            echo "Post-trained PLY not found: $POST_PLY" >&2
            exit 1
        fi

        render_extra_args=()
        if [ -n "$EXTRA_RENDER_ARGS" ]; then
            # shellcheck disable=SC2206
            render_extra_args=($EXTRA_RENDER_ARGS)
        fi

        echo "Rendering post-trained model on GPU $RENDER_CUDA_ID"
        CUDA_VISIBLE_DEVICES="$RENDER_CUDA_ID" python render.py \
            --config "$CONFIG" \
            --override "render.iteration=$RENDER_ITERATION" \
            "${render_extra_args[@]}"

        if [ -d "$POST_OUTPUT_PATH/train/ours_${RENDER_ITERATION}" ]; then
            mkdir -p "$POST_OUTPUT_PATH/test"
            rm -rf "$POST_OUTPUT_PATH/test/ours_${RENDER_ITERATION}"
            mv "$POST_OUTPUT_PATH/train/ours_${RENDER_ITERATION}" "$POST_OUTPUT_PATH/test/"
            rmdir "$POST_OUTPUT_PATH/train" 2>/dev/null || true
        fi
        if [ ! -d "$TEST_RENDER_DIR" ]; then
            echo "Render did not produce test renders at $TEST_RENDER_DIR" >&2
            exit 1
        fi
    fi
else
    echo "RUN_RENDER_EVAL=0, skipping render."
fi

if [ "$RUN_METRICS" = "1" ]; then
    if [ ! -d "$TEST_RENDER_DIR" ]; then
        echo "Cannot run metrics; missing test renders: $TEST_RENDER_DIR" >&2
        exit 1
    fi

    metrics_extra_args=()
    if [ -n "$EXTRA_METRICS_ARGS" ]; then
        # shellcheck disable=SC2206
        metrics_extra_args=($EXTRA_METRICS_ARGS)
    fi

    echo "Evaluating post-trained test renders under $POST_OUTPUT_PATH"
    CUDA_VISIBLE_DEVICES="$RENDER_CUDA_ID" python metrics.py \
        --config "$CONFIG" \
        "${metrics_extra_args[@]}"
    cp "$POST_OUTPUT_PATH/results.json" "$POST_OUTPUT_PATH/results_test.json"
    cp "$POST_OUTPUT_PATH/per_view.json" "$POST_OUTPUT_PATH/per_view_test.json"
else
    echo "RUN_METRICS=0, skipping metrics."
fi
