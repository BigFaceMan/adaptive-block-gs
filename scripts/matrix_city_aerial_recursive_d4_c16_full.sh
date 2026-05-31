#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-config/mc_aerial_recursive_d4_c16.yaml}"
CUDA_IDS="${CUDA_IDS:-6,7,8,9}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

RUN_PARTITION="${RUN_PARTITION:-1}"
RUN_TRAIN_BLOCKS="${RUN_TRAIN_BLOCKS:-1}"
RUN_MERGE="${RUN_MERGE:-1}"
RUN_RENDER_EVAL="${RUN_RENDER_EVAL:-1}"
RUN_METRICS="${RUN_METRICS:-1}"

FORCE_PARTITION="${FORCE_PARTITION:-0}"
SKIP_TRAINED_BLOCKS="${SKIP_TRAINED_BLOCKS:-1}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-1}"
MERGE_ALLOW_MISSING="${MERGE_ALLOW_MISSING:-0}"

EXTRA_PARTITION_ARGS="${EXTRA_PARTITION_ARGS:-}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_MERGE_ARGS="${EXTRA_MERGE_ARGS:-}"
EXTRA_RENDER_ARGS="${EXTRA_RENDER_ARGS:-}"

if [ ! -f "$CONFIG" ]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

PARTITION_TREE="$(
    python -c 'import sys; from utils.config_utils import load_yaml_config, partition_tree_path; print(partition_tree_path(load_yaml_config(sys.argv[1])))' "$CONFIG"
)"

partition_args=(
    --config "$CONFIG"
    --override partition.max_depth=4
    --override partition.max_blocks=16
)
if [ -n "$EXTRA_PARTITION_ARGS" ]; then
    # shellcheck disable=SC2206
    partition_args+=($EXTRA_PARTITION_ARGS)
fi

if [ "$RUN_PARTITION" = "1" ]; then
    if [ -f "$PARTITION_TREE" ] && [ "$FORCE_PARTITION" != "1" ]; then
        echo "Partition tree already exists: $PARTITION_TREE"
        echo "Skipping partition. Set FORCE_PARTITION=1 to regenerate it."
    else
        echo "Running recursive partition with max depth 4 and max 16 blocks..."
        python partition.py "${partition_args[@]}"
    fi
else
    echo "RUN_PARTITION=0, skipping partition."
fi

if [ ! -f "$PARTITION_TREE" ]; then
    echo "Partition tree missing after partition step: $PARTITION_TREE" >&2
    exit 1
fi

echo "Running block training / merge / render eval / metrics..."
CONFIG="$CONFIG" \
CUDA_IDS="$CUDA_IDS" \
MAX_PARALLEL="$MAX_PARALLEL" \
TRAIN_BLOCKS="$RUN_TRAIN_BLOCKS" \
MERGE_BLOCKS="$RUN_MERGE" \
RENDER_TEST_SET="$RUN_RENDER_EVAL" \
RUN_METRICS="$RUN_METRICS" \
SKIP_TRAINED_BLOCKS="$SKIP_TRAINED_BLOCKS" \
STOP_ON_FAILURE="$STOP_ON_FAILURE" \
MERGE_ALLOW_MISSING="$MERGE_ALLOW_MISSING" \
EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS" \
EXTRA_MERGE_ARGS="$EXTRA_MERGE_ARGS" \
EXTRA_RENDER_ARGS="$EXTRA_RENDER_ARGS" \
bash scripts/matrix_city_aerial_train_blocks.sh
