#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-config/mc_aerial_recursive_d5_c32_depth_core_w05.yaml}"
CUDA_IDS="${CUDA_IDS:-2,3,4,5,6,7,8,9}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"

RUN_TRAIN_BLOCKS="${RUN_TRAIN_BLOCKS:-1}"
RUN_MERGE="${RUN_MERGE:-1}"
RUN_RENDER_EVAL="${RUN_RENDER_EVAL:-1}"
RUN_METRICS="${RUN_METRICS:-1}"

SKIP_TRAINED_BLOCKS="${SKIP_TRAINED_BLOCKS:-1}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-1}"
MERGE_ALLOW_MISSING="${MERGE_ALLOW_MISSING:-0}"

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

if [ ! -f "$PARTITION_TREE" ]; then
    echo "Partition tree missing: $PARTITION_TREE" >&2
    echo "This ablation reuses the existing d5/c32 partition; generate it before running this script." >&2
    exit 1
fi

echo "Running d5/c32 depth core-w02 block training with 6-GPU defaults..."
echo "Config: $CONFIG"
echo "Partition: $PARTITION_TREE"
echo "CUDA_IDS: $CUDA_IDS"
echo "MAX_PARALLEL: $MAX_PARALLEL"

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
