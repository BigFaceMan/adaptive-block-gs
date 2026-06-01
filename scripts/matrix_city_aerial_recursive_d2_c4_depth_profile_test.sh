#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-config/mc_aerial_recursive_d2_c4_depth.yaml}"
BLOCK_ID="${BLOCK_ID:-block_000}"
CUDA_ID="${CUDA_ID:-2}"
TEST_ITERATIONS="${TEST_ITERATIONS:-300}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/mc_aerial_recursive_d2_c4_depth_profile_test}"
PARTITION_TREE="${PARTITION_TREE:-output/mc_aerial_recursive_d2_c4_depth/partitions/partition_tree.json}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

if [ ! -f "$CONFIG" ]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi
if [ ! -f "$PARTITION_TREE" ]; then
    echo "Partition tree not found: $PARTITION_TREE" >&2
    echo "Run/copy the d2 c4 partition first." >&2
    exit 1
fi

echo "Running short depth-reg block profile"
echo "  config:        $CONFIG"
echo "  block:         $BLOCK_ID"
echo "  iterations:    $TEST_ITERATIONS"
echo "  cuda:          $CUDA_ID"
echo "  output_root:   $OUTPUT_ROOT"
echo "  partition:     $PARTITION_TREE"
echo "  swanlab_mode:  $SWANLAB_MODE"

train_args=(
    --config "$CONFIG"
    --block_id "$BLOCK_ID"
    --override "experiment.name=mc_aerial_recursive_d2_c4_depth_profile_test"
    --override "experiment.output_root=$OUTPUT_ROOT"
    --override "block_training.partition_path=$PARTITION_TREE"
    --override "block_training.blocks_root=$OUTPUT_ROOT/blocks"
    --override "optimization.iterations=$TEST_ITERATIONS"
    --override "optimization.densify_until_iter=$TEST_ITERATIONS"
    --override "block_training.test_iterations=[-1]"
    --override "block_training.save_iterations=[$TEST_ITERATIONS]"
    --override "block_training.checkpoint_iterations=[]"
    --override "logging.swanlab_experiment_prefix=mc_aerial_recursive_d2_c4_depth_profile_test"
    --override "logging.swanlab_experiment_name=mc_aerial_recursive_d2_c4_depth_profile_test-$BLOCK_ID"
    --override "logging.swanlab_mode=$SWANLAB_MODE"
)

if [ -n "$EXTRA_TRAIN_ARGS" ]; then
    # shellcheck disable=SC2206
    train_args+=($EXTRA_TRAIN_ARGS)
fi

CUDA_VISIBLE_DEVICES="$CUDA_ID" python train.py "${train_args[@]}"

echo "Profile test finished."
echo "  block output: $OUTPUT_ROOT/blocks/$BLOCK_ID"
echo "  log dir:      $OUTPUT_ROOT/blocks/$BLOCK_ID/swanlog"
