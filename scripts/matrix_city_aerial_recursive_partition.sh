#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_CONFIG="${DEFAULT_CONFIG-config/mc_aerial_recursive_c4.yaml}"
CONFIG="${CONFIG-$DEFAULT_CONFIG}"
EXTRA_PARTITION_ARGS="${EXTRA_PARTITION_ARGS-}"
if [ -n "$CONFIG" ]; then
    extra_partition_args=()
    if [ -n "$EXTRA_PARTITION_ARGS" ]; then
        # shellcheck disable=SC2206
        extra_partition_args=($EXTRA_PARTITION_ARGS)
    fi
    python partition.py --config "$CONFIG" "${extra_partition_args[@]}"
    exit 0
fi

TRAIN_DATA_PATH="${TRAIN_DATA_PATH-/lfs3/users/spsong/dataset/MatrixCity/small_city/aerial/train/block_all}"
TEST_DATA_PATH="${TEST_DATA_PATH-/lfs3/users/spsong/dataset/MatrixCity/small_city/aerial/test/block_all_test}"
COARSE_MODEL="${COARSE_MODEL-/lfs1/users/spsong/Code/project/adaptive-block-gs/output/mc_aerial_block_all_30000/point_cloud/iteration_30000/point_cloud.ply}"
PARTITION_OUTPUT="${PARTITION_OUTPUT-/lfs1/users/spsong/Code/project/adaptive-block-gs/output/mc_aerial_recursive_depth6/partitions}"
IMAGES="${IMAGES-input}"
TEST_IMAGES="${TEST_IMAGES-$IMAGES}"
TEST_DEPTHS="${TEST_DEPTHS-}"

MAX_DEPTH="${MAX_DEPTH-6}"
MAX_BLOCKS="${MAX_BLOCKS-36}"
MIN_POINTS="${MIN_POINTS-50000}"
MIN_CAMERAS="${MIN_CAMERAS-20}"
MIN_TEST_CAMERAS="${MIN_TEST_CAMERAS-20}"
EXPAND_RATIO="${EXPAND_RATIO-0.005}"
TAU_PROJECTION="${TAU_PROJECTION-0.8}"
TAU_TEST_PROJECTION="${TAU_TEST_PROJECTION-0.02}"
LAMBDA_BOUNDARY="${LAMBDA_BOUNDARY-0.2}"
NUM_SPLIT_CANDIDATES="${NUM_SPLIT_CANDIDATES-64}"
IMPORTANCE="${IMPORTANCE-opacity_scale}"
SUPPLEMENT_CAMERAS="${SUPPLEMENT_CAMERAS-1}"
PARTITION_AXES="${PARTITION_AXES-x y}"
PARTITION_COORD_SPACE="${PARTITION_COORD_SPACE-contracted}"
CONTRACT_AABB="${CONTRACT_AABB-}"
RENDER_DIFFERENCE_CAMERAS="${RENDER_DIFFERENCE_CAMERAS-1}"
RENDER_DIFFERENCE_THRESHOLD="${RENDER_DIFFERENCE_THRESHOLD-0.03}"
RENDER_DIFFERENCE_MAX_CANDIDATES_PER_BLOCK="${RENDER_DIFFERENCE_MAX_CANDIDATES_PER_BLOCK-0}"
RENDER_DIFFERENCE_MAX_WIDTH="${RENDER_DIFFERENCE_MAX_WIDTH-800}"
RENDER_DIFFERENCE_CACHE_FULL="${RENDER_DIFFERENCE_CACHE_FULL-0}"
CAMERA_PROJECTION_MAX_POINTS="${CAMERA_PROJECTION_MAX_POINTS-5000}"
VISUALIZE_BLOCKS="${VISUALIZE_BLOCKS-1}"
VISUALIZE_TOPDOWN_MAX_POINTS="${VISUALIZE_TOPDOWN_MAX_POINTS-0}"
VISUALIZE_TOPDOWN_IMAGE_SIZE="${VISUALIZE_TOPDOWN_IMAGE_SIZE-4096}"
VISUALIZE_TOPDOWN_POINT_RADIUS="${VISUALIZE_TOPDOWN_POINT_RADIUS-0}"
VISUALIZE_TOPDOWN_BBOX_MODE="${VISUALIZE_TOPDOWN_BBOX_MODE-both}"
VISUALIZE_TOPDOWN_CLIP_PERCENTILE="${VISUALIZE_TOPDOWN_CLIP_PERCENTILE-0.2}"
VISUALIZE_TOPDOWN_COLOR="${VISUALIZE_TOPDOWN_COLOR-auto}"
VISUALIZE_TOPDOWN_KEEP="${VISUALIZE_TOPDOWN_KEEP-max}"

visualize_args=()
if [ "$VISUALIZE_BLOCKS" = "1" ]; then
    visualize_args=(
        --visualize_topdown
        --visualize_topdown_max_points "$VISUALIZE_TOPDOWN_MAX_POINTS"
        --visualize_topdown_image_size "$VISUALIZE_TOPDOWN_IMAGE_SIZE"
        --visualize_topdown_point_radius "$VISUALIZE_TOPDOWN_POINT_RADIUS"
        --visualize_topdown_bbox_mode "$VISUALIZE_TOPDOWN_BBOX_MODE"
        --visualize_topdown_clip_percentile "$VISUALIZE_TOPDOWN_CLIP_PERCENTILE"
        --visualize_topdown_color "$VISUALIZE_TOPDOWN_COLOR"
        --visualize_topdown_keep "$VISUALIZE_TOPDOWN_KEEP"
    )
fi

supplement_args=()
if [ "$SUPPLEMENT_CAMERAS" = "1" ]; then
    supplement_args=(--supplement_cameras)
fi

contract_args=(--partition_coord_space "$PARTITION_COORD_SPACE")
if [ -n "$CONTRACT_AABB" ]; then
    # shellcheck disable=SC2206
    contract_values=($CONTRACT_AABB)
    contract_args+=(--contract_aabb "${contract_values[@]}")
fi

render_diff_args=(
    --camera_projection_max_points "$CAMERA_PROJECTION_MAX_POINTS"
)
if [ "$RENDER_DIFFERENCE_CAMERAS" = "1" ]; then
    render_diff_args+=(
        --render_difference_cameras
        --render_difference_threshold "$RENDER_DIFFERENCE_THRESHOLD"
        --render_difference_max_candidates_per_block "$RENDER_DIFFERENCE_MAX_CANDIDATES_PER_BLOCK"
        --render_difference_max_width "$RENDER_DIFFERENCE_MAX_WIDTH"
    )
    if [ "$RENDER_DIFFERENCE_CACHE_FULL" = "1" ]; then
        render_diff_args+=(--render_difference_cache_full)
    fi
fi

test_args=()
if [ -n "$TEST_DATA_PATH" ]; then
    test_args=(
        --test_source_path "$TEST_DATA_PATH"
        --test_images "$TEST_IMAGES"
        --test_depths "$TEST_DEPTHS"
        --tau_test_projection "$TAU_TEST_PROJECTION"
        --min_test_cameras "$MIN_TEST_CAMERAS"
    )
fi

python partition.py \
    -s "$TRAIN_DATA_PATH" \
    --images "$IMAGES" \
    --coarse_model "$COARSE_MODEL" \
    --partition_output "$PARTITION_OUTPUT" \
    --partition_axes $PARTITION_AXES \
    --max_depth "$MAX_DEPTH" \
    --max_blocks "$MAX_BLOCKS" \
    --min_points "$MIN_POINTS" \
    --min_cameras "$MIN_CAMERAS" \
    --expand_ratio "$EXPAND_RATIO" \
    --tau_projection "$TAU_PROJECTION" \
    --lambda_boundary "$LAMBDA_BOUNDARY" \
    --num_split_candidates "$NUM_SPLIT_CANDIDATES" \
    --importance "$IMPORTANCE" \
    "${contract_args[@]}" \
    "${render_diff_args[@]}" \
    "${test_args[@]}" \
    "${supplement_args[@]}" \
    "${visualize_args[@]}"
