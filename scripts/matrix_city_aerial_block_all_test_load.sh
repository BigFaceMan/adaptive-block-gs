#!/usr/bin/env bash
set -euo pipefail

TRAIN_DATA_PATH="/lfs3/users/spsong/dataset/MatrixCity/small_city/aerial/train/block_all"
TEST_DATA_PATH="/lfs3/users/spsong/dataset/MatrixCity/small_city/aerial/test/block_all_test"
OUTPUT_PATH="/lfs1/users/spsong/Code/project/gaussian-splatting/output/test_load"
CUDA_ID=5
ITERATIONS=30000
SWANLAB_EXP_NAME="test_load"
START_CHECKPOINT="${START_CHECKPOINT-$OUTPUT_PATH/chkpnt7000.pth}"
CAMERA_LOAD_WORKERS="${CAMERA_LOAD_WORKERS-16}"
DATA_DEVICE="${DATA_DEVICE-cpu}"

MODEL_DIR="$OUTPUT_PATH/point_cloud/iteration_${ITERATIONS}"
TRAIN_RENDER_DIR="$OUTPUT_PATH/train/ours_${ITERATIONS}/renders"
TEST_RENDER_DIR="$OUTPUT_PATH/test/ours_${ITERATIONS}/renders"

train_model() {
    local train_args=(
        -s "$TRAIN_DATA_PATH" \
        --images input \
        -m "$OUTPUT_PATH" \
        --iterations "$ITERATIONS" \
        --test_iterations 7000 "$ITERATIONS" \
        --save_iterations 7000 "$ITERATIONS" \
        --camera_load_workers "$CAMERA_LOAD_WORKERS" \
        --data_device "$DATA_DEVICE" \
        --swanlab_experiment_name "$SWANLAB_EXP_NAME"
    )

    echo "Camera loading:"
    echo "  camera_load_workers=$CAMERA_LOAD_WORKERS"
    echo "  data_device=$DATA_DEVICE"

    if [ -n "$START_CHECKPOINT" ]; then
        if [ ! -f "$START_CHECKPOINT" ]; then
            echo "Start checkpoint not found: $START_CHECKPOINT" >&2
            exit 1
        fi
        echo "Resuming training from checkpoint: $START_CHECKPOINT"
        train_args+=(--start_checkpoint "$START_CHECKPOINT")
    fi

    CUDA_VISIBLE_DEVICES=$CUDA_ID python train.py "${train_args[@]}"
}

render_train_split() {
    CUDA_VISIBLE_DEVICES=$CUDA_ID python render.py \
        -m "$OUTPUT_PATH" \
        -s "$TRAIN_DATA_PATH" \
        --images input \
        --iteration "$ITERATIONS" \
        --camera_load_workers "$CAMERA_LOAD_WORKERS" \
        --data_device "$DATA_DEVICE" \
        --skip_test
}

render_test_split() {
    local tmp_render_root
    tmp_render_root="$(mktemp -d "${OUTPUT_PATH}/test_render_tmp.XXXXXX")"

    # render.py only writes the active split to train/ or test/ under model_path.
    # Use a temporary model root so the external test set does not overwrite train renders.
    ln -s "$OUTPUT_PATH/point_cloud" "$tmp_render_root/point_cloud"
    cp "$OUTPUT_PATH/cfg_args" "$tmp_render_root/cfg_args"

    CUDA_VISIBLE_DEVICES=$CUDA_ID python render.py \
        -m "$tmp_render_root" \
        -s "$TEST_DATA_PATH" \
        --images input \
        --iteration "$ITERATIONS" \
        --camera_load_workers "$CAMERA_LOAD_WORKERS" \
        --data_device "$DATA_DEVICE" \
        --skip_test

    mkdir -p "$OUTPUT_PATH/test"
    mv "$tmp_render_root/train/ours_${ITERATIONS}" "$OUTPUT_PATH/test/"

    rm "$tmp_render_root/point_cloud"
    rm "$tmp_render_root/cfg_args"
    rmdir "$tmp_render_root/train"
    rmdir "$tmp_render_root"
}

run_test_metrics() {
    CUDA_VISIBLE_DEVICES=$CUDA_ID python metrics.py -m "$OUTPUT_PATH" -d test
    cp "$OUTPUT_PATH/results.json" "$OUTPUT_PATH/results_test.json"
    cp "$OUTPUT_PATH/per_view.json" "$OUTPUT_PATH/per_view_test.json"
}

if [ -d "$MODEL_DIR" ] && [ -d "$TRAIN_RENDER_DIR" ] && [ -d "$TEST_RENDER_DIR" ]; then
    echo "Train and both renders already completed. Running test metrics only..."
    run_test_metrics
    exit 0
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "Training model on $TRAIN_DATA_PATH ..."
    train_model
else
    echo "Training already completed. Skipping train."
fi

if [ ! -d "$TEST_RENDER_DIR" ]; then
    echo "Rendering external test split from $TEST_DATA_PATH ..."
    render_test_split
else
    echo "External test render already completed. Skipping test render."
fi

if [ ! -d "$TRAIN_RENDER_DIR" ]; then
    echo "Rendering training split from $TRAIN_DATA_PATH ..."
    render_train_split
else
    echo "Train render already completed. Skipping train render."
fi

run_test_metrics
