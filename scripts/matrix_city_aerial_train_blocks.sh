#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG-}"
if [ -n "$CONFIG" ]; then
    eval "$(
        python - "$CONFIG" <<'PY'
import os
import shlex
import sys

from utils.config_utils import config_name, get_in, load_yaml_config, merge_args, partition_tree_path

cfg = load_yaml_config(sys.argv[1])
output_root = get_in(cfg, "experiment.output_root", os.path.join("output", config_name(cfg)))
block_training = cfg.get("block_training", {})
dataset = cfg.get("dataset", {})
merge = merge_args(cfg)
optimization = cfg.get("optimization", {})
logging = cfg.get("logging", {})

def emit(name, value):
    print(f"{name}={shlex.quote(str(value))}")

emit("CFG_TRAIN_DATA_PATH", dataset.get("source_path", ""))
emit("CFG_TEST_DATA_PATH", dataset.get("test_source_path", ""))
emit("CFG_RUN_ROOT", output_root)
emit("CFG_PARTITION_PATH", block_training.get("partition_path") or partition_tree_path(cfg))
emit("CFG_OUTPUT_ROOT", block_training.get("blocks_root", os.path.join(output_root, "blocks")))
emit("CFG_MERGE_OUTPUT_PATH", merge["output_path"])
emit("CFG_IMAGES", dataset.get("images", "images"))
emit("CFG_DEPTHS", dataset.get("depths", ""))
emit("CFG_TEST_IMAGES", dataset.get("test_images") or dataset.get("images", "images"))
emit("CFG_TEST_DEPTHS", dataset.get("test_depths", ""))
emit("CFG_ITERATIONS", optimization.get("iterations", 30000))
emit("CFG_MERGE_ITERATION", merge.get("iteration", optimization.get("iterations", 30000)))
emit("CFG_LOG_ROOT", os.path.join(output_root, "logs"))
emit("CFG_SWANLAB_PROJECT", logging.get("swanlab_project", "block-gs"))
emit("CFG_SWANLAB_WORKSPACE", logging.get("swanlab_workspace", ""))
emit("CFG_SWANLAB_MODE", logging.get("swanlab_mode", "cloud"))
emit("CFG_SWANLAB_EXP_PREFIX", logging.get("swanlab_experiment_prefix", config_name(cfg)))
PY
    )"
fi

TRAIN_DATA_PATH="${TRAIN_DATA_PATH-${CFG_TRAIN_DATA_PATH:-/lfs3/users/spsong/dataset/MatrixCity/small_city/aerial/train/block_all}}"
TEST_DATA_PATH="${TEST_DATA_PATH-${CFG_TEST_DATA_PATH:-/lfs3/users/spsong/dataset/MatrixCity/small_city/aerial/test/block_all_test}}"
RUN_ROOT="${RUN_ROOT-${CFG_RUN_ROOT:-/lfs1/users/spsong/Code/project/gaussian-splatting/output/mc_aerial_recursive_depth3}}"
PARTITION_PATH="${PARTITION_PATH-${CFG_PARTITION_PATH:-$RUN_ROOT/partitions/partition_tree.json}}"
OUTPUT_ROOT="${OUTPUT_ROOT-${CFG_OUTPUT_ROOT:-$RUN_ROOT/blocks_coarse_cache}}"
MERGE_OUTPUT_PATH="${MERGE_OUTPUT_PATH-${CFG_MERGE_OUTPUT_PATH:-$RUN_ROOT/merged_coarse_cache}}"
IMAGES="${IMAGES-${CFG_IMAGES:-input}}"
DEPTHS="${DEPTHS-${CFG_DEPTHS:-}}"
TEST_IMAGES="${TEST_IMAGES-${CFG_TEST_IMAGES:-$IMAGES}}"
TEST_DEPTHS="${TEST_DEPTHS-${CFG_TEST_DEPTHS:-}}"
CUDA_IDS="${CUDA_IDS-9}"
MAX_PARALLEL="${MAX_PARALLEL-}"
ITERATIONS="${ITERATIONS-${CFG_ITERATIONS:-30000}}"
BLOCK_TEST_ITERATIONS="${BLOCK_TEST_ITERATIONS--1}"
MERGE_ITERATION="${MERGE_ITERATION-${CFG_MERGE_ITERATION:-$ITERATIONS}}"
TRAIN_BLOCKS="${TRAIN_BLOCKS-1}"
SKIP_TRAINED_BLOCKS="${SKIP_TRAINED_BLOCKS-1}"
STOP_ON_FAILURE="${STOP_ON_FAILURE-1}"
MERGE_BLOCKS="${MERGE_BLOCKS-1}"
MERGE_ALLOW_MISSING="${MERGE_ALLOW_MISSING-0}"
MERGE_CFG_ARGS_SOURCE="${MERGE_CFG_ARGS_SOURCE-}"
RENDER_TEST_SET="${RENDER_TEST_SET-1}"
RUN_METRICS="${RUN_METRICS-1}"
RENDER_CUDA_ID="${RENDER_CUDA_ID-}"
CAMERA_LOAD_WORKERS="${CAMERA_LOAD_WORKERS-2}"
DATA_DEVICE="${DATA_DEVICE-cpu}"
LAZY_LOAD_IMAGES="${LAZY_LOAD_IMAGES-0}"
IMAGE_LOAD_MODE="${IMAGE_LOAD_MODE-cache}"
MAX_CACHE_NUM="${MAX_CACHE_NUM-512}"
IMAGE_CACHE_WORKERS="${IMAGE_CACHE_WORKERS-8}"
PARTITION_BBOX_MODE="${PARTITION_BBOX_MODE-expanded}"
PARTITION_INIT_MODE="${PARTITION_INIT_MODE-coarse}"
LOG_ROOT="${LOG_ROOT-${CFG_LOG_ROOT:-$RUN_ROOT/logs_coarse_cache}}"
SWANLAB_PROJECT="${SWANLAB_PROJECT-${CFG_SWANLAB_PROJECT:-block-gs}}"
SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE-${CFG_SWANLAB_WORKSPACE:-}}"
SWANLAB_MODE="${SWANLAB_MODE-${CFG_SWANLAB_MODE:-cloud}}"
SWANLAB_EXP_PREFIX="${SWANLAB_EXP_PREFIX-${CFG_SWANLAB_EXP_PREFIX:-mc_aerial_recursive_depth3-coarse-cache}}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS-}"
EXTRA_MERGE_ARGS="${EXTRA_MERGE_ARGS-}"
EXTRA_RENDER_ARGS="${EXTRA_RENDER_ARGS-}"

IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_IDS// /,}"
if [ "${#GPU_ARRAY[@]}" -eq 0 ]; then
    echo "CUDA_IDS is empty" >&2
    exit 1
fi

if [ -z "$MAX_PARALLEL" ]; then
    MAX_PARALLEL="${#GPU_ARRAY[@]}"
fi

if [ -z "$RENDER_CUDA_ID" ]; then
    RENDER_CUDA_ID="${GPU_ARRAY[0]}"
fi

mkdir -p "$LOG_ROOT"

mapfile -t BLOCK_IDS < <(
    python -c 'import json,sys; data=json.load(open(sys.argv[1])); [print(block["id"]) for block in data.get("blocks", [])]' "$PARTITION_PATH"
)

if [ "${#BLOCK_IDS[@]}" -eq 0 ]; then
    echo "No blocks found in $PARTITION_PATH" >&2
    exit 1
fi

run_block() {
    local block_id="$1"
    local gpu_id="$2"
    local output_path="$OUTPUT_ROOT/$block_id"
    local model_ply="$output_path/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
    local swanlab_name="$SWANLAB_EXP_PREFIX-$block_id"
    local extra_args=()
    local block_test_args=()
    local block_test_data_args=()
    local image_load_args=()
    local depth_args=()

    if [ "$SKIP_TRAINED_BLOCKS" = "1" ] && [ -f "$model_ply" ]; then
        echo "Block $block_id already has $model_ply. Skipping train."
        return
    fi

    if [ -n "$EXTRA_TRAIN_ARGS" ]; then
        # shellcheck disable=SC2206
        extra_args=($EXTRA_TRAIN_ARGS)
    fi

    if [ -n "$DEPTHS" ]; then
        depth_args=(--depths "$DEPTHS")
    fi

    if [ -n "$CONFIG" ]; then
        echo "Training $block_id on GPU $gpu_id with config $CONFIG"
        CUDA_VISIBLE_DEVICES="$gpu_id" python train.py \
            --config "$CONFIG" \
            --block_id "$block_id" \
            "${extra_args[@]}"
        return
    fi

    if [ -n "$BLOCK_TEST_ITERATIONS" ]; then
        # shellcheck disable=SC2206
        block_test_args=(--test_iterations $BLOCK_TEST_ITERATIONS)
    else
        block_test_args=(--test_iterations -1)
    fi

    if [ -n "$TEST_DATA_PATH" ] && [ "$BLOCK_TEST_ITERATIONS" != "-1" ]; then
        block_test_data_args=(
            --test_source_path "$TEST_DATA_PATH"
            --test_images "$TEST_IMAGES"
            --test_depths "$TEST_DEPTHS"
        )
    fi

    if [ "$LAZY_LOAD_IMAGES" = "1" ]; then
        image_load_args=(--image_load_mode dataloader --max_cache_num 0)
    else
        image_load_args=(
            --image_load_mode "$IMAGE_LOAD_MODE"
            --max_cache_num "$MAX_CACHE_NUM"
            --image_cache_workers "$IMAGE_CACHE_WORKERS"
        )
    fi

    echo "Training $block_id on GPU $gpu_id"
    CUDA_VISIBLE_DEVICES="$gpu_id" python train.py \
        -s "$TRAIN_DATA_PATH" \
        --images "$IMAGES" \
        "${depth_args[@]}" \
        -m "$output_path" \
        --iterations "$ITERATIONS" \
        "${block_test_args[@]}" \
        --save_iterations "$ITERATIONS" \
        --checkpoint_iterations "$ITERATIONS" \
        --camera_load_workers "$CAMERA_LOAD_WORKERS" \
        --data_device "$DATA_DEVICE" \
        "${image_load_args[@]}" \
        --partition_path "$PARTITION_PATH" \
        --block_id "$block_id" \
        --partition_bbox_mode "$PARTITION_BBOX_MODE" \
        --partition_init_mode "$PARTITION_INIT_MODE" \
        --swanlab_project "$SWANLAB_PROJECT" \
        --swanlab_workspace "$SWANLAB_WORKSPACE" \
        --swanlab_mode "$SWANLAB_MODE" \
        --swanlab_experiment_name "$swanlab_name" \
        "${block_test_data_args[@]}" \
        "${extra_args[@]}"
}

render_test_split() {
    local render_iteration="$1"
    local test_render_dir="$MERGE_OUTPUT_PATH/test/ours_${render_iteration}/renders"

    if [ -d "$test_render_dir" ]; then
        echo "Merged test render already exists at $test_render_dir. Skipping render."
        return
    fi

    if [ ! -d "$MERGE_OUTPUT_PATH/point_cloud/iteration_${render_iteration}" ]; then
        echo "Merged point cloud not found: $MERGE_OUTPUT_PATH/point_cloud/iteration_${render_iteration}" >&2
        exit 1
    fi
    if [ -n "$CONFIG" ]; then
        local render_extra_args=()
        if [ -n "$EXTRA_RENDER_ARGS" ]; then
            # shellcheck disable=SC2206
            render_extra_args=($EXTRA_RENDER_ARGS)
        fi

        echo "Rendering merged model with config $CONFIG"
        CUDA_VISIBLE_DEVICES="$RENDER_CUDA_ID" python render.py \
            --config "$CONFIG" \
            --override "render.iteration=$render_iteration" \
            "${render_extra_args[@]}"

        if [ -d "$MERGE_OUTPUT_PATH/train/ours_${render_iteration}" ]; then
            mkdir -p "$MERGE_OUTPUT_PATH/test"
            rm -rf "$MERGE_OUTPUT_PATH/test/ours_${render_iteration}"
            mv "$MERGE_OUTPUT_PATH/train/ours_${render_iteration}" "$MERGE_OUTPUT_PATH/test/"
            rmdir "$MERGE_OUTPUT_PATH/train" 2>/dev/null || true
        fi
        if [ ! -d "$test_render_dir" ]; then
            echo "Config render did not produce test renders at $test_render_dir" >&2
            exit 1
        fi
        return
    fi
    if [ ! -f "$MERGE_OUTPUT_PATH/cfg_args" ]; then
        echo "Merged cfg_args not found: $MERGE_OUTPUT_PATH/cfg_args" >&2
        exit 1
    fi

    local tmp_render_root
    tmp_render_root="$(mktemp -d "${MERGE_OUTPUT_PATH}/test_render_tmp.XXXXXX")"

    ln -s "$MERGE_OUTPUT_PATH/point_cloud" "$tmp_render_root/point_cloud"
    cp "$MERGE_OUTPUT_PATH/cfg_args" "$tmp_render_root/cfg_args"

    echo "Rendering merged model on test set from $TEST_DATA_PATH"
    local render_test_data_args=()
    if [ -n "$TEST_DEPTHS" ]; then
        render_test_data_args=(--depths "$TEST_DEPTHS")
    fi

    CUDA_VISIBLE_DEVICES="$RENDER_CUDA_ID" python render.py \
        -m "$tmp_render_root" \
        -s "$TEST_DATA_PATH" \
        --images "$TEST_IMAGES" \
        --iteration "$render_iteration" \
        --camera_load_workers "$CAMERA_LOAD_WORKERS" \
        --data_device "$DATA_DEVICE" \
        "${render_test_data_args[@]}" \
        --skip_test

    mkdir -p "$MERGE_OUTPUT_PATH/test"
    mv "$tmp_render_root/train/ours_${render_iteration}" "$MERGE_OUTPUT_PATH/test/"

    rm "$tmp_render_root/point_cloud"
    rm "$tmp_render_root/cfg_args"
    rmdir "$tmp_render_root/train"
    rmdir "$tmp_render_root"
}

run_test_metrics() {
    local render_iteration="$1"
    local test_render_dir="$MERGE_OUTPUT_PATH/test/ours_${render_iteration}/renders"

    if [ ! -d "$test_render_dir" ]; then
        echo "Cannot run metrics; missing test renders: $test_render_dir" >&2
        exit 1
    fi

    echo "Evaluating merged test renders under $MERGE_OUTPUT_PATH"
    if [ -n "$CONFIG" ]; then
        CUDA_VISIBLE_DEVICES="$RENDER_CUDA_ID" python metrics.py --config "$CONFIG"
    else
        CUDA_VISIBLE_DEVICES="$RENDER_CUDA_ID" python metrics.py -m "$MERGE_OUTPUT_PATH"
    fi
    cp "$MERGE_OUTPUT_PATH/results.json" "$MERGE_OUTPUT_PATH/results_test.json"
    cp "$MERGE_OUTPUT_PATH/per_view.json" "$MERGE_OUTPUT_PATH/per_view_test.json"
}

if [ "$TRAIN_BLOCKS" = "1" ]; then
    failed_jobs=0
    block_index=0
    stop_launching=0
    job_ids=()
    job_pids=()
    job_gpus=()
    job_blocks=()
    JOB_STATE_DIR="$(mktemp -d "$LOG_ROOT/job_state.XXXXXX")"

    cleanup_job_state() {
        rm -rf "$JOB_STATE_DIR"
    }
    terminate_jobs() {
        local pid
        for pid in "${job_pids[@]}"; do
            kill -TERM "$pid" 2>/dev/null || true
        done
    }
    trap cleanup_job_state EXIT
    trap 'terminate_jobs; exit 130' INT
    trap 'terminate_jobs; exit 143' TERM

    launch_block_job() {
        local block_id="$1"
        local gpu_id="$2"
        local job_id="${block_id}_${gpu_id}_$RANDOM"
        local status_file="$JOB_STATE_DIR/${job_id}.status"

        echo "Launching $block_id on GPU $gpu_id. Log: $LOG_ROOT/${block_id}.log"
        (
            set +e
            run_block "$block_id" "$gpu_id"
            status=$?
            printf '%s %s %s\n' "$gpu_id" "$block_id" "$status" > "$status_file"
            exit "$status"
        ) > "$LOG_ROOT/${block_id}.log" 2>&1 &

        job_ids+=("$job_id")
        job_pids+=("$!")
        job_gpus+=("$gpu_id")
        job_blocks+=("$block_id")
    }

    wait_for_one() {
        local status_file
        local job_id
        local finished_gpu
        local finished_block
        local status
        local index
        local wait_status

        while true; do
            for status_file in "$JOB_STATE_DIR"/*.status; do
                [ -e "$status_file" ] || continue
                job_id="$(basename "$status_file" .status)"
                index=-1
                for i in "${!job_ids[@]}"; do
                    if [ "${job_ids[$i]}" = "$job_id" ]; then
                        index="$i"
                        break
                    fi
                done
                if [ "$index" -lt 0 ]; then
                    rm -f "$status_file"
                    continue
                fi

                read -r finished_gpu finished_block status < "$status_file"
                rm -f "$status_file"

                set +e
                wait "${job_pids[$index]}"
                wait_status=$?
                set -e
                if [ -z "${status:-}" ]; then
                    status="$wait_status"
                fi

                FREED_GPU="$finished_gpu"
                FINISHED_BLOCK="$finished_block"
                FINISHED_STATUS="$status"

                unset 'job_ids[index]'
                unset 'job_pids[index]'
                unset 'job_gpus[index]'
                unset 'job_blocks[index]'
                job_ids=("${job_ids[@]}")
                job_pids=("${job_pids[@]}")
                job_gpus=("${job_gpus[@]}")
                job_blocks=("${job_blocks[@]}")

                if [ "$status" -ne 0 ]; then
                    failed_jobs=$((failed_jobs + 1))
                    echo "Block $finished_block failed on GPU $finished_gpu with status $status. Check $LOG_ROOT/${finished_block}.log" >&2
                    if [ "$STOP_ON_FAILURE" = "1" ]; then
                        stop_launching=1
                    fi
                fi
                return
            done

            set +e
            wait -n
            wait_status=$?
            set -e

            index=-1
            for i in "${!job_pids[@]}"; do
                if ! kill -0 "${job_pids[$i]}" 2>/dev/null; then
                    index="$i"
                    break
                fi
            done
            if [ "$index" -lt 0 ]; then
                sleep 2
                continue
            fi

            FREED_GPU="${job_gpus[$index]}"
            FINISHED_BLOCK="${job_blocks[$index]}"
            FINISHED_STATUS="$wait_status"

            unset 'job_ids[index]'
            unset 'job_pids[index]'
            unset 'job_gpus[index]'
            unset 'job_blocks[index]'
            job_ids=("${job_ids[@]}")
            job_pids=("${job_pids[@]}")
            job_gpus=("${job_gpus[@]}")
            job_blocks=("${job_blocks[@]}")

            if [ "$FINISHED_STATUS" -ne 0 ]; then
                failed_jobs=$((failed_jobs + 1))
                echo "Block $FINISHED_BLOCK failed on GPU $FREED_GPU with status $FINISHED_STATUS. Check $LOG_ROOT/${FINISHED_BLOCK}.log" >&2
                if [ "$STOP_ON_FAILURE" = "1" ]; then
                    stop_launching=1
                fi
            fi
            return
        done
    }

    while [ "$block_index" -lt "${#BLOCK_IDS[@]}" ] && [ "${#job_ids[@]}" -lt "$MAX_PARALLEL" ]; do
        gpu_id="${GPU_ARRAY[$((block_index % ${#GPU_ARRAY[@]}))]}"
        launch_block_job "${BLOCK_IDS[$block_index]}" "$gpu_id"
        block_index=$((block_index + 1))
    done

    while [ "${#job_ids[@]}" -gt 0 ]; do
        wait_for_one
        if [ "$stop_launching" = "0" ] && [ "$block_index" -lt "${#BLOCK_IDS[@]}" ]; then
            launch_block_job "${BLOCK_IDS[$block_index]}" "$FREED_GPU"
            block_index=$((block_index + 1))
        fi
    done

    if [ "$failed_jobs" -ne 0 ]; then
        echo "$failed_jobs block training job(s) failed. Not running merge/render/eval." >&2
        exit 1
    fi
    echo "All block training jobs finished."
else
    echo "TRAIN_BLOCKS=0, skipping block training."
fi

if [ "$MERGE_BLOCKS" = "1" ]; then
    merge_args=()
    if [ "$MERGE_ALLOW_MISSING" = "1" ]; then
        if [ -n "$CONFIG" ]; then
            merge_args+=(--override merge.allow_missing=true)
        else
            merge_args+=(--allow_missing)
        fi
    fi
    if [ -n "$EXTRA_MERGE_ARGS" ]; then
        # shellcheck disable=SC2206
        merge_args+=($EXTRA_MERGE_ARGS)
    fi
    if [ -n "$MERGE_CFG_ARGS_SOURCE" ]; then
        if [ -n "$CONFIG" ]; then
            merge_args+=(--override "merge.cfg_args_source=$MERGE_CFG_ARGS_SOURCE")
        else
            merge_args+=(--cfg_args_source "$MERGE_CFG_ARGS_SOURCE")
        fi
    fi

    echo "Merging block outputs to $MERGE_OUTPUT_PATH"
    if [ -n "$CONFIG" ]; then
        python merge_blocks.py --config "$CONFIG" "${merge_args[@]}"
    else
        python merge_blocks.py \
            --partition_path "$PARTITION_PATH" \
            --blocks_root "$OUTPUT_ROOT" \
            --iteration "$MERGE_ITERATION" \
            --output_path "$MERGE_OUTPUT_PATH" \
            "${merge_args[@]}"
    fi
    echo "Merged model written under $MERGE_OUTPUT_PATH"
fi

if [ "$RENDER_TEST_SET" = "1" ]; then
    render_test_split "$MERGE_ITERATION"
fi

if [ "$RUN_METRICS" = "1" ]; then
    run_test_metrics "$MERGE_ITERATION"
fi
