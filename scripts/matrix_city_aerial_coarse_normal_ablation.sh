#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
PARALLEL="${PARALLEL:-1}"
W10X_CUDA_ID="${W10X_CUDA_ID:-0}"
DELAY5K_CUDA_ID="${DELAY5K_CUDA_ID:-1}"

COMMON_SCRIPT="scripts/matrix_city_aerial_coarse_normal.sh"

run_experiment() {
    local name="$1"
    local config="$2"
    local cuda_id="$3"
    local log_dir="output/${name}"
    local log_path="${log_dir}/pipeline.log"

    mkdir -p "$log_dir"
    echo "[$name] config=$config cuda=$cuda_id log=$log_path"
    env \
        CONFIG="$config" \
        CUDA_ID="$cuda_id" \
        RENDER_CUDA_ID="$cuda_id" \
        METRICS_CUDA_ID="$cuda_id" \
        PYTHON_BIN="$PYTHON_BIN" \
        bash "$COMMON_SCRIPT" 2>&1 | tee "$log_path"
}

if [ "$PARALLEL" = "1" ]; then
    run_experiment \
        mc_aerial_coarse_normal_w10x_30000 \
        config/mc_aerial_coarse_normal_w10x.yaml \
        "$W10X_CUDA_ID" &
    pid_w10x=$!

    run_experiment \
        mc_aerial_coarse_normal_delay5k_30000 \
        config/mc_aerial_coarse_normal_delay5k.yaml \
        "$DELAY5K_CUDA_ID" &
    pid_delay5k=$!

    wait "$pid_w10x"
    wait "$pid_delay5k"
else
    run_experiment \
        mc_aerial_coarse_normal_w10x_30000 \
        config/mc_aerial_coarse_normal_w10x.yaml \
        "$W10X_CUDA_ID"
    run_experiment \
        mc_aerial_coarse_normal_delay5k_30000 \
        config/mc_aerial_coarse_normal_delay5k.yaml \
        "$DELAY5K_CUDA_ID"
fi
