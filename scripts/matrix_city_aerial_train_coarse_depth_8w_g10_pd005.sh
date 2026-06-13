#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-config/mc_aerial_coarse_depth_8w_g10_pd005.yaml}"
export CONFIG

bash scripts/matrix_city_aerial_train_coarse_depth_8w_densify32k.sh
