#!/usr/bin/env bash
set -euo pipefail

# Required:
#   DATASET_ROOT=/path/to/3dgs/dataset bash scripts/prepare_normal_regularization.sh
#
# Optional:
#   IMAGES=input
#   NORMAL_DIR_NAME=normals_dsine
#   NORMAL_VIS_DIR_NAME=normals_dsine_vis
#   DSINE_ROOT=/lfs1/users/spsong/Code/project/DSINE
#   DSINE_ENV=DSINE
#   GPU=0
#   BATCH_SIZE=1
#   LIMIT=0
#   FORCE_NORMAL=1
#   NO_VIS=1

DATASET_ROOT="${DATASET_ROOT:-}"
if [ -z "$DATASET_ROOT" ]; then
    echo "DATASET_ROOT is required" >&2
    exit 1
fi

IMAGES="${IMAGES:-input}"
NORMAL_DIR_NAME="${NORMAL_DIR_NAME:-normals_dsine}"
NORMAL_VIS_DIR_NAME="${NORMAL_VIS_DIR_NAME:-normals_dsine_vis}"
DSINE_ROOT="${DSINE_ROOT:-/lfs1/users/spsong/Code/project/DSINE}"
DSINE_ENV="${DSINE_ENV:-DSINE}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LIMIT="${LIMIT:-0}"
FORCE_NORMAL="${FORCE_NORMAL:-0}"
NO_VIS="${NO_VIS:-0}"

DATASET_ROOT="$(realpath "$DATASET_ROOT")"
IMAGES_DIR="$DATASET_ROOT/$IMAGES"
NORMALS_DIR="$DATASET_ROOT/$NORMAL_DIR_NAME"
VIS_DIR="$DATASET_ROOT/$NORMAL_VIS_DIR_NAME"
DSINE_TOOL="$DSINE_ROOT/tools/generate_normals_for_gs.py"

if [ ! -d "$IMAGES_DIR" ]; then
    echo "Image directory not found: $IMAGES_DIR" >&2
    exit 1
fi
if [ ! -f "$DSINE_TOOL" ]; then
    echo "DSINE normal generation tool not found: $DSINE_TOOL" >&2
    exit 1
fi

args=(
    "$DSINE_TOOL"
    --source "$DATASET_ROOT"
    --images "$IMAGES"
    --output "$NORMAL_DIR_NAME"
    --vis-output "$NORMAL_VIS_DIR_NAME"
    --batch-size "$BATCH_SIZE"
)

if [ "$LIMIT" != "0" ]; then
    args+=(--limit "$LIMIT")
fi
if [ "$FORCE_NORMAL" = "1" ]; then
    args+=(--overwrite)
fi
if [ "$NO_VIS" = "1" ]; then
    args+=(--no-vis)
fi

echo "Generating DSINE normals"
echo "  dataset: $DATASET_ROOT"
echo "  images:  $IMAGES_DIR"
echo "  output:  $NORMALS_DIR"
if [ "$NO_VIS" != "1" ]; then
    echo "  vis:     $VIS_DIR"
fi

CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$DSINE_ENV" python "${args[@]}"

python - "$IMAGES_DIR" "$NORMALS_DIR" <<'PY'
import sys
from pathlib import Path

image_root = Path(sys.argv[1])
normal_root = Path(sys.argv[2])
suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
images = sorted(p for p in image_root.rglob("*") if p.suffix.lower() in suffixes)
missing = [p.relative_to(image_root).with_suffix(".npy") for p in images if not (normal_root / p.relative_to(image_root).with_suffix(".npy")).is_file()]
print(f"Normal generation check: images={len(images)} missing_normals={len(missing)}")
if missing:
    for rel in missing[:10]:
        print(f"  missing: {rel}")
    raise SystemExit(1)
PY

echo "Normal regularization inputs ready: $NORMALS_DIR"
