#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/prepare_depth_regularization.sh DATASET_ROOT [IMAGE_DIR_NAME]

Prepare Depth Anything V2 priors for COLMAP-format 3DGS training.

Arguments:
  DATASET_ROOT      COLMAP scene root containing images/ and sparse/0/.
  IMAGE_DIR_NAME   Image directory under DATASET_ROOT. Default: images

Environment overrides:
  DA_ROOT          Depth-Anything-V2 repo. Default: /lfs1/users/spsong/Code/Depth-Anything-V2
  DA_PYTHON        Python for Depth Anything. Default: $DA_ROOT/.venv/bin/python if present, else python
  GS_PYTHON        Python for this 3DGS repo utilities. Default: python
  DEPTH_DIR_NAME   Output depth directory under DATASET_ROOT. Default: depths_any
  ENCODER          Depth Anything encoder: vits, vitb, vitl, vitg. Default: vitl
  INPUT_SIZE       Depth Anything inference input size. Default: 518
  MODEL_TYPE       COLMAP model type: auto, bin, txt. Default: auto
  FORCE_DEPTH      Regenerate existing depth PNGs when 1. Default: 0
  RUN_DEPTH        Run Depth Anything inference when 1. Default: 1
  RUN_ALIGN        Run SfM inverse-depth alignment when 1. Default: 1
  CHECK_ONLY       Only validate paths and print resolved settings. Default: 0

Example:
  bash scripts/prepare_depth_regularization.sh /lfs3/users/spsong/dataset/tandt/train
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    usage >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="$(cd "$1" && pwd)"
IMAGE_DIR_NAME="${2:-${IMAGE_DIR_NAME:-images}}"
IMAGES_DIR="$DATASET_ROOT/$IMAGE_DIR_NAME"
SPARSE_DIR="$DATASET_ROOT/sparse/0"

DA_ROOT="${DA_ROOT:-/lfs1/users/spsong/Code/Depth-Anything-V2}"
if [ -z "${DA_PYTHON:-}" ]; then
    if [ -x "$DA_ROOT/.venv/bin/python" ]; then
        DA_PYTHON="$DA_ROOT/.venv/bin/python"
    else
        DA_PYTHON="python"
    fi
fi
GS_PYTHON="${GS_PYTHON:-python}"

DEPTH_DIR_NAME="${DEPTH_DIR_NAME:-depths_any}"
DEPTHS_DIR="$DATASET_ROOT/$DEPTH_DIR_NAME"
ENCODER="${ENCODER:-vitl}"
INPUT_SIZE="${INPUT_SIZE:-518}"
MODEL_TYPE="${MODEL_TYPE:-auto}"
FORCE_DEPTH="${FORCE_DEPTH:-0}"
RUN_DEPTH="${RUN_DEPTH:-1}"
RUN_ALIGN="${RUN_ALIGN:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"

if [ ! -d "$IMAGES_DIR" ]; then
    echo "Image directory not found: $IMAGES_DIR" >&2
    exit 1
fi

if [ ! -d "$SPARSE_DIR" ]; then
    echo "COLMAP sparse directory not found: $SPARSE_DIR" >&2
    exit 1
fi

if [ ! -d "$DA_ROOT" ]; then
    echo "Depth-Anything-V2 repo not found: $DA_ROOT" >&2
    exit 1
fi

if [ "$MODEL_TYPE" = "auto" ]; then
    if [ -f "$SPARSE_DIR/cameras.bin" ] && [ -f "$SPARSE_DIR/images.bin" ] && [ -f "$SPARSE_DIR/points3D.bin" ]; then
        MODEL_TYPE="bin"
    elif [ -f "$SPARSE_DIR/cameras.txt" ] && [ -f "$SPARSE_DIR/images.txt" ] && [ -f "$SPARSE_DIR/points3D.txt" ]; then
        MODEL_TYPE="txt"
    else
        echo "Could not detect COLMAP model type in $SPARSE_DIR" >&2
        echo "Expected cameras/images/points3D in either .bin or .txt format." >&2
        exit 1
    fi
fi

case "$MODEL_TYPE" in
    bin)
        required=(cameras.bin images.bin points3D.bin)
        ;;
    txt)
        required=(cameras.txt images.txt points3D.txt)
        ;;
    *)
        echo "MODEL_TYPE must be auto, bin, or txt; got: $MODEL_TYPE" >&2
        exit 1
        ;;
esac

for name in "${required[@]}"; do
    if [ ! -f "$SPARSE_DIR/$name" ]; then
        echo "Missing COLMAP file: $SPARSE_DIR/$name" >&2
        exit 1
    fi
done

CHECKPOINT="$DA_ROOT/checkpoints/depth_anything_v2_${ENCODER}.pth"
if [ ! -f "$CHECKPOINT" ]; then
    echo "Depth Anything checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

echo "Preparing depth regularization data"
echo "  dataset:      $DATASET_ROOT"
echo "  images:       $IMAGES_DIR"
echo "  depths:       $DEPTHS_DIR"
echo "  colmap type:  $MODEL_TYPE"
echo "  DA root:      $DA_ROOT"
echo "  DA python:    $DA_PYTHON"
echo "  GS python:    $GS_PYTHON"
echo "  encoder:      $ENCODER"
echo "  input size:   $INPUT_SIZE"

if [ "$CHECK_ONLY" = "1" ]; then
    echo "CHECK_ONLY=1, preflight checks passed; no files were generated."
    exit 0
fi

mkdir -p "$DEPTHS_DIR"

if [ "$RUN_DEPTH" = "1" ]; then
    "$DA_PYTHON" - "$DA_ROOT" "$IMAGES_DIR" "$DEPTHS_DIR" "$ENCODER" "$INPUT_SIZE" "$FORCE_DEPTH" <<'PY'
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

da_root = Path(sys.argv[1]).resolve()
images_dir = Path(sys.argv[2]).resolve()
depths_dir = Path(sys.argv[3]).resolve()
encoder = sys.argv[4]
input_size = int(sys.argv[5])
force_depth = sys.argv[6] == "1"

sys.path.insert(0, str(da_root))
from depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}
if encoder not in model_configs:
    raise SystemExit(f"Unsupported encoder: {encoder}")

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
image_paths = sorted(
    path for path in images_dir.rglob("*")
    if path.is_file() and path.suffix.lower() in extensions
)
if not image_paths:
    raise SystemExit(f"No image files found under {images_dir}")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
checkpoint = da_root / "checkpoints" / f"depth_anything_v2_{encoder}.pth"

print(f"[DepthAnything] images={len(image_paths)} device={device} checkpoint={checkpoint}")
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
model = model.to(device).eval()

written = 0
skipped = 0
with torch.inference_mode():
    for idx, image_path in enumerate(image_paths, start=1):
        rel_path = image_path.relative_to(images_dir)
        out_path = (depths_dir / rel_path).with_suffix(".png")
        if out_path.exists() and not force_depth:
            skipped += 1
            if idx == 1 or idx == len(image_paths) or idx % 25 == 0:
                print(f"[DepthAnything] {idx}/{len(image_paths)} skip existing {rel_path}")
            continue

        raw_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if raw_image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        depth = model.infer_image(raw_image, input_size).astype(np.float32)
        min_value = float(np.nanmin(depth))
        max_value = float(np.nanmax(depth))
        if not np.isfinite(min_value) or not np.isfinite(max_value) or max_value - min_value < 1e-8:
            depth_u16 = np.zeros(depth.shape, dtype=np.uint16)
        else:
            depth_norm = (depth - min_value) / (max_value - min_value)
            depth_u16 = np.clip(depth_norm * 65535.0, 0.0, 65535.0).astype(np.uint16)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), depth_u16):
            raise RuntimeError(f"Failed to write depth: {out_path}")
        written += 1

        if idx == 1 or idx == len(image_paths) or idx % 10 == 0:
            print(f"[DepthAnything] {idx}/{len(image_paths)} wrote {out_path.relative_to(depths_dir)}")

print(f"[DepthAnything] done written={written} skipped={skipped} outdir={depths_dir}")
PY
else
    echo "RUN_DEPTH=0, skipping Depth Anything inference."
fi

if [ "$RUN_ALIGN" = "1" ]; then
    echo "Generating COLMAP inverse-depth alignment parameters"
    (
        cd "$REPO_ROOT"
        "$GS_PYTHON" utils/make_depth_scale.py \
            --base_dir "$DATASET_ROOT" \
            --depths_dir "$DEPTHS_DIR" \
            --model_type "$MODEL_TYPE"
    )
else
    echo "RUN_ALIGN=0, skipping depth_params.json generation."
fi

DEPTH_PARAMS="$SPARSE_DIR/depth_params.json"
if [ ! -f "$DEPTH_PARAMS" ]; then
    echo "Missing expected output: $DEPTH_PARAMS" >&2
    exit 1
fi

"$GS_PYTHON" - "$REPO_ROOT" "$DATASET_ROOT" "$IMAGE_DIR_NAME" "$DEPTH_DIR_NAME" "$MODEL_TYPE" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
dataset_root = Path(sys.argv[2]).resolve()
image_dir_name = sys.argv[3]
depth_dir_name = sys.argv[4]
model_type = sys.argv[5]

sys.path.insert(0, str(repo_root / "utils"))
from read_write_model import read_model

sparse_dir = dataset_root / "sparse" / "0"
_, images, _ = read_model(str(sparse_dir), ext=f".{model_type}")
depths_dir = dataset_root / depth_dir_name
params_path = sparse_dir / "depth_params.json"
with params_path.open("r") as f:
    params = json.load(f)

missing_depths = []
missing_params = []
for image in images.values():
    image_name = image.name
    stem = image_name.rsplit(".", 1)[0]
    if not (depths_dir / f"{stem}.png").is_file():
        missing_depths.append(f"{stem}.png")
    if stem not in params:
        missing_params.append(stem)

positive_scales = sum(1 for value in params.values() if value.get("scale", 0) > 0)
zero_scales = len(params) - positive_scales

print("Validation")
print(f"  COLMAP images:      {len(images)}")
print(f"  depth params:       {len(params)}")
print(f"  positive scales:    {positive_scales}")
print(f"  zero scales:        {zero_scales}")
print(f"  missing depths:     {len(missing_depths)}")
print(f"  missing params:     {len(missing_params)}")

if missing_depths:
    print("First missing depth files:")
    for name in missing_depths[:10]:
        print(f"  {name}")
if missing_params:
    print("First missing depth_params keys:")
    for name in missing_params[:10]:
        print(f"  {name}")

if missing_depths or missing_params or not params:
    raise SystemExit(1)

if positive_scales == 0:
    print("Warning: no positive depth alignment scales were found.")
PY

echo "Depth regularization data is ready."
echo
echo "Training command:"
echo "  python train.py \\"
echo "    -s \"$DATASET_ROOT\" \\"
echo "    -i \"$IMAGE_DIR_NAME\" \\"
echo "    -d \"$DEPTH_DIR_NAME\" \\"
echo "    -m output/$(basename "$DATASET_ROOT")_depth \\"
echo "    --data_device cpu"
