#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.camera_utils import get_camera_resolution


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_image_paths(source_path, images):
    image_root = Path(source_path) / images
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_root}")
    paths = [path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
    if not paths:
        raise RuntimeError(f"No images found under {image_root}")
    return image_root, sorted(paths)


def load_uint8_chw(path, resolution):
    with Image.open(path) as image:
        has_alpha = image.mode in {"RGBA", "LA"} or "transparency" in image.info
        image = image.convert("RGBA" if has_alpha else "RGB")
        image = image.resize(resolution)
        array = np.asarray(image, dtype=np.uint8)

    if array.shape[-1] == 4:
        rgb = array[..., :3]
        alpha = array[..., 3:4]
    else:
        rgb = array
        alpha = None
    rgb = np.transpose(rgb, (2, 0, 1)).copy()
    if alpha is not None:
        alpha = np.transpose(alpha, (2, 0, 1)).copy()
    return rgb, alpha


def image_record(args, image_root, image_path):
    with Image.open(image_path) as image:
        resolution = get_camera_resolution(args, image.width, image.height, 1.0)
        has_alpha = image.mode in {"RGBA", "LA"} or "transparency" in image.info
    width, height = resolution
    rel_name = image_path.relative_to(image_root).as_posix()
    return {
        "name": rel_name,
        "path": image_path,
        "resolution": resolution,
        "image_shape": [3, height, width],
        "alpha_shape": [1, height, width] if has_alpha else None,
    }


def build_cache(args):
    image_root, image_paths = find_image_paths(args.source_path, args.images)
    os.makedirs(args.output, exist_ok=True)

    records = [
        image_record(args, image_root, image_path)
        for image_path in tqdm(image_paths, desc="Scanning images")
    ]

    image_offset = 0
    alpha_offset = 0
    items = {}
    for record in records:
        image_size = int(np.prod(record["image_shape"]))
        alpha_shape = record["alpha_shape"]
        alpha_size = int(np.prod(alpha_shape)) if alpha_shape is not None else 0
        items[record["name"]] = {
            "image_offset": image_offset,
            "image_shape": record["image_shape"],
            "alpha_offset": alpha_offset if alpha_shape is not None else None,
            "alpha_shape": alpha_shape,
        }
        image_offset += image_size
        alpha_offset += alpha_size

    image_file = "images.uint8.bin"
    alpha_file = "alpha.uint8.bin" if alpha_offset > 0 else None
    image_mm = np.memmap(os.path.join(args.output, image_file), dtype=np.uint8, mode="w+", shape=(image_offset,))
    alpha_mm = None
    if alpha_file:
        alpha_mm = np.memmap(os.path.join(args.output, alpha_file), dtype=np.uint8, mode="w+", shape=(alpha_offset,))

    for record in tqdm(records, desc="Writing mmap cache"):
        item = items[record["name"]]
        rgb, alpha = load_uint8_chw(record["path"], tuple(record["resolution"]))

        start = item["image_offset"]
        end = start + rgb.size
        image_mm[start:end] = rgb.reshape(-1)

        if alpha is not None and alpha_mm is not None:
            start = item["alpha_offset"]
            end = start + alpha.size
            alpha_mm[start:end] = alpha.reshape(-1)

    image_mm.flush()
    if alpha_mm is not None:
        alpha_mm.flush()

    manifest = {
        "version": 1,
        "source_path": os.path.abspath(args.source_path),
        "images": args.images,
        "resolution": int(args.resolution),
        "dtype": "uint8",
        "layout": "CHW",
        "image_file": image_file,
        "alpha_file": alpha_file,
        "num_images": len(records),
        "image_bytes": int(image_offset),
        "alpha_bytes": int(alpha_offset),
        "items": items,
    }
    with open(os.path.join(args.output, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        "Wrote shared image mmap cache: "
        f"images={len(records)} image_bytes={image_offset} "
        f"alpha_bytes={alpha_offset} output={args.output}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Build a shared mmap cache for resized dataset images")
    parser.add_argument("--source_path", "-s", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--resolution", "-r", type=int, default=-1)
    parser.add_argument("--output", "-o", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build_cache(parse_args())
