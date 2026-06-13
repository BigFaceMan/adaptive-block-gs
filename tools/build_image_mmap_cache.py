#!/usr/bin/env python
import argparse
import cv2
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


def load_depth_params(source_path, depths):
    if not depths:
        return None
    depth_params_path = Path(source_path) / "sparse" / "0" / "depth_params.json"
    with open(depth_params_path, "r") as f:
        depth_params = json.load(f)
    scales = np.array([value["scale"] for value in depth_params.values()], dtype=np.float32)
    positive_scales = scales[scales > 0]
    med_scale = float(np.median(positive_scales)) if positive_scales.size else 0.0
    for value in depth_params.values():
        value["med_scale"] = med_scale
    return depth_params


def read_invdepth(depth_path, is_nerf_synthetic):
    raw = cv2.imread(str(depth_path), -1)
    if raw is None:
        raise RuntimeError(f"Failed to read depth map: {depth_path}")
    if is_nerf_synthetic:
        return raw.astype(np.float32) / 512
    return raw.astype(np.float32) / float(2**16)


def load_depth_chw(depth_path, resolution, depth_params):
    raw_invdepthmap = read_invdepth(depth_path, False)
    raw_invdepthmap = cv2.resize(raw_invdepthmap, resolution)
    raw_invdepthmap[raw_invdepthmap < 0] = 0
    reliable = True

    if depth_params is not None:
        if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
            reliable = False
        if depth_params["scale"] > 0:
            raw_invdepthmap = raw_invdepthmap * depth_params["scale"] + depth_params["offset"]

    if raw_invdepthmap.ndim != 2:
        raw_invdepthmap = raw_invdepthmap[..., 0]
    return raw_invdepthmap[None].astype(np.float32, copy=True), reliable


def depth_path_for(image_root, image_path, depths_root):
    rel_path = image_path.relative_to(image_root)
    return depths_root / rel_path.with_suffix(".png")


def normal_path_for(image_root, image_path, normals_root):
    rel_path = image_path.relative_to(image_root)
    npy_path = normals_root / rel_path.with_suffix(".npy")
    if npy_path.exists():
        return npy_path
    for suffix in (".png", ".jpg", ".jpeg"):
        image_normal_path = normals_root / rel_path.with_suffix(suffix)
        if image_normal_path.exists():
            return image_normal_path
    raise FileNotFoundError(f"Normal map not found for image: {image_path}")


def normal_to_chw(array, encoded_image=False):
    normal = np.asarray(array)
    if normal.ndim != 3:
        raise ValueError(f"Expected normal map with 3 dimensions, got shape={normal.shape}")

    if normal.shape[0] == 3 and normal.shape[-1] != 3:
        normal = normal.astype(np.float32, copy=False)
    elif normal.shape[-1] >= 3:
        normal = np.transpose(normal[..., :3], (2, 0, 1)).astype(np.float32, copy=False)
    else:
        raise ValueError(f"Expected normal map with 3 channels, got shape={normal.shape}")

    finite = np.isfinite(normal)
    normal = np.where(finite, normal, 0.0).astype(np.float32, copy=False)
    if encoded_image and normal.size and normal.max() > 2.0:
        normal = normal / 255.0 * 2.0 - 1.0
    elif encoded_image and normal.size and normal.min() >= 0.0 and normal.max() <= 1.0:
        normal = normal * 2.0 - 1.0
    return normal


def normalize_normal_chw(normal):
    norm = np.linalg.norm(normal, axis=0, keepdims=True)
    valid = np.isfinite(norm) & (norm > 1e-6)
    normal = np.divide(normal, np.maximum(norm, 1e-6), where=valid, out=np.zeros_like(normal, dtype=np.float32))
    return normal.astype(np.float32, copy=False), bool(valid.any())


def load_normal_chw(normal_path, resolution):
    if str(normal_path).lower().endswith(".npy"):
        normal = normal_to_chw(np.load(normal_path))
    else:
        with Image.open(normal_path) as image:
            normal = normal_to_chw(np.asarray(image.convert("RGB")), encoded_image=True)

    if normal.shape[-2:] != (resolution[1], resolution[0]):
        normal_hwc = np.transpose(normal, (1, 2, 0))
        normal_hwc = cv2.resize(normal_hwc, resolution, interpolation=cv2.INTER_LINEAR)
        normal = np.transpose(normal_hwc, (2, 0, 1))

    return normalize_normal_chw(normal)


def depth_params_for(depth_params, image_name):
    if depth_params is None:
        return None
    stem_name = str(Path(image_name).with_suffix("")).replace("\\", "/")
    return depth_params.get(stem_name)


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
    depths_root = Path(args.source_path) / args.depths if args.depths else None
    normals_root = Path(args.source_path) / args.normals if args.normals else None
    if normals_root is not None and not normals_root.is_dir():
        raise FileNotFoundError(f"Normal directory not found: {normals_root}")
    depth_params = load_depth_params(args.source_path, args.depths)
    os.makedirs(args.output, exist_ok=True)

    records = [
        image_record(args, image_root, image_path)
        for image_path in tqdm(image_paths, desc="Scanning images")
    ]

    image_offset = 0
    alpha_offset = 0
    depth_offset = 0
    normal_offset = 0
    items = {}
    for record in records:
        image_size = int(np.prod(record["image_shape"]))
        alpha_shape = record["alpha_shape"]
        alpha_size = int(np.prod(alpha_shape)) if alpha_shape is not None else 0
        depth_shape = [1, record["image_shape"][1], record["image_shape"][2]] if depths_root is not None else None
        depth_size = int(np.prod(depth_shape)) if depth_shape is not None else 0
        normal_shape = [3, record["image_shape"][1], record["image_shape"][2]] if normals_root is not None else None
        normal_size = int(np.prod(normal_shape)) if normal_shape is not None else 0
        items[record["name"]] = {
            "image_offset": image_offset,
            "image_shape": record["image_shape"],
            "alpha_offset": alpha_offset if alpha_shape is not None else None,
            "alpha_shape": alpha_shape,
            "depth_offset": depth_offset if depth_shape is not None else None,
            "depth_shape": depth_shape,
            "depth_reliable": False,
            "normal_offset": normal_offset if normal_shape is not None else None,
            "normal_shape": normal_shape,
            "normal_reliable": False,
        }
        image_offset += image_size
        alpha_offset += alpha_size
        depth_offset += depth_size
        normal_offset += normal_size

    image_file = "images.uint8.bin"
    alpha_file = "alpha.uint8.bin" if alpha_offset > 0 else None
    depth_file = "depths.float32.bin" if depth_offset > 0 else None
    normal_file = "normals.float32.bin" if normal_offset > 0 else None
    image_mm = np.memmap(os.path.join(args.output, image_file), dtype=np.uint8, mode="w+", shape=(image_offset,))
    alpha_mm = None
    if alpha_file:
        alpha_mm = np.memmap(os.path.join(args.output, alpha_file), dtype=np.uint8, mode="w+", shape=(alpha_offset,))
    depth_mm = None
    if depth_file:
        depth_mm = np.memmap(os.path.join(args.output, depth_file), dtype=np.float32, mode="w+", shape=(depth_offset,))
    normal_mm = None
    if normal_file:
        normal_mm = np.memmap(os.path.join(args.output, normal_file), dtype=np.float32, mode="w+", shape=(normal_offset,))

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

        if depth_mm is not None:
            depth_path = depth_path_for(image_root, record["path"], depths_root)
            depth, reliable = load_depth_chw(
                depth_path,
                tuple(record["resolution"]),
                depth_params_for(depth_params, record["name"]),
            )
            start = item["depth_offset"]
            end = start + depth.size
            depth_mm[start:end] = depth.reshape(-1)
            item["depth_reliable"] = bool(reliable)

        if normal_mm is not None:
            normal_path = normal_path_for(image_root, record["path"], normals_root)
            normal, reliable = load_normal_chw(normal_path, tuple(record["resolution"]))
            start = item["normal_offset"]
            end = start + normal.size
            normal_mm[start:end] = normal.reshape(-1)
            item["normal_reliable"] = bool(reliable)

    image_mm.flush()
    if alpha_mm is not None:
        alpha_mm.flush()
    if depth_mm is not None:
        depth_mm.flush()
    if normal_mm is not None:
        normal_mm.flush()

    manifest = {
        "version": 1,
        "source_path": os.path.abspath(args.source_path),
        "images": args.images,
        "depths": args.depths,
        "normals": args.normals,
        "resolution": int(args.resolution),
        "dtype": "uint8",
        "layout": "CHW",
        "image_file": image_file,
        "alpha_file": alpha_file,
        "depth_file": depth_file,
        "normal_file": normal_file,
        "num_images": len(records),
        "image_bytes": int(image_offset),
        "alpha_bytes": int(alpha_offset),
        "depth_bytes": int(depth_offset * np.dtype(np.float32).itemsize),
        "normal_bytes": int(normal_offset * np.dtype(np.float32).itemsize),
        "items": items,
    }
    with open(os.path.join(args.output, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        "Wrote shared image mmap cache: "
        f"images={len(records)} image_bytes={image_offset} "
        f"alpha_bytes={alpha_offset} depth_bytes={depth_offset * np.dtype(np.float32).itemsize} "
        f"normal_bytes={normal_offset * np.dtype(np.float32).itemsize} "
        f"output={args.output}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Build a shared mmap cache for resized dataset images")
    parser.add_argument("--source_path", "-s", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--depths", default="")
    parser.add_argument("--normals", default="")
    parser.add_argument("--resolution", "-r", type=int, default=-1)
    parser.add_argument("--output", "-o", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build_cache(parse_args())
