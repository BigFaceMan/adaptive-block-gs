import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from plyfile import PlyData, PlyElement

from utils.graphics_utils import BasicPointCloud


AXIS_TO_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
}


def resolve_partition_tree_path(partition_path: str) -> str:
    if not partition_path:
        raise ValueError("partition_path is empty")

    partition_path = os.path.abspath(partition_path)
    if os.path.isdir(partition_path):
        partition_path = os.path.join(partition_path, "partition_tree.json")
    if not os.path.isfile(partition_path):
        raise FileNotFoundError(f"Partition tree not found: {partition_path}")
    return partition_path


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, payload, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=indent)


def append_jsonl(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload) + "\n")


def load_partition_tree(partition_path: str) -> Tuple[dict, str]:
    tree_path = resolve_partition_tree_path(partition_path)
    return load_json(tree_path), tree_path


def get_partition_block(partition_tree: dict, block_id: str) -> dict:
    for block in partition_tree.get("blocks", []):
        if block.get("id") == block_id:
            return block
    raise KeyError(f"Block id '{block_id}' was not found in partition tree")


def normalize_bbox(bbox: Sequence[float]) -> np.ndarray:
    if len(bbox) != 6:
        raise ValueError(f"Expected bbox with 6 values, got {bbox}")
    bbox = np.asarray(bbox, dtype=np.float64)
    mins = np.minimum(bbox[:3], bbox[3:])
    maxs = np.maximum(bbox[:3], bbox[3:])
    return np.concatenate([mins, maxs])


def bbox_min_max(bbox: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    bbox = normalize_bbox(bbox)
    return bbox[:3], bbox[3:]


def bbox_center(bbox: Sequence[float]) -> np.ndarray:
    mins, maxs = bbox_min_max(bbox)
    return 0.5 * (mins + maxs)


def bbox_size(bbox: Sequence[float]) -> np.ndarray:
    mins, maxs = bbox_min_max(bbox)
    return np.maximum(maxs - mins, 0.0)


def expand_bbox(bbox: Sequence[float], expand_ratio: float, min_expand: float = 0.0) -> List[float]:
    mins, maxs = bbox_min_max(bbox)
    extent = np.maximum(maxs - mins, 0.0)
    margin = np.maximum(extent * float(expand_ratio), float(min_expand))
    expanded = np.concatenate([mins - margin, maxs + margin])
    return expanded.astype(float).tolist()


def points_in_bbox(points: np.ndarray, bbox: Sequence[float], eps: float = 0.0) -> np.ndarray:
    mins, maxs = bbox_min_max(bbox)
    return np.all((points >= mins - eps) & (points <= maxs + eps), axis=1)


def contract_to_unisphere(
    points: np.ndarray,
    aabb: Sequence[float],
    ord=np.inf,
    eps: float = 1e-6,
) -> np.ndarray:
    aabb = normalize_bbox(aabb)
    aabb_min = aabb[:3]
    aabb_max = aabb[3:]
    denom = np.maximum(aabb_max - aabb_min, eps)
    x = (np.asarray(points, dtype=np.float64) - aabb_min) / denom
    x = x * 2.0 - 1.0
    mag = np.linalg.norm(x, ord=ord, axis=-1, keepdims=True)
    mask = mag.squeeze(-1) > 1.0
    if np.any(mask):
        x[mask] = (2.0 - 1.0 / mag[mask]) * (x[mask] / mag[mask])
    return x / 4.0 + 0.5


def partition_coord_space(block: Dict) -> str:
    return (
        block.get("partition_coord_space")
        or block.get("coord_space")
        or block.get("_partition_coord_space")
        or "world"
    )


def partition_points(points: np.ndarray, block: Dict) -> np.ndarray:
    coord_space = partition_coord_space(block)
    if coord_space == "world":
        return np.asarray(points, dtype=np.float64)
    if coord_space == "contracted":
        contract_aabb = block.get("contract_aabb") or block.get("_partition_contract_aabb")
        if contract_aabb is None:
            raise KeyError("Contracted partition block has no contract_aabb")
        return contract_to_unisphere(points, contract_aabb, ord=np.inf)
    raise ValueError(f"Unknown partition coord space: {coord_space}")


def points_in_partition_bbox(
    points: np.ndarray,
    block: Dict,
    bbox_key: str = "core_bbox",
    eps: float = 0.0,
) -> np.ndarray:
    if bbox_key not in block:
        raise KeyError(f"Block '{block.get('id', '<unknown>')}' has no '{bbox_key}'")
    return points_in_bbox(partition_points(points, block), block[bbox_key], eps=eps)


def crop_basic_point_cloud(pcd: Optional[BasicPointCloud], bbox: Sequence[float]) -> Optional[BasicPointCloud]:
    if pcd is None:
        return None

    points = np.asarray(pcd.points)
    mask = points_in_bbox(points, bbox)
    if not np.any(mask):
        raise ValueError(f"Partition bbox selected zero initial points: {bbox}")

    return BasicPointCloud(
        points=np.asarray(pcd.points)[mask],
        colors=np.asarray(pcd.colors)[mask],
        normals=np.asarray(pcd.normals)[mask],
    )


def crop_basic_point_cloud_by_block(
    pcd: Optional[BasicPointCloud],
    block: Dict,
    bbox_key: str,
) -> Optional[BasicPointCloud]:
    if pcd is None:
        return None

    points = np.asarray(pcd.points)
    mask = points_in_partition_bbox(points, block, bbox_key)
    if not np.any(mask):
        raise ValueError(f"Partition {bbox_key} selected zero initial points: {block.get(bbox_key)}")

    return BasicPointCloud(
        points=np.asarray(pcd.points)[mask],
        colors=np.asarray(pcd.colors)[mask],
        normals=np.asarray(pcd.normals)[mask],
    )


def write_basic_point_cloud_ply(path: str, pcd: BasicPointCloud) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    xyz = np.asarray(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32)
    rgb = np.asarray(pcd.colors)
    if rgb.size == 0:
        raise ValueError("Cannot write an empty point cloud")
    if np.max(rgb) <= 1.0:
        rgb = np.clip(rgb * 255.0, 0, 255)
    rgb = rgb.astype(np.uint8)

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, np.concatenate([xyz, normals, rgb], axis=1)))
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def camera_name_aliases(name: str) -> set:
    path = Path(name)
    aliases = {name, path.name, path.stem}
    if path.suffix:
        aliases.add(str(path.with_suffix("")))
    return aliases


def filter_camera_infos(camera_infos: Sequence, camera_names: Iterable[str]) -> List:
    allowed = set()
    for name in camera_names:
        allowed.update(camera_name_aliases(str(name)))

    selected = []
    for cam in camera_infos:
        if camera_name_aliases(cam.image_name) & allowed:
            selected.append(cam)
    return selected


def all_camera_infos(scene_info) -> List:
    return list(scene_info.train_cameras) + list(scene_info.test_cameras)


def mark_camera_infos_as_test(camera_infos: Sequence) -> List:
    marked = []
    for cam in camera_infos:
        if getattr(cam, "is_test", False) or not hasattr(cam, "_replace"):
            marked.append(cam)
        else:
            marked.append(cam._replace(is_test=True))
    return marked


def camera_center_from_info(cam_info) -> np.ndarray:
    rt = np.eye(4, dtype=np.float64)
    rt[:3, :3] = cam_info.R.transpose()
    rt[:3, 3] = cam_info.T
    c2w = np.linalg.inv(rt)
    return c2w[:3, 3]


def apply_partition_to_scene_info(
    scene_info,
    partition_path: str,
    block_id: str,
    bbox_mode: str = "expanded",
    crop_initial_points: bool = True,
    load_test_cameras: bool = False,
    test_scene_info=None,
):
    partition_tree, tree_path = load_partition_tree(partition_path)
    block = get_partition_block(partition_tree, block_id)
    block = dict(block)
    block.setdefault(
        "partition_coord_space",
        partition_tree.get("partition_coord_space")
        or partition_tree.get("config", {}).get("partition_coord_space", "world"),
    )
    if block["partition_coord_space"] == "contracted":
        block.setdefault(
            "contract_aabb",
            partition_tree.get("contract_aabb") or partition_tree.get("config", {}).get("contract_aabb"),
        )

    train_cameras = filter_camera_infos(scene_info.train_cameras, block.get("train_cameras", []))
    if not train_cameras:
        raise ValueError(f"Block '{block_id}' selected zero train cameras")

    external_test_cameras = None
    if test_scene_info is not None:
        external_test_cameras = mark_camera_infos_as_test(all_camera_infos(test_scene_info))

    if block.get("test_cameras"):
        test_source_cameras = external_test_cameras if external_test_cameras is not None else scene_info.test_cameras
        test_cameras = filter_camera_infos(test_source_cameras, block.get("test_cameras", []))
    elif load_test_cameras:
        test_cameras = external_test_cameras if external_test_cameras is not None else scene_info.test_cameras
    else:
        test_cameras = []

    if bbox_mode not in {"expanded", "core"}:
        raise ValueError("partition_bbox_mode must be 'expanded' or 'core'")
    bbox_key = "expanded_bbox" if bbox_mode == "expanded" else "core_bbox"
    if bbox_key not in block:
        raise KeyError(f"Block '{block_id}' has no '{bbox_key}'")

    if crop_initial_points:
        point_cloud = crop_basic_point_cloud_by_block(scene_info.point_cloud, block, bbox_key)
    else:
        point_cloud = scene_info.point_cloud

    block["_partition_tree_path"] = tree_path
    block["_partition_bbox_mode"] = bbox_mode
    block["_partition_init_mode"] = "cropped" if crop_initial_points else "coarse"
    block["_partition_coarse_model"] = partition_tree.get("coarse_model") or partition_tree.get("config", {}).get("coarse_model", "")
    block["_partition_coord_space"] = partition_coord_space(block)
    block["_partition_contract_aabb"] = block.get("contract_aabb")
    block["_num_train_cameras_loaded"] = len(train_cameras)
    block["_num_test_cameras_loaded"] = len(test_cameras)
    block["_num_initial_points_loaded"] = int(len(point_cloud.points)) if point_cloud is not None else 0

    filtered_scene_info = scene_info._replace(
        point_cloud=point_cloud,
        train_cameras=train_cameras,
        test_cameras=test_cameras,
    )
    return filtered_scene_info, block


def write_block_training_metadata(model_path: str, block_metadata: Dict) -> None:
    save_json(os.path.join(model_path, "partition_block_metadata.json"), block_metadata)


def read_gaussian_or_point_ply(path: str) -> Dict[str, np.ndarray]:
    plydata = PlyData.read(path)
    vertex = plydata["vertex"]
    names = vertex.data.dtype.names
    xyz = np.stack([np.asarray(vertex[name], dtype=np.float64) for name in ("x", "y", "z")], axis=1)

    if "opacity" in names:
        raw_opacity = np.asarray(vertex["opacity"], dtype=np.float64)
        opacity = 1.0 / (1.0 + np.exp(-raw_opacity))
    else:
        opacity = np.ones(xyz.shape[0], dtype=np.float64)

    scale_names = sorted(
        [name for name in names if name.startswith("scale_")],
        key=lambda item: int(item.split("_")[-1]),
    )
    if scale_names:
        raw_scale = np.stack([np.asarray(vertex[name], dtype=np.float64) for name in scale_names], axis=1)
        scale = np.exp(raw_scale)
    else:
        scale = np.ones((xyz.shape[0], 3), dtype=np.float64)

    return {
        "xyz": xyz,
        "opacity": opacity,
        "scale": scale,
    }


def bbox_corners(bbox: Sequence[float]) -> np.ndarray:
    mins, maxs = bbox_min_max(bbox)
    return np.array(
        [
            [mins[0], mins[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], maxs[1], maxs[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], mins[2]],
            [maxs[0], maxs[1], maxs[2]],
        ],
        dtype=np.float64,
    )
