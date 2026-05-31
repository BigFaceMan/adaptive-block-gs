import json
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from utils.partition_utils import AXIS_TO_INDEX, load_partition_tree, partition_points


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def input_ply_path(input_model_path: str, iteration: int) -> str:
    return os.path.join(
        input_model_path,
        "point_cloud",
        f"iteration_{iteration}",
        "point_cloud.ply",
    )


def normalize_boundary_axes(axes: Iterable[str]) -> List[str]:
    normalized = []
    for axis in axes:
        axis = str(axis).lower()
        if axis not in AXIS_TO_INDEX:
            raise ValueError(f"Unknown boundary axis: {axis}")
        if axis not in normalized:
            normalized.append(axis)
    if not normalized:
        raise ValueError("boundary_axes must not be empty")
    return normalized


def block_ranges_from_merge_report(merge_report: Dict) -> List[Tuple[str, int, int]]:
    ranges = []
    cursor = 0
    for block in merge_report.get("blocks", []):
        kept = int(block.get("kept_gaussians", 0))
        block_id = block.get("id")
        if block.get("status", "merged") == "missing" or kept <= 0:
            continue
        if not block_id:
            raise ValueError("merge_report block entry has no id")
        ranges.append((block_id, cursor, cursor + kept))
        cursor += kept
    return ranges


def _block_boundary_mask(points: np.ndarray, block: Dict, band_ratio: float, axes: Sequence[str]) -> np.ndarray:
    if "core_bbox" not in block:
        raise KeyError(f"Block {block.get('id', '<unknown>')} has no core_bbox")

    partition_xyz = partition_points(points, block)
    bbox = np.asarray(block["core_bbox"], dtype=np.float64)
    mins = bbox[:3]
    maxs = bbox[3:]
    sizes = np.maximum(maxs - mins, 1e-12)

    boundary = np.zeros((partition_xyz.shape[0],), dtype=bool)
    for axis in axes:
        axis_idx = AXIS_TO_INDEX[axis]
        band = float(sizes[axis_idx]) * float(band_ratio)
        lower_distance = partition_xyz[:, axis_idx] - mins[axis_idx]
        upper_distance = maxs[axis_idx] - partition_xyz[:, axis_idx]
        boundary |= np.minimum(lower_distance, upper_distance) <= band
    return boundary


def build_boundary_mask(
    xyz: np.ndarray,
    partition_path: str,
    merge_report_path: str,
    band_ratio: float,
    axes: Sequence[str],
) -> Tuple[np.ndarray, Dict]:
    if band_ratio < 0:
        raise ValueError("boundary_band_ratio must be >= 0")

    axes = normalize_boundary_axes(axes)
    partition_tree, resolved_partition_path = load_partition_tree(partition_path)
    merge_report = load_json(merge_report_path)
    block_by_id = {block["id"]: block for block in partition_tree.get("blocks", [])}
    ranges = block_ranges_from_merge_report(merge_report)

    if not ranges:
        raise ValueError(f"No merged block ranges found in {merge_report_path}")

    expected_points = ranges[-1][2]
    if int(xyz.shape[0]) != expected_points:
        raise ValueError(
            f"Merged PLY point count ({xyz.shape[0]}) does not match merge_report total ({expected_points})"
        )

    boundary_mask = np.zeros((xyz.shape[0],), dtype=bool)
    block_reports = []
    for block_id, start, end in ranges:
        block = block_by_id.get(block_id)
        if block is None:
            raise KeyError(f"Block {block_id} from merge_report not found in partition tree")

        local_mask = _block_boundary_mask(xyz[start:end], block, band_ratio, axes)
        boundary_mask[start:end] = local_mask
        boundary_count = int(local_mask.sum())
        total_count = int(end - start)
        block_reports.append(
            {
                "block_id": block_id,
                "start": int(start),
                "end": int(end),
                "points": total_count,
                "boundary_points": boundary_count,
                "internal_points": total_count - boundary_count,
                "boundary_ratio": float(boundary_count / max(total_count, 1)),
            }
        )

    report = {
        "partition_path": resolved_partition_path,
        "merge_report_path": merge_report_path,
        "boundary_band_ratio": float(band_ratio),
        "boundary_axes": list(axes),
        "total_points": int(xyz.shape[0]),
        "boundary_points": int(boundary_mask.sum()),
        "internal_points": int(xyz.shape[0] - boundary_mask.sum()),
        "boundary_ratio": float(boundary_mask.sum() / max(int(xyz.shape[0]), 1)),
        "blocks": block_reports,
    }
    return boundary_mask, report


def save_boundary_report(path: str, report: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
