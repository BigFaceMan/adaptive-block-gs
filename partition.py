import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.config_schema import (
    ExperimentConfig,
    config_from_dict,
    config_to_dict,
    get_default_config,
    load_config,
)
from utils.config_utils import save_yaml_config
from utils.partition_utils import (
    AXIS_TO_INDEX,
    bbox_center,
    bbox_corners,
    bbox_size,
    camera_center_from_info,
    contract_to_unisphere,
    expand_bbox,
    partition_points,
    points_in_bbox,
    read_gaussian_or_point_ply,
    save_json,
)
from utils.graphics_utils import fov2focal, getProjectionMatrix, getWorld2View2


PARTITION_LOG_INTERVAL = 1000


def make_progress_logger(prefix: str = "partition") -> Callable[[str], None]:
    start = time.perf_counter()

    def log(message: str) -> None:
        elapsed = time.perf_counter() - start
        print(f"[{prefix} {elapsed:8.1f}s] {message}", flush=True)

    return log


def score_source_counts(scores: Dict[str, dict]) -> Dict[str, int]:
    counts = {}
    for score in scores.values():
        source = score.get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def format_counts(counts: Dict[str, int]) -> str:
    if not counts:
        return "none"
    return " ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def camera_assignment_summary(records: Sequence[dict], selected: Dict[str, dict], candidates: Dict[str, dict]) -> Dict[str, int]:
    pool_in_block = sum(1 for record in records if record.get("inside_core", False))
    selected_in_block = sum(1 for score in selected.values() if score.get("center_inside_core_bbox", False))
    candidate_in_block = sum(1 for score in candidates.values() if score.get("center_inside_core_bbox", False))
    return {
        "pool_cameras": len(records),
        "pool_in_block_cameras": int(pool_in_block),
        "pool_outside_block_cameras": int(len(records) - pool_in_block),
        "selected_cameras": len(selected),
        "selected_in_block_cameras": int(selected_in_block),
        "selected_added_cameras": int(len(selected) - selected_in_block),
        "candidate_cameras": len(candidates),
        "candidate_in_block_cameras": int(candidate_in_block),
        "candidate_added_cameras": int(len(candidates) - candidate_in_block),
    }


def parse_config() -> ExperimentConfig:
    defaults = get_default_config()
    dataset_defaults = defaults.dataset
    model_defaults = defaults.model
    partition_defaults = defaults.partition
    camera_defaults = defaults.camera_assignment
    render_diff_defaults = camera_defaults.render_difference
    vis_defaults = defaults.visualization

    parser = argparse.ArgumentParser(description="Recursive coarse-GS guided scene partitioning")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("-s", "--source_path", default=dataset_defaults.source_path)
    parser.add_argument("--images", default=dataset_defaults.images)
    parser.add_argument("--depths", default=dataset_defaults.depths)
    parser.add_argument("--test_source_path", default=dataset_defaults.test_source_path)
    parser.add_argument("--test_images", default=dataset_defaults.test_images)
    parser.add_argument("--test_depths", default=dataset_defaults.test_depths)
    parser.add_argument("--eval", action="store_true", default=dataset_defaults.eval)
    parser.add_argument("--train_test_exp", action="store_true", default=dataset_defaults.train_test_exp)
    parser.add_argument("--white_background", action="store_true", default=dataset_defaults.white_background)
    parser.add_argument("--coarse_model", default=model_defaults.coarse_model)
    parser.add_argument("--partition_output", default=partition_defaults.output_path)
    parser.add_argument("--partition_coord_space", choices=["world", "contracted"], default=partition_defaults.coord_space)
    parser.add_argument("--contract_aabb", nargs=6, type=float, default=partition_defaults.contract_aabb)
    parser.add_argument("--partition_axes", nargs="+", default=list(partition_defaults.axes), choices=["x", "y", "z"])
    parser.add_argument("--max_depth", type=int, default=partition_defaults.max_depth)
    parser.add_argument("--max_blocks", type=int, default=partition_defaults.max_blocks)
    parser.add_argument("--max_block_importance", type=float, default=partition_defaults.max_block_importance)
    parser.add_argument("--max_block_density", type=float, default=partition_defaults.max_block_density)
    parser.add_argument("--min_points", type=int, default=partition_defaults.min_points)
    parser.add_argument("--min_size", type=float, default=partition_defaults.min_size)
    parser.add_argument("--expand_ratio", type=float, default=partition_defaults.expand_ratio)
    parser.add_argument("--num_split_candidates", type=int, default=partition_defaults.num_split_candidates)
    parser.add_argument("--lambda_boundary", type=float, default=partition_defaults.lambda_boundary)
    parser.add_argument("--importance", choices=["opacity", "opacity_scale"], default=partition_defaults.importance)
    parser.add_argument("--tau_projection", type=float, default=camera_defaults.tau_projection)
    parser.add_argument("--tau_test_projection", type=float, default=camera_defaults.tau_test_projection)
    parser.add_argument("--min_cameras", type=int, default=camera_defaults.min_cameras)
    parser.add_argument("--min_test_cameras", type=int, default=camera_defaults.min_test_cameras)
    parser.add_argument("--supplement_cameras", action="store_true", default=camera_defaults.supplement_cameras)
    parser.add_argument("--camera_projection_max_points", type=int, default=camera_defaults.projection_max_points)
    parser.add_argument("--render_difference_cameras", action="store_true", default=render_diff_defaults.enabled)
    parser.add_argument("--render_difference_threshold", type=float, default=render_diff_defaults.threshold)
    parser.add_argument("--render_difference_max_candidates_per_block", type=int, default=render_diff_defaults.max_candidates_per_block)
    parser.add_argument("--render_difference_max_width", type=int, default=render_diff_defaults.max_width)
    parser.add_argument("--render_difference_cache_full", action="store_true", default=render_diff_defaults.cache_full)
    parser.add_argument("--visualize_blocks", action="store_true", default=vis_defaults.visualize_blocks)
    parser.add_argument("--visualize_output", default=vis_defaults.visualize_output)
    parser.add_argument("--visualize_bbox_mode", choices=["core", "expanded"], default=vis_defaults.visualize_bbox_mode)
    parser.add_argument("--visualize_max_cameras_per_block", type=int, default=vis_defaults.visualize_max_cameras_per_block)
    parser.add_argument("--visualize_max_points_per_block", type=int, default=vis_defaults.visualize_max_points_per_block)
    parser.add_argument("--visualize_point_radius", type=int, default=vis_defaults.visualize_point_radius)
    parser.add_argument("--visualize_max_image_width", type=int, default=vis_defaults.visualize_max_image_width)
    parser.add_argument("--visualize_random_seed", type=int, default=vis_defaults.visualize_random_seed)
    parser.add_argument("--visualize_topdown", action="store_true", default=vis_defaults.visualize_topdown)
    parser.add_argument("--visualize_topdown_output", default=vis_defaults.visualize_topdown_output)
    parser.add_argument("--visualize_topdown_max_points", type=int, default=vis_defaults.visualize_topdown_max_points)
    parser.add_argument("--visualize_topdown_image_size", type=int, default=vis_defaults.visualize_topdown_image_size)
    parser.add_argument("--visualize_topdown_point_radius", type=int, default=vis_defaults.visualize_topdown_point_radius)
    parser.add_argument("--visualize_topdown_bbox_mode", choices=["core", "expanded", "both"], default=vis_defaults.visualize_topdown_bbox_mode)
    parser.add_argument("--visualize_topdown_clip_percentile", type=float, default=vis_defaults.visualize_topdown_clip_percentile)
    parser.add_argument("--visualize_topdown_color", choices=["auto", "rgb", "height", "gray"], default=vis_defaults.visualize_topdown_color)
    parser.add_argument("--visualize_topdown_keep", choices=["max", "min", "all"], default=vis_defaults.visualize_topdown_keep)
    args = parser.parse_args()
    cfg = load_config(args.config, args.override) if args.config else config_from_legacy_args(args)
    if cfg.camera_assignment.tau_test_projection < 0:
        cfg.camera_assignment.tau_test_projection = cfg.camera_assignment.tau_projection
    missing = []
    if not cfg.dataset.source_path:
        missing.append("dataset.source_path")
    if not cfg.model.coarse_model:
        missing.append("model.coarse_model")
    if not cfg.partition.output_path:
        missing.append("partition.output_path")
    if missing:
        raise ValueError(f"Missing required partition argument(s): {', '.join(missing)}")
    return cfg


def config_from_legacy_args(args) -> ExperimentConfig:
    return config_from_dict(
        {
            "experiment": {
                "name": "legacy_partition",
                "output_root": os.path.dirname(os.path.abspath(args.partition_output)) if args.partition_output else "",
            },
            "dataset": {
                "source_path": args.source_path,
                "images": args.images,
                "depths": args.depths,
                "test_source_path": args.test_source_path,
                "test_images": args.test_images,
                "test_depths": args.test_depths,
                "eval": args.eval,
                "train_test_exp": args.train_test_exp,
                "white_background": args.white_background,
            },
            "model": {
                "coarse_model": args.coarse_model,
            },
            "partition": {
                "output_path": args.partition_output,
                "coord_space": args.partition_coord_space,
                "contract_aabb": args.contract_aabb,
                "axes": args.partition_axes,
                "max_depth": args.max_depth,
                "max_blocks": args.max_blocks,
                "max_block_importance": args.max_block_importance,
                "max_block_density": args.max_block_density,
                "min_points": args.min_points,
                "min_size": args.min_size,
                "expand_ratio": args.expand_ratio,
                "num_split_candidates": args.num_split_candidates,
                "lambda_boundary": args.lambda_boundary,
                "importance": args.importance,
            },
            "camera_assignment": {
                "tau_projection": args.tau_projection,
                "tau_test_projection": args.tau_test_projection,
                "min_cameras": args.min_cameras,
                "min_test_cameras": args.min_test_cameras,
                "supplement_cameras": args.supplement_cameras,
                "projection_max_points": args.camera_projection_max_points,
                "render_difference": {
                    "enabled": args.render_difference_cameras,
                    "threshold": args.render_difference_threshold,
                    "max_candidates_per_block": args.render_difference_max_candidates_per_block,
                    "max_width": args.render_difference_max_width,
                    "cache_full": args.render_difference_cache_full,
                },
            },
            "visualization": {
                "visualize_blocks": args.visualize_blocks,
                "visualize_output": args.visualize_output,
                "visualize_bbox_mode": args.visualize_bbox_mode,
                "visualize_max_cameras_per_block": args.visualize_max_cameras_per_block,
                "visualize_max_points_per_block": args.visualize_max_points_per_block,
                "visualize_point_radius": args.visualize_point_radius,
                "visualize_max_image_width": args.visualize_max_image_width,
                "visualize_random_seed": args.visualize_random_seed,
                "visualize_topdown": args.visualize_topdown,
                "visualize_topdown_output": args.visualize_topdown_output,
                "visualize_topdown_max_points": args.visualize_topdown_max_points,
                "visualize_topdown_image_size": args.visualize_topdown_image_size,
                "visualize_topdown_point_radius": args.visualize_topdown_point_radius,
                "visualize_topdown_bbox_mode": args.visualize_topdown_bbox_mode,
                "visualize_topdown_clip_percentile": args.visualize_topdown_clip_percentile,
                "visualize_topdown_color": args.visualize_topdown_color,
                "visualize_topdown_keep": args.visualize_topdown_keep,
            },
        }
    )


def load_scene_info_from_path(source_path, images, depths, eval_split, train_test_exp, white_background):
    source_path = os.path.abspath(source_path)
    if os.path.exists(os.path.join(source_path, "sparse")):
        return sceneLoadTypeCallbacks["Colmap"](
            source_path,
            images,
            depths,
            eval_split,
            train_test_exp,
        )
    if os.path.exists(os.path.join(source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        return sceneLoadTypeCallbacks["Blender"](
            source_path,
            white_background,
            depths,
            eval_split,
        )
    raise RuntimeError(f"Could not recognize scene type: {source_path}")


def load_scene_info(cfg: ExperimentConfig):
    return load_scene_info_from_path(
        cfg.dataset.source_path,
        cfg.dataset.images,
        cfg.dataset.depths,
        cfg.dataset.eval,
        cfg.dataset.train_test_exp,
        cfg.dataset.white_background,
    )


def all_camera_infos(scene_info) -> List:
    return list(scene_info.train_cameras) + list(scene_info.test_cameras)


def compute_importance(coarse: Dict[str, np.ndarray], mode: str) -> np.ndarray:
    opacity = coarse["opacity"].astype(np.float64)
    if mode == "opacity":
        weights = opacity
    else:
        mean_scale = np.mean(coarse["scale"], axis=1)
        weights = opacity * np.maximum(mean_scale, 1e-8)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(weights, 0.0)


def root_bbox_from_points(points: np.ndarray) -> List[float]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extent = np.maximum(maxs - mins, 1e-6)
    pad = extent * 1e-6
    return np.concatenate([mins - pad, maxs + pad]).astype(float).tolist()


def block_density(importance: float, bbox: Sequence[float], partition_axes: Sequence[str]) -> float:
    sizes = bbox_size(bbox)
    area = 1.0
    for axis in partition_axes:
        area *= max(float(sizes[AXIS_TO_INDEX[axis]]), 1e-12)
    return float(importance / area)


def weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    total = weights.sum()
    if total <= 0:
        return float(np.var(values))
    mean = np.sum(values * weights) / total
    return float(np.sum(weights * (values - mean) ** 2) / total)


def choose_split_axis(
    xyz: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
    bbox: Sequence[float],
    partition_axes: Sequence[str],
    min_size: float,
) -> Optional[str]:
    sizes = bbox_size(bbox)
    candidates = []
    for axis in partition_axes:
        axis_idx = AXIS_TO_INDEX[axis]
        if sizes[axis_idx] <= min_size:
            continue
        variance = weighted_variance(xyz[indices, axis_idx], weights[indices])
        candidates.append((variance, axis))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def choose_split_position(
    xyz: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
    axis: str,
    num_candidates: int,
    lambda_boundary: float,
    min_points: int,
) -> Tuple[Optional[float], Optional[dict]]:
    axis_idx = AXIS_TO_INDEX[axis]
    coords = xyz[indices, axis_idx]
    node_weights = weights[indices]
    total_weight = float(node_weights.sum())
    if total_weight <= 0 or coords.size < 2:
        return None, None

    low = float(coords.min())
    high = float(coords.max())
    if high <= low:
        return None, None

    quantiles = np.linspace(0.05, 0.95, max(num_candidates, 2))
    candidates = np.unique(np.quantile(coords, quantiles))
    boundary_width = max((high - low) * 0.02, 1e-9)
    min_child_points = max(1, min_points)

    best = None
    for candidate in candidates:
        left_mask = coords <= candidate
        right_mask = ~left_mask
        left_count = int(left_mask.sum())
        right_count = int(right_mask.sum())
        if left_count < min_child_points or right_count < min_child_points:
            continue

        left_weight = float(node_weights[left_mask].sum())
        right_weight = float(node_weights[right_mask].sum())
        balance_loss = abs(left_weight - right_weight) / total_weight
        boundary_penalty = float(node_weights[np.abs(coords - candidate) <= boundary_width].sum() / total_weight)
        score = balance_loss + lambda_boundary * boundary_penalty
        record = {
            "position": float(candidate),
            "score": float(score),
            "balance_loss": float(balance_loss),
            "boundary_penalty": float(boundary_penalty),
            "left_points": left_count,
            "right_points": right_count,
            "left_importance": left_weight,
            "right_importance": right_weight,
        }
        if best is None or score < best["score"]:
            best = record

    if best is None:
        return None, None
    return best["position"], best


def project_bbox_coverage(cam_info, bbox: Sequence[float]) -> float:
    corners = bbox_corners(bbox)
    camera_points = corners @ cam_info.R + cam_info.T
    z = camera_points[:, 2]
    front = z > 1e-6
    if not np.any(front):
        return 0.0

    camera_points = camera_points[front]
    z = camera_points[:, 2]
    ndc_x = (camera_points[:, 0] / z) / math.tan(cam_info.FovX * 0.5)
    ndc_y = (camera_points[:, 1] / z) / math.tan(cam_info.FovY * 0.5)
    min_x, max_x = float(ndc_x.min()), float(ndc_x.max())
    min_y, max_y = float(ndc_y.min()), float(ndc_y.max())
    if max_x < -1.0 or min_x > 1.0 or max_y < -1.0 or min_y > 1.0:
        return 0.0

    clipped_min_x = max(min_x, -1.0)
    clipped_max_x = min(max_x, 1.0)
    clipped_min_y = max(min_y, -1.0)
    clipped_max_y = min(max_y, 1.0)
    if clipped_max_x <= clipped_min_x or clipped_max_y <= clipped_min_y:
        return 0.0
    return float((clipped_max_x - clipped_min_x) * (clipped_max_y - clipped_min_y) / 4.0)


def project_points_to_image(points: np.ndarray, cam_info) -> Tuple[np.ndarray, np.ndarray]:
    camera_points = points @ cam_info.R + cam_info.T
    z = camera_points[:, 2]
    front = z > 1e-6
    if not np.any(front):
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float64)

    camera_points = camera_points[front]
    z = z[front]
    fx = fov2focal(cam_info.FovX, cam_info.width)
    fy = fov2focal(cam_info.FovY, cam_info.height)
    u = fx * (camera_points[:, 0] / z) + cam_info.width * 0.5
    v = fy * (camera_points[:, 1] / z) + cam_info.height * 0.5
    in_image = (u >= 0) & (u < cam_info.width) & (v >= 0) & (v < cam_info.height)
    if not np.any(in_image):
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float64)

    pixels = np.stack([u[in_image], v[in_image]], axis=1).astype(np.int32)
    return pixels, z[in_image]


def sample_indices(indices: np.ndarray, weights: Optional[np.ndarray], max_points: int, rng: np.random.RandomState):
    if max_points <= 0 or indices.size <= max_points:
        return indices
    if weights is not None:
        probs = np.asarray(weights[indices], dtype=np.float64)
        if probs.sum() > 0:
            probs = probs / probs.sum()
            return rng.choice(indices, size=max_points, replace=False, p=probs)
    return rng.choice(indices, size=max_points, replace=False)


def project_point_coverage(
    cam_info,
    points: np.ndarray,
    weights: Optional[np.ndarray] = None,
    max_points: int = 5000,
    rng: Optional[np.random.RandomState] = None,
) -> float:
    if points is None or points.shape[0] == 0:
        return 0.0
    if rng is None:
        rng = np.random.RandomState(0)
    point_indices = np.arange(points.shape[0])
    if max_points > 0 and point_indices.size > max_points:
        point_indices = sample_indices(point_indices, weights, max_points, rng)
    pixels, _ = project_points_to_image(points[point_indices], cam_info)
    if pixels.shape[0] == 0:
        return 0.0
    min_xy = pixels.min(axis=0)
    max_xy = pixels.max(axis=0)
    span = np.maximum(max_xy - min_xy + 1, 0)
    return float((span[0] * span[1]) / max(cam_info.width * cam_info.height, 1))


class RenderCamera:
    def __init__(self, cam_info, max_width: int = 0):
        self.uid = cam_info.uid
        self.colmap_id = cam_info.uid
        self.R = cam_info.R
        self.T = cam_info.T
        self.FoVx = cam_info.FovX
        self.FoVy = cam_info.FovY
        self.image_name = cam_info.image_name
        width = int(cam_info.width)
        height = int(cam_info.height)
        if max_width > 0 and width > max_width:
            scale = max_width / width
            width = max_width
            height = max(1, int(round(height * scale)))
        self.image_width = width
        self.image_height = height
        self.zfar = 100.0
        self.znear = 0.01
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear,
            zfar=self.zfar,
            fovX=self.FoVx,
            fovY=self.FoVy,
        ).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def masked_gaussian_model(gaussians, mask):
    from scene.gaussian_model import GaussianModel

    model = GaussianModel(gaussians.max_sh_degree)
    model.active_sh_degree = gaussians.active_sh_degree
    model._xyz = gaussians._xyz[mask]
    model._features_dc = gaussians._features_dc[mask]
    model._features_rest = gaussians._features_rest[mask]
    model._opacity = gaussians._opacity[mask]
    model._scaling = gaussians._scaling[mask]
    model._rotation = gaussians._rotation[mask]
    model.max_radii2D = gaussians.max_radii2D[mask] if gaussians.max_radii2D.numel() else torch.empty(0, device="cuda")
    return model


class RenderDifferenceAssigner:
    def __init__(self, cfg: ExperimentConfig, coarse_xyz: np.ndarray):
        from gaussian_renderer import render
        from scene.gaussian_model import GaussianModel
        from utils.loss_utils import ssim

        self.cfg = cfg
        self.render = render
        self.ssim = ssim
        self.coarse_xyz = coarse_xyz
        self.gaussians = GaussianModel(3)
        self.gaussians.load_ply(cfg.model.coarse_model, False)
        self.background = torch.tensor(
            [1, 1, 1] if cfg.dataset.white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )
        self.pipe = type(
            "Pipeline",
            (),
            {
                "convert_SHs_python": False,
                "compute_cov3D_python": False,
                "debug": False,
                "antialiasing": False,
            },
        )()
        self.full_cache = {}
        self.keep_mask_cache = {}

    def full_render(self, cam_info):
        cache_key = f"{getattr(cam_info, 'image_path', cam_info.image_name)}:{cam_info.width}x{cam_info.height}"
        render_diff = self.cfg.camera_assignment.render_difference
        if render_diff.cache_full and cache_key in self.full_cache:
            return self.full_cache[cache_key].cuda()
        camera = RenderCamera(cam_info, render_diff.max_width)
        with torch.no_grad():
            image = self.render(camera, self.gaussians, self.pipe, self.background)["render"].detach()
        if render_diff.cache_full:
            self.full_cache[cache_key] = image.cpu()
        return image

    def score(self, block: dict, cam_info) -> float:
        full_image = self.full_render(cam_info)
        block_id = block.get("id", str(id(block)))
        if block_id not in self.keep_mask_cache:
            in_block = points_in_bbox(partition_points(self.coarse_xyz, block), block["core_bbox"])
            self.keep_mask_cache[block_id] = torch.from_numpy(~in_block).bool().cuda()
        keep_mask = self.keep_mask_cache[block_id]
        if keep_mask.sum().item() == keep_mask.shape[0]:
            return 0.0
        if keep_mask.sum().item() == 0:
            return 1.0
        without_model = masked_gaussian_model(self.gaussians, keep_mask)
        camera = RenderCamera(cam_info, self.cfg.camera_assignment.render_difference.max_width)
        with torch.no_grad():
            without_image = self.render(camera, without_model, self.pipe, self.background)["render"].detach()
            return float((1.0 - self.ssim(without_image, full_image)).item())


def block_color(block_index: int) -> np.ndarray:
    palette = np.array(
        [
            [255, 64, 64],
            [64, 180, 255],
            [255, 210, 64],
            [96, 220, 120],
            [220, 100, 255],
            [255, 140, 64],
            [80, 255, 230],
            [255, 96, 170],
        ],
        dtype=np.float32,
    )
    return palette[block_index % len(palette)]


def overlay_projected_points(
    image_path: str,
    output_path: str,
    pixels: np.ndarray,
    bbox_pixels: np.ndarray,
    color: np.ndarray,
    radius: int,
    max_image_width: int,
) -> None:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        arr = np.asarray(image).copy()

    if pixels.shape[0] > 0:
        alpha = 0.75
        radius = max(0, int(radius))
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                xs = pixels[:, 0] + dx
                ys = pixels[:, 1] + dy
                valid = (xs >= 0) & (xs < arr.shape[1]) & (ys >= 0) & (ys < arr.shape[0])
                if np.any(valid):
                    arr[ys[valid], xs[valid]] = (
                        arr[ys[valid], xs[valid]].astype(np.float32) * (1.0 - alpha) + color * alpha
                    ).astype(np.uint8)

    image = Image.fromarray(arr)
    if bbox_pixels.shape[0] > 0:
        draw = ImageDraw.Draw(image)
        min_xy = bbox_pixels.min(axis=0)
        max_xy = bbox_pixels.max(axis=0)
        draw.rectangle(
            [tuple(min_xy.tolist()), tuple(max_xy.tolist())],
            outline=(0, 255, 255),
            width=3,
        )

    if max_image_width > 0 and image.width > max_image_width:
        scale = max_image_width / image.width
        new_size = (max_image_width, max(1, int(round(image.height * scale))))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path, quality=92)


def select_visualization_cameras(block: dict, train_cameras: Sequence, max_cameras: int) -> List:
    if max_cameras <= 0:
        return []
    camera_by_name = {cam.image_name: cam for cam in train_cameras}
    scores = block.get("camera_scores", {})
    ranked_names = sorted(
        block.get("train_cameras", []),
        key=lambda name: scores.get(name, {}).get("final_score", 0.0),
        reverse=True,
    )
    selected = []
    for name in ranked_names:
        cam = camera_by_name.get(name)
        if cam is not None:
            selected.append(cam)
        if len(selected) >= max_cameras:
            break
    return selected


def visualize_partition_blocks(
    cfg: ExperimentConfig,
    blocks: Sequence[dict],
    train_cameras: Sequence,
    coarse_xyz: np.ndarray,
    partition_xyz: np.ndarray,
    weights: np.ndarray,
    partition_output: str,
) -> List[dict]:
    vis = cfg.visualization
    output_root = os.path.abspath(vis.visualize_output) if vis.visualize_output else os.path.join(partition_output, "visualizations")
    rng = np.random.RandomState(vis.visualize_random_seed)
    reports = []
    contracted_visualization = cfg.partition.coord_space == "contracted"

    for block_index, block in enumerate(blocks):
        bbox_key = "core_bbox" if contracted_visualization or vis.visualize_bbox_mode == "core" else "expanded_bbox"
        point_mask = points_in_bbox(partition_xyz, block[bbox_key])
        point_indices = np.flatnonzero(point_mask)
        num_block_points = int(point_indices.size)
        if point_indices.size == 0:
            reports.append({"block_id": block["id"], "num_block_points": 0, "images": []})
            continue

        max_points = vis.visualize_max_points_per_block
        if max_points > 0 and point_indices.size > max_points:
            probs = weights[point_indices].astype(np.float64)
            if probs.sum() > 0:
                probs = probs / probs.sum()
                point_indices = rng.choice(point_indices, size=max_points, replace=False, p=probs)
            else:
                point_indices = rng.choice(point_indices, size=max_points, replace=False)

        block_points = coarse_xyz[point_indices]
        cameras = select_visualization_cameras(block, train_cameras, vis.visualize_max_cameras_per_block)
        block_dir = os.path.join(output_root, block["id"])
        color = block_color(block_index)
        image_reports = []

        for cam in cameras:
            pixels, _ = project_points_to_image(block_points, cam)
            if contracted_visualization:
                bbox_pixels = np.empty((0, 2), dtype=np.int32)
            else:
                bbox_pixels, _ = project_points_to_image(bbox_corners(block[bbox_key]), cam)
            output_name = f"{Path(cam.image_name).stem}.jpg"
            output_path = os.path.join(block_dir, output_name)
            overlay_projected_points(
                cam.image_path,
                output_path,
                pixels,
                bbox_pixels,
                color,
                vis.visualize_point_radius,
                vis.visualize_max_image_width,
            )
            image_reports.append(
                {
                    "image_name": cam.image_name,
                    "output_path": output_path,
                    "num_projected_points": int(pixels.shape[0]),
                }
            )

        reports.append(
            {
                "block_id": block["id"],
                "bbox_mode": "core_point_membership" if contracted_visualization else vis.visualize_bbox_mode,
                "draw_bbox": not contracted_visualization,
                "num_block_points": num_block_points,
                "num_sampled_points": int(block_points.shape[0]),
                "images": image_reports,
            }
        )

    save_json(os.path.join(output_root, "visualization_report.json"), reports)
    return reports


def sample_points_for_visualization(points: np.ndarray, max_points: int, rng: np.random.RandomState) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return np.arange(points.shape[0])
    return rng.choice(np.arange(points.shape[0]), size=max_points, replace=False)


def projection_depth_axis(axes: Sequence[str]) -> str:
    remaining = [axis for axis in ("x", "y", "z") if axis not in axes[:2]]
    if not remaining:
        return axes[-1]
    return remaining[0]


def point_cloud_rgb(point_cloud, num_points: int) -> Optional[np.ndarray]:
    colors = getattr(point_cloud, "colors", None)
    if colors is None:
        return None
    colors = np.asarray(colors)
    if colors.shape[0] != num_points or colors.ndim != 2 or colors.shape[1] < 3:
        return None

    rgb = colors[:, :3].astype(np.float32, copy=False)
    if rgb.size == 0:
        return None
    finite = np.isfinite(rgb)
    if not finite.any():
        return None
    if np.nanmax(rgb[finite]) <= 1.0 + 1e-6:
        rgb = rgb * 255.0
    return np.clip(np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0), 0, 255).astype(np.uint8)


def projection_bounds(
    points_2d: np.ndarray,
    depth: np.ndarray,
    clip_percentile: float,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    u = points_2d[:, 0]
    v = points_2d[:, 1]
    finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(depth)
    if not finite.any():
        raise ValueError("No finite points found for top-down visualization")

    if clip_percentile > 0:
        lo = clip_percentile
        hi = 100.0 - clip_percentile
        u_min, u_max = np.percentile(u[finite], [lo, hi])
        v_min, v_max = np.percentile(v[finite], [lo, hi])
    else:
        u_min = float(np.min(u[finite]))
        u_max = float(np.max(u[finite]))
        v_min = float(np.min(v[finite]))
        v_max = float(np.max(v[finite]))

    valid = finite & (u >= u_min) & (u <= u_max) & (v >= v_min) & (v <= v_max)
    if not valid.any():
        raise ValueError("No points left after top-down clipping")
    return valid, (float(u_min), float(u_max), float(v_min), float(v_max))


def topdown_height_colors(values: np.ndarray, limits: Optional[Tuple[float, float]] = None) -> np.ndarray:
    if limits is None:
        lo, hi = np.percentile(values, [2.0, 98.0])
    else:
        lo, hi = limits
    if hi <= lo:
        hi = lo + 1.0

    t = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    stops = np.array(
        [
            [49, 54, 149],
            [69, 117, 180],
            [116, 173, 209],
            [171, 217, 233],
            [224, 243, 248],
            [254, 224, 144],
            [253, 174, 97],
            [244, 109, 67],
            [215, 48, 39],
            [165, 0, 38],
        ],
        dtype=np.float32,
    )
    pos = t * (len(stops) - 1)
    idx = np.floor(pos).astype(np.int32)
    idx = np.clip(idx, 0, len(stops) - 2)
    frac = (pos - idx)[:, None]
    colors = stops[idx] * (1.0 - frac) + stops[idx + 1] * frac
    return colors.astype(np.uint8)


def select_visible_topdown_points(
    px: np.ndarray,
    py: np.ndarray,
    depth: np.ndarray,
    width: int,
    keep: str,
) -> np.ndarray:
    if keep == "all":
        return np.arange(len(px), dtype=np.int64)

    linear = py.astype(np.int64) * width + px.astype(np.int64)
    depth_key = depth if keep == "max" else -depth
    order = np.lexsort((depth_key, linear))
    sorted_linear = linear[order]
    last_in_pixel = np.r_[np.flatnonzero(np.diff(sorted_linear)), len(sorted_linear) - 1]
    return order[last_in_pixel]


def paint_topdown_points(
    image: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    colors: np.ndarray,
    point_radius: int,
) -> None:
    if px.size == 0:
        return

    height, width = image.shape[:2]
    iy = height - 1 - py
    radius = max(0, int(point_radius))

    if radius <= 0:
        image[iy, px] = colors
        return

    for dy in range(-radius, radius + 1):
        y = iy + dy
        y_mask = (y >= 0) & (y < height)
        if not y_mask.any():
            continue
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            x = px + dx
            mask = y_mask & (x >= 0) & (x < width)
            if mask.any():
                image[y[mask], x[mask]] = colors[mask]


def rasterize_topdown_projection(
    points_2d: np.ndarray,
    depth: np.ndarray,
    rgb: Optional[np.ndarray],
    image_size: int,
    clip_percentile: float,
    color_mode: str,
    keep: str,
    point_radius: int,
    max_points: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, dict, np.ndarray, np.ndarray, np.ndarray, float]:
    if image_size < 16:
        raise ValueError("--visualize_topdown_image_size must be at least 16")
    if not 0 <= clip_percentile < 50:
        raise ValueError("--visualize_topdown_clip_percentile must be in [0, 50)")

    valid, bounds = projection_bounds(points_2d, depth, clip_percentile)
    u_min, u_max, v_min, v_max = bounds
    span_u = max(u_max - u_min, 1e-12)
    span_v = max(v_max - v_min, 1e-12)
    scale = (image_size - 1) / max(span_u, span_v)
    width = max(1, int(np.ceil(span_u * scale)) + 1)
    height = max(1, int(np.ceil(span_v * scale)) + 1)

    valid_indices = np.flatnonzero(valid)
    if max_points > 0 and valid_indices.size > max_points:
        draw_indices = rng.choice(valid_indices, size=max_points, replace=False)
    else:
        draw_indices = valid_indices

    u = points_2d[draw_indices, 0]
    v = points_2d[draw_indices, 1]
    px = np.floor((u - u_min) * scale).astype(np.int64)
    py = np.floor((v - v_min) * scale).astype(np.int64)
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)

    draw_depth = depth[draw_indices]
    selected_local = select_visible_topdown_points(px, py, draw_depth, width, keep)
    selected_global = draw_indices[selected_local]

    resolved_color_mode = color_mode
    if resolved_color_mode == "auto":
        resolved_color_mode = "rgb" if rgb is not None else "height"
    if resolved_color_mode == "rgb" and rgb is None:
        resolved_color_mode = "height"

    if resolved_color_mode == "rgb":
        colors = rgb[selected_global]
    elif resolved_color_mode == "height":
        colors = topdown_height_colors(depth[selected_global])
    else:
        colors = np.full((selected_global.shape[0], 3), 190, dtype=np.uint8)

    image = np.full((height, width, 3), 255, dtype=np.uint8)
    paint_topdown_points(
        image,
        px[selected_local],
        py[selected_local],
        colors,
        point_radius=point_radius,
    )

    sampled_pixels = np.stack([px, height - 1 - py], axis=1).astype(np.int32)
    selected_linear = py[selected_local].astype(np.int64) * width + px[selected_local].astype(np.int64)
    stats = {
        "total_points": int(points_2d.shape[0]),
        "valid_points": int(valid.sum()),
        "rasterized_points": int(draw_indices.shape[0]),
        "selected_points": int(selected_local.shape[0]),
        "drawn_pixels": int(np.unique(selected_linear).shape[0]) if selected_linear.size else 0,
        "width": int(width),
        "height": int(height),
        "bounds": bounds,
        "color_mode": resolved_color_mode,
        "keep": keep,
        "clip_percentile": float(clip_percentile),
    }
    return image, stats, draw_indices, sampled_pixels, np.array([u_min, v_min], dtype=np.float64), scale


def topdown_canvas_geometry(points_2d: np.ndarray, image_size: int):
    mins = points_2d.min(axis=0)
    maxs = points_2d.max(axis=0)
    extent = np.maximum(maxs - mins, 1e-9)
    pad = extent * 0.03
    mins = mins - pad
    maxs = maxs + pad
    extent = np.maximum(maxs - mins, 1e-9)

    long_side = max(float(extent[0]), float(extent[1]))
    scale = max(1.0, float(image_size) / long_side)
    width = max(1, int(round(extent[0] * scale)))
    height = max(1, int(round(extent[1] * scale)))
    return mins, scale, width, height


def world_2d_to_pixels(points_2d: np.ndarray, mins: np.ndarray, scale: float, height: int) -> np.ndarray:
    px = np.empty((points_2d.shape[0], 2), dtype=np.int32)
    px[:, 0] = np.round((points_2d[:, 0] - mins[0]) * scale).astype(np.int32)
    px[:, 1] = height - 1 - np.round((points_2d[:, 1] - mins[1]) * scale).astype(np.int32)
    return px


def bbox_to_topdown_rect(bbox: Sequence[float], axes: Sequence[str], mins: np.ndarray, scale: float, height: int):
    axis_indices = [AXIS_TO_INDEX[axis] for axis in axes]
    bbox_array = np.asarray(bbox, dtype=np.float64)
    rect_points = np.array(
        [
            [bbox_array[axis_indices[0]], bbox_array[axis_indices[1]]],
            [bbox_array[axis_indices[0] + 3], bbox_array[axis_indices[1] + 3]],
        ],
        dtype=np.float64,
    )
    rect_pixels = world_2d_to_pixels(rect_points, mins, scale, height)
    x0, y0 = rect_pixels[0]
    x1, y1 = rect_pixels[1]
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def rasterize_points(canvas: np.ndarray, pixels: np.ndarray, color: Sequence[int], radius: int) -> None:
    if pixels.size == 0:
        return
    radius = max(0, int(radius))
    h, w = canvas.shape[:2]
    base_x = pixels[:, 0]
    base_y = pixels[:, 1]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            xs = base_x + dx
            ys = base_y + dy
            valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            if np.any(valid):
                canvas[ys[valid], xs[valid]] = color


def draw_block_rectangles(draw: ImageDraw.ImageDraw, block: dict, axes: Sequence[str], mins, scale, height, color):
    mode = block.get("_visualize_bbox_mode", "both")
    if mode in {"core", "both"}:
        rect = bbox_to_topdown_rect(block["core_bbox"], axes, mins, scale, height)
        draw.rectangle(rect, outline=tuple(color), width=3)
    if mode in {"expanded", "both"}:
        rect = bbox_to_topdown_rect(block["expanded_bbox"], axes, mins, scale, height)
        draw.rectangle(rect, outline=(0, 180, 220), width=2)


def visualize_partition_topdown(
    cfg: ExperimentConfig,
    blocks: Sequence[dict],
    point_cloud,
    partition_output: str,
) -> List[dict]:
    partition = cfg.partition
    vis = cfg.visualization
    if point_cloud is None:
        raise ValueError("Scene has no original point cloud for top-down visualization")
    if len(partition.axes) < 2:
        raise ValueError("Top-down visualization requires at least two partition axes")

    axes = partition.axes[:2]
    axis_indices = [AXIS_TO_INDEX[axis] for axis in axes]
    depth_axis = projection_depth_axis(axes)
    depth_axis_index = AXIS_TO_INDEX[depth_axis]
    original_points = np.asarray(point_cloud.points)
    points_2d = original_points[:, axis_indices]
    depth = original_points[:, depth_axis_index]
    rgb = point_cloud_rgb(point_cloud, original_points.shape[0])
    rng = np.random.RandomState(vis.visualize_random_seed)
    output_root = (
        os.path.abspath(vis.visualize_topdown_output)
        if vis.visualize_topdown_output
        else os.path.join(partition_output, "visualizations_topdown")
    )
    os.makedirs(output_root, exist_ok=True)
    partition_points = original_points
    if partition.coord_space == "contracted":
        partition_points = contract_to_unisphere(original_points, partition.contract_aabb, ord=np.inf)

    background, topdown_stats, sampled_indices, sampled_pixels, mins, scale = rasterize_topdown_projection(
        points_2d=points_2d,
        depth=depth,
        rgb=rgb,
        image_size=vis.visualize_topdown_image_size,
        clip_percentile=vis.visualize_topdown_clip_percentile,
        color_mode=vis.visualize_topdown_color,
        keep=vis.visualize_topdown_keep,
        point_radius=vis.visualize_topdown_point_radius,
        max_points=vis.visualize_topdown_max_points,
        rng=rng,
    )
    height, width = background.shape[:2]
    sampled_partition_points = partition_points[sampled_indices]
    contracted_coloring = partition.coord_space == "contracted"
    if contracted_coloring:
        global_canvas = background.copy()
        global_labels = []
    else:
        global_image = Image.fromarray(background)
        global_draw = ImageDraw.Draw(global_image)

    reports = []
    for block_index, block in enumerate(blocks):
        color = block_color(block_index).astype(np.uint8).tolist()
        bbox_for_points = block["core_bbox"] if contracted_coloring else (
            block["expanded_bbox"] if vis.visualize_topdown_bbox_mode == "expanded" else block["core_bbox"]
        )
        block_mask = points_in_bbox(sampled_partition_points, bbox_for_points)
        block_pixels = sampled_pixels[block_mask]
        if contracted_coloring:
            rasterize_points(global_canvas, block_pixels, color, vis.visualize_topdown_point_radius)
            global_labels.append((block["id"], tuple(color)))
            block_canvas = background.copy()
            rasterize_points(block_canvas, block_pixels, color, vis.visualize_topdown_point_radius)
            block_image = Image.fromarray(block_canvas)
            block_draw = ImageDraw.Draw(block_image)
            block_draw.text((10, 10), block["id"], fill=tuple(color))
            bbox_mode = "core_point_membership"
        else:
            block_for_draw = dict(block)
            block_for_draw["_visualize_bbox_mode"] = vis.visualize_topdown_bbox_mode
            draw_block_rectangles(global_draw, block_for_draw, axes, mins, scale, height, color)
            global_draw.text((10, 10 + 18 * block_index), block["id"], fill=tuple(color))
            block_canvas = background.copy()
            rasterize_points(block_canvas, block_pixels, color, vis.visualize_topdown_point_radius)
            block_image = Image.fromarray(block_canvas)
            block_draw = ImageDraw.Draw(block_image)
            draw_block_rectangles(block_draw, block_for_draw, axes, mins, scale, height, color)
            block_draw.text((10, 10), block["id"], fill=tuple(color))
            bbox_mode = vis.visualize_topdown_bbox_mode

        block_output = os.path.join(output_root, f"{block['id']}.png")
        block_image.save(block_output)
        reports.append(
            {
                "block_id": block["id"],
                "output_path": block_output,
                "partition_axes": axes,
                "depth_axis": depth_axis,
                "bbox_mode": bbox_mode,
                "num_sampled_points": int(sampled_partition_points.shape[0]),
                "num_block_points_drawn": int(block_mask.sum()),
            }
        )

    if contracted_coloring:
        global_image = Image.fromarray(global_canvas)
        global_draw = ImageDraw.Draw(global_image)
        for block_index, (block_id, color) in enumerate(global_labels):
            global_draw.text((10, 10 + 18 * block_index), block_id, fill=color)

    global_output = os.path.join(output_root, "global_blocks.png")
    global_image.save(global_output)
    save_json(
        os.path.join(output_root, "visualization_report.json"),
        {
            "mode": "topdown_original_space_contracted_membership" if contracted_coloring else "topdown_original_point_cloud",
            "membership_coord_space": partition.coord_space,
            "raster_coord_space": "world",
            "partition_axes": axes,
            "depth_axis": depth_axis,
            "global_output": global_output,
            "num_original_points": int(original_points.shape[0]),
            "num_sampled_points": int(sampled_partition_points.shape[0]),
            "rasterization": topdown_stats,
            "blocks": reports,
        },
    )
    return reports


def assign_cameras_to_block(
    block: dict,
    train_cameras: Sequence,
    tau_projection: float,
    min_cameras: int,
    supplement_cameras: bool,
    candidate_points: Optional[np.ndarray] = None,
    candidate_weights: Optional[np.ndarray] = None,
    render_assigner: Optional[RenderDifferenceAssigner] = None,
    render_difference_threshold: float = 0.03,
    max_render_candidates: int = 0,
    projection_max_points: int = 5000,
    rng: Optional[np.random.RandomState] = None,
    log: Optional[Callable[[str], None]] = None,
    log_prefix: str = "",
    log_interval: int = PARTITION_LOG_INTERVAL,
):
    core_bbox = block["core_bbox"]
    expanded_bbox = block["expanded_bbox"]
    center = bbox_center(expanded_bbox)
    selected = {}
    candidates = {}
    records = []
    if rng is None:
        rng = np.random.RandomState(0)

    total_cameras = len(train_cameras)
    for cam_idx, cam in enumerate(train_cameras, start=1):
        cam_center = camera_center_from_info(cam)
        cam_partition_center = partition_points(cam_center[None, :], block)[0]
        inside_core = bool(points_in_bbox(cam_partition_center[None, :], core_bbox)[0])
        inside = bool(points_in_bbox(cam_partition_center[None, :], expanded_bbox)[0])
        if candidate_points is not None:
            coverage = project_point_coverage(
                cam,
                candidate_points,
                weights=candidate_weights,
                max_points=projection_max_points,
                rng=rng,
            )
        else:
            coverage = project_bbox_coverage(cam, expanded_bbox)
        distance = float(np.linalg.norm(cam_partition_center - center))
        records.append(
            {
                "cam": cam,
                "inside_core": inside_core,
                "inside": inside,
                "coverage": coverage,
                "distance": distance,
            }
        )
        if coverage > 0.0:
            candidates[cam.image_name] = {
                "source": "projection_candidate",
                "center_inside_core_bbox": inside_core,
                "center_inside_assignment_bbox": inside,
                "projection_coverage": coverage,
                "render_difference": None,
                "final_score": coverage,
            }

        if inside:
            source = "center_inside_bbox"
            if coverage >= tau_projection:
                source = "center_inside_and_projected_bbox"
            selected[cam.image_name] = {
                "source": source,
                "center_inside_core_bbox": inside_core,
                "center_inside_assignment_bbox": inside,
                "projection_coverage": coverage,
                "render_difference": None,
                "final_score": coverage,
            }

        if log is not None and log_interval > 0 and (cam_idx % log_interval == 0 or cam_idx == total_cameras):
            log(
                f"[{log_prefix}] projection scan {cam_idx}/{total_cameras} "
                f"selected={len(selected)} candidates={len(candidates)}"
            )

    if render_assigner is not None:
        selected = {}
        render_candidates = [
            record for record in records
            if record["inside"] or record["coverage"] > 0.0
        ]
        render_candidates = sorted(
            render_candidates,
            key=lambda record: (not record["inside"], -record["coverage"], record["distance"]),
        )
        if max_render_candidates > 0:
            render_candidates = render_candidates[:max_render_candidates]

        if log is not None:
            log(f"[{log_prefix}] render difference candidates={len(render_candidates)}")

        for render_idx, record in enumerate(render_candidates, start=1):
            cam = record["cam"]
            score = render_assigner.score(block, cam)
            candidates[cam.image_name] = {
                "source": "render_difference_candidate",
                "center_inside_core_bbox": record["inside_core"],
                "center_inside_assignment_bbox": record["inside"],
                "projection_coverage": record["coverage"],
                "render_difference": score,
                "final_score": score,
            }
            if score >= render_difference_threshold:
                selected[cam.image_name] = {
                    "source": "render_difference",
                    "center_inside_core_bbox": record["inside_core"],
                    "center_inside_assignment_bbox": record["inside"],
                    "projection_coverage": record["coverage"],
                    "render_difference": score,
                    "final_score": score,
                }

            if log is not None and log_interval > 0 and (
                render_idx % log_interval == 0 or render_idx == len(render_candidates)
            ):
                log(
                    f"[{log_prefix}] render scoring {render_idx}/{len(render_candidates)} "
                    f"selected={len(selected)}"
                )

        if len(selected) < min_cameras:
            projection_supplements = sorted(
                (
                    record
                    for record in records
                    if (record["coverage"] > 0.0 or record["inside"]) and record["cam"].image_name not in selected
                ),
                key=lambda record: (not record["inside"], -record["coverage"], record["distance"]),
            )
            for record in projection_supplements:
                cam = record["cam"]
                score_info = candidates.get(cam.image_name, {})
                selected[cam.image_name] = {
                    "source": "min_cameras_projection_fallback",
                    "center_inside_core_bbox": record["inside_core"],
                    "center_inside_assignment_bbox": record["inside"],
                    "projection_coverage": record["coverage"],
                    "render_difference": score_info.get("render_difference"),
                    "final_score": score_info.get("render_difference", record["coverage"]),
                }
                if len(selected) >= min_cameras:
                    break

        ordered_train_cameras = [cam.image_name for cam in train_cameras if cam.image_name in selected]
        ordered_candidate_cameras = [cam.image_name for cam in train_cameras if cam.image_name in candidates]
        return ordered_train_cameras, ordered_candidate_cameras, selected, camera_assignment_summary(records, selected, candidates)

    if supplement_cameras:
        for record in records:
            cam = record["cam"]
            if cam.image_name in selected or record["inside"] or record["coverage"] < tau_projection:
                continue
            selected[cam.image_name] = {
                "source": "projected_bbox",
                "center_inside_core_bbox": record["inside_core"],
                "center_inside_assignment_bbox": record["inside"],
                "projection_coverage": record["coverage"],
                "render_difference": None,
                "final_score": record["coverage"],
            }

        if len(selected) < min_cameras:
            projection_supplements = sorted(
                (
                    record
                    for record in records
                    if record["coverage"] > 0.0 and record["cam"].image_name not in selected
                ),
                key=lambda record: (-record["coverage"], record["distance"]),
            )
            for record in projection_supplements:
                cam = record["cam"]
                selected[cam.image_name] = {
                    "source": "min_cameras_projected",
                    "center_inside_core_bbox": record["inside_core"],
                    "center_inside_assignment_bbox": record["inside"],
                    "projection_coverage": record["coverage"],
                    "render_difference": None,
                    "final_score": record["coverage"],
                }
                if len(selected) >= min_cameras:
                    break

    ordered_train_cameras = [cam.image_name for cam in train_cameras if cam.image_name in selected]
    ordered_candidate_cameras = [cam.image_name for cam in train_cameras if cam.image_name in candidates]
    return ordered_train_cameras, ordered_candidate_cameras, selected, camera_assignment_summary(records, selected, candidates)


def build_partition_tree(
    cfg: ExperimentConfig,
    xyz: np.ndarray,
    weights: np.ndarray,
    log: Optional[Callable[[str], None]] = None,
):
    partition = cfg.partition
    max_block_importance = partition.max_block_importance
    total_importance = float(weights.sum())
    if max_block_importance <= 0:
        max_block_importance = total_importance / max(partition.max_blocks, 1)

    root_bbox = root_bbox_from_points(xyz)
    all_indices = np.arange(xyz.shape[0])
    split_events = []
    leaf_nodes = []

    def recurse(node_id: str, bbox: List[float], indices: np.ndarray, depth: int, path: List[str]):
        importance = float(weights[indices].sum()) if indices.size else 0.0
        density = block_density(importance, bbox, partition.axes)
        node = {
            "id": node_id,
            "depth": depth,
            "bbox": [float(value) for value in bbox],
            "importance": importance,
            "importance_density": density,
            "num_coarse_gaussians": int(indices.size),
        }

        stop_reason = None
        if depth >= partition.max_depth:
            stop_reason = "max_depth"
        elif indices.size <= partition.min_points:
            stop_reason = "min_points"
        elif importance <= max_block_importance and (partition.max_block_density <= 0 or density <= partition.max_block_density):
            stop_reason = "importance_below_threshold"

        axis = None
        position = None
        split_record = None
        if stop_reason is None:
            axis = choose_split_axis(xyz, weights, indices, bbox, partition.axes, partition.min_size)
            if axis is None:
                stop_reason = "no_valid_axis"
            else:
                position, split_record = choose_split_position(
                    xyz,
                    weights,
                    indices,
                    axis,
                    partition.num_split_candidates,
                    partition.lambda_boundary,
                    partition.min_points,
                )
                if position is None:
                    stop_reason = "no_valid_split"

        if stop_reason is not None:
            node["is_leaf"] = True
            node["stop_reason"] = stop_reason
            node["split_path"] = path + [node_id]
            leaf_nodes.append((node, indices))
            if log is not None:
                log(
                    f"[tree] leaf node={node_id} depth={depth} points={indices.size} "
                    f"importance={importance:.6f} density={density:.6f} reason={stop_reason}"
                )
            return node

        axis_idx = AXIS_TO_INDEX[axis]
        left_mask = xyz[indices, axis_idx] <= position
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]
        left_bbox = list(bbox)
        right_bbox = list(bbox)
        left_bbox[axis_idx + 3] = float(position)
        right_bbox[axis_idx] = float(position)

        event = {
            "node_id": node_id,
            "depth": depth,
            "num_points": int(indices.size),
            "importance": importance,
            "importance_density": density,
            "split_axis": axis,
            "split_position": float(position),
            **split_record,
        }
        split_events.append(event)
        if log is not None:
            log(
                f"[tree] split node={node_id} depth={depth} axis={axis} pos={float(position):.6f} "
                f"points={indices.size} importance={importance:.6f} density={density:.6f} "
                f"left={split_record['left_points']} right={split_record['right_points']} "
                f"score={split_record['score']:.6f}"
            )

        node.update(
            {
                "is_leaf": False,
                "split_axis": axis,
                "split_position": float(position),
                "split_score": split_record["score"],
                "children": [
                    recurse(f"{node_id}_l", left_bbox, left_indices, depth + 1, path + [node_id]),
                    recurse(f"{node_id}_r", right_bbox, right_indices, depth + 1, path + [node_id]),
                ],
            }
        )
        return node

    root = recurse("root", root_bbox, all_indices, 0, [])
    return root, leaf_nodes, split_events, max_block_importance


def write_text_list(path: str, values: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for value in values:
            f.write(f"{value}\n")


def main():
    log = make_progress_logger()
    cfg = parse_config()
    dataset = cfg.dataset
    partition = cfg.partition
    camera_cfg = cfg.camera_assignment
    render_diff = camera_cfg.render_difference
    vis = cfg.visualization
    log(f"start config={cfg.config_path or '<legacy args>'}")

    source_path = os.path.abspath(dataset.source_path)
    test_source_path = os.path.abspath(dataset.test_source_path) if dataset.test_source_path else ""
    coarse_model = os.path.abspath(cfg.model.coarse_model)
    cfg.model.coarse_model = coarse_model
    partition_output = os.path.abspath(partition.output_path)
    partition.output_path = partition_output
    os.makedirs(partition_output, exist_ok=True)
    log(f"output={partition_output}")

    log(f"loading train scene: source={source_path}")
    scene_info = load_scene_info(cfg)
    train_cameras = all_camera_infos(scene_info)
    sparse_count_for_log = len(scene_info.point_cloud.points) if scene_info.point_cloud is not None else 0
    log(
        f"loaded train scene: cameras={len(train_cameras)} "
        f"sparse_points={sparse_count_for_log}"
    )
    test_scene_info = None
    test_cameras_pool = []
    if test_source_path:
        log(f"loading external test scene: source={test_source_path}")
        test_scene_info = load_scene_info_from_path(
            test_source_path,
            dataset.test_images or dataset.images,
            dataset.test_depths,
            False,
            dataset.train_test_exp,
            dataset.white_background,
        )
        test_cameras_pool = all_camera_infos(test_scene_info)
        log(f"loaded external test cameras: {len(test_cameras_pool)}")

    log(f"loading coarse model: {coarse_model}")
    coarse = read_gaussian_or_point_ply(coarse_model)
    weights = compute_importance(coarse, partition.importance)
    log(
        f"loaded coarse model: gaussians={coarse['xyz'].shape[0]} "
        f"importance_mode={partition.importance} total_importance={float(weights.sum()):.6f}"
    )
    contract_aabb = None
    if partition.coord_space == "contracted":
        contract_aabb = partition.contract_aabb if partition.contract_aabb is not None else root_bbox_from_points(coarse["xyz"])
        partition.contract_aabb = contract_aabb
        log(f"using contract aabb={contract_aabb}")
        partition_xyz = contract_to_unisphere(coarse["xyz"], contract_aabb, ord=np.inf)
    else:
        log("using world-space partition coordinates")
        partition_xyz = coarse["xyz"]
    save_yaml_config(os.path.join(partition_output, "resolved_config.yaml"), config_to_dict(cfg))
    log(
        f"building partition tree: points={partition_xyz.shape[0]} "
        f"axes={partition.axes} max_blocks={partition.max_blocks} max_depth={partition.max_depth}"
    )
    root, leaf_nodes, split_events, max_block_importance = build_partition_tree(
        cfg,
        partition_xyz,
        weights,
        log=log,
    )
    log(
        f"build tree done: leaves={len(leaf_nodes)} splits={len(split_events)} "
        f"derived_max_block_importance={max_block_importance:.6f}"
    )
    render_assigner = RenderDifferenceAssigner(cfg, coarse["xyz"]) if render_diff.enabled else None
    if render_assigner is not None:
        log(
            f"render-difference camera assignment enabled: threshold={render_diff.threshold} "
            f"max_candidates={render_diff.max_candidates_per_block}"
        )
    rng = np.random.RandomState(vis.visualize_random_seed)

    blocks = []
    camera_assignment_events = []
    sparse_points = np.asarray(scene_info.point_cloud.points) if scene_info.point_cloud is not None else None
    sparse_partition_points = None
    if sparse_points is not None:
        if partition.coord_space == "contracted":
            sparse_partition_points = contract_to_unisphere(sparse_points, contract_aabb, ord=np.inf)
        else:
            sparse_partition_points = sparse_points

    log(f"assigning cameras for {len(leaf_nodes)} blocks")
    for block_idx, (leaf, indices) in enumerate(leaf_nodes):
        block_id = f"block_{block_idx:03d}"
        leaf["block_id"] = block_id
        core_bbox = leaf["bbox"]
        expanded_bbox = expand_bbox(core_bbox, partition.expand_ratio)
        sparse_count = int(points_in_bbox(sparse_partition_points, expanded_bbox).sum()) if sparse_partition_points is not None else 0
        expanded_mask = points_in_bbox(partition_xyz, expanded_bbox)
        num_expanded_coarse_gaussians = int(expanded_mask.sum())
        expanded_indices = np.flatnonzero(expanded_mask)
        expanded_indices = sample_indices(
            expanded_indices,
            weights,
            camera_cfg.projection_max_points,
            rng,
        )
        candidate_points = coarse["xyz"][expanded_indices] if expanded_indices.size > 0 else coarse["xyz"][indices]
        candidate_weights = weights[expanded_indices] if expanded_indices.size > 0 else weights[indices]
        log(
            f"[block {block_idx + 1}/{len(leaf_nodes)}] {block_id} start "
            f"node={leaf['id']} depth={leaf['depth']} "
            f"core_gaussians={leaf['num_coarse_gaussians']} "
            f"expanded_gaussians={num_expanded_coarse_gaussians} "
            f"sampled_candidates={candidate_points.shape[0]} sparse_points={sparse_count}"
        )
        block = {
            "id": block_id,
            "node_id": leaf["id"],
            "parent": leaf["id"].rsplit("_", 1)[0] if "_" in leaf["id"] else None,
            "depth": leaf["depth"],
            "core_bbox": core_bbox,
            "expanded_bbox": expanded_bbox,
            "partition_coord_space": partition.coord_space,
            "contract_aabb": contract_aabb,
            "importance": leaf["importance"],
            "importance_density": leaf["importance_density"],
            "num_coarse_gaussians": leaf["num_coarse_gaussians"],
            "num_expanded_coarse_gaussians": num_expanded_coarse_gaussians,
            "num_sparse_points": sparse_count,
            "trace": {
                "split_path": leaf["split_path"],
                "stop_reason": leaf["stop_reason"],
            },
        }
        log(f"[block {block_idx + 1}/{len(leaf_nodes)}] {block_id} assigning train cameras: pool={len(scene_info.train_cameras)}")
        train_cameras, candidate_cameras, camera_scores, train_camera_summary = assign_cameras_to_block(
            block,
            scene_info.train_cameras,
            camera_cfg.tau_projection,
            camera_cfg.min_cameras,
            camera_cfg.supplement_cameras,
            candidate_points=candidate_points,
            candidate_weights=candidate_weights,
            render_assigner=render_assigner,
            render_difference_threshold=render_diff.threshold,
            max_render_candidates=render_diff.max_candidates_per_block,
            projection_max_points=camera_cfg.projection_max_points,
            rng=rng,
            log=log,
            log_prefix=f"{block_id} train",
        )
        block["train_cameras"] = train_cameras
        block["candidate_cameras"] = candidate_cameras
        block["camera_scores"] = camera_scores
        block["train_camera_assignment"] = train_camera_summary
        block["num_train_camera_pool_in_block"] = train_camera_summary["pool_in_block_cameras"]
        block["num_train_cameras_in_block"] = train_camera_summary["selected_in_block_cameras"]
        block["num_train_cameras_added"] = train_camera_summary["selected_added_cameras"]
        log(
            f"[block {block_idx + 1}/{len(leaf_nodes)}] {block_id} train cameras done: "
            f"selected={len(train_cameras)} candidates={len(candidate_cameras)} "
            f"in_block={train_camera_summary['selected_in_block_cameras']} "
            f"added={train_camera_summary['selected_added_cameras']} "
            f"sources={format_counts(score_source_counts(camera_scores))}"
        )
        if test_cameras_pool:
            log(f"[block {block_idx + 1}/{len(leaf_nodes)}] {block_id} assigning test cameras: pool={len(test_cameras_pool)}")
            test_cameras, candidate_test_cameras, test_camera_scores, test_camera_summary = assign_cameras_to_block(
                block,
                test_cameras_pool,
                camera_cfg.tau_test_projection,
                camera_cfg.min_test_cameras,
                camera_cfg.supplement_cameras,
                candidate_points=candidate_points,
                candidate_weights=candidate_weights,
                render_assigner=render_assigner,
                render_difference_threshold=render_diff.threshold,
                max_render_candidates=render_diff.max_candidates_per_block,
                projection_max_points=camera_cfg.projection_max_points,
                rng=rng,
                log=log,
                log_prefix=f"{block_id} test",
            )
            block["test_cameras"] = test_cameras
            block["candidate_test_cameras"] = candidate_test_cameras
            block["test_camera_scores"] = test_camera_scores
            block["test_camera_assignment"] = test_camera_summary
            block["num_test_camera_pool_in_block"] = test_camera_summary["pool_in_block_cameras"]
            block["num_test_cameras_in_block"] = test_camera_summary["selected_in_block_cameras"]
            block["num_test_cameras_added"] = test_camera_summary["selected_added_cameras"]
            log(
                f"[block {block_idx + 1}/{len(leaf_nodes)}] {block_id} test cameras done: "
                f"selected={len(test_cameras)} candidates={len(candidate_test_cameras)} "
                f"in_block={test_camera_summary['selected_in_block_cameras']} "
                f"added={test_camera_summary['selected_added_cameras']} "
                f"sources={format_counts(score_source_counts(test_camera_scores))}"
            )
        else:
            test_camera_summary = camera_assignment_summary([], {}, {})
            block["test_cameras"] = []
            block["candidate_test_cameras"] = []
            block["test_camera_scores"] = {}
            block["test_camera_assignment"] = test_camera_summary
            block["num_test_camera_pool_in_block"] = 0
            block["num_test_cameras_in_block"] = 0
            block["num_test_cameras_added"] = 0
        blocks.append(block)
        log(f"[block {block_idx + 1}/{len(leaf_nodes)}] {block_id} done")

        camera_assignment_events.append(
            {
                "block_id": block_id,
                "num_train_cameras": len(train_cameras),
                "num_train_camera_pool_in_block": train_camera_summary["pool_in_block_cameras"],
                "num_train_cameras_in_block": train_camera_summary["selected_in_block_cameras"],
                "num_train_cameras_added": train_camera_summary["selected_added_cameras"],
                "train_camera_assignment": train_camera_summary,
                "num_candidate_cameras": len(candidate_cameras),
                "num_test_cameras": len(block["test_cameras"]),
                "num_test_camera_pool_in_block": test_camera_summary["pool_in_block_cameras"],
                "num_test_cameras_in_block": test_camera_summary["selected_in_block_cameras"],
                "num_test_cameras_added": test_camera_summary["selected_added_cameras"],
                "test_camera_assignment": test_camera_summary,
                "num_candidate_test_cameras": len(block["candidate_test_cameras"]),
                "camera_scores": camera_scores,
                "test_camera_scores": block["test_camera_scores"],
            }
        )

    tree = {
        "schema_version": 1,
        "method": "coarse_gs_importance_recursive_partitioning",
        "source_path": source_path,
        "test_source_path": test_source_path,
        "coarse_model": coarse_model,
        "partition_coord_space": partition.coord_space,
        "contract_aabb": contract_aabb,
        "partition_axes": partition.axes,
        "config": {
            **config_to_dict(cfg),
            "source_path": source_path,
            "test_source_path": test_source_path,
            "test_images": dataset.test_images or dataset.images,
            "coarse_model": coarse_model,
            "partition_output": partition_output,
            "derived_max_block_importance": max_block_importance,
        },
        "root": root,
        "blocks": blocks,
    }

    save_json(os.path.join(partition_output, "partition_config.json"), tree["config"])
    save_json(os.path.join(partition_output, "partition_tree.json"), tree)

    traces_dir = os.path.join(partition_output, "traces")
    os.makedirs(traces_dir, exist_ok=True)
    with open(os.path.join(traces_dir, "split_events.jsonl"), "w") as f:
        for event in split_events:
            f.write(json.dumps(event) + "\n")
    with open(os.path.join(traces_dir, "camera_assignment.jsonl"), "w") as f:
        for event in camera_assignment_events:
            f.write(json.dumps(event) + "\n")

    block_stats = {
        "num_blocks": len(blocks),
        "total_importance": float(weights.sum()),
        "max_block_importance": max(block["importance"] for block in blocks) if blocks else 0.0,
        "min_block_importance": min(block["importance"] for block in blocks) if blocks else 0.0,
        "max_train_cameras": max(len(block["train_cameras"]) for block in blocks) if blocks else 0,
        "min_train_cameras": min(len(block["train_cameras"]) for block in blocks) if blocks else 0,
        "max_train_cameras_in_block": max(block["num_train_cameras_in_block"] for block in blocks) if blocks else 0,
        "min_train_cameras_in_block": min(block["num_train_cameras_in_block"] for block in blocks) if blocks else 0,
        "max_train_cameras_added": max(block["num_train_cameras_added"] for block in blocks) if blocks else 0,
        "min_train_cameras_added": min(block["num_train_cameras_added"] for block in blocks) if blocks else 0,
        "max_test_cameras": max(len(block["test_cameras"]) for block in blocks) if blocks else 0,
        "min_test_cameras": min(len(block["test_cameras"]) for block in blocks) if blocks else 0,
        "max_test_cameras_in_block": max(block["num_test_cameras_in_block"] for block in blocks) if blocks else 0,
        "min_test_cameras_in_block": min(block["num_test_cameras_in_block"] for block in blocks) if blocks else 0,
        "max_test_cameras_added": max(block["num_test_cameras_added"] for block in blocks) if blocks else 0,
        "min_test_cameras_added": min(block["num_test_cameras_added"] for block in blocks) if blocks else 0,
        "blocks": [
            {
                "id": block["id"],
                "depth": block["depth"],
                "importance": block["importance"],
                "importance_density": block["importance_density"],
                "num_coarse_gaussians": block["num_coarse_gaussians"],
                "num_expanded_coarse_gaussians": block["num_expanded_coarse_gaussians"],
                "num_sparse_points": block["num_sparse_points"],
                "num_train_cameras": len(block["train_cameras"]),
                "num_train_camera_pool_in_block": block["num_train_camera_pool_in_block"],
                "num_train_cameras_in_block": block["num_train_cameras_in_block"],
                "num_train_cameras_added": block["num_train_cameras_added"],
                "train_camera_assignment": block["train_camera_assignment"],
                "num_test_cameras": len(block["test_cameras"]),
                "num_test_camera_pool_in_block": block["num_test_camera_pool_in_block"],
                "num_test_cameras_in_block": block["num_test_cameras_in_block"],
                "num_test_cameras_added": block["num_test_cameras_added"],
                "test_camera_assignment": block["test_camera_assignment"],
            }
            for block in blocks
        ],
    }
    save_json(os.path.join(traces_dir, "block_stats.json"), block_stats)

    blocks_dir = os.path.join(partition_output, "blocks")
    for block in blocks:
        block_dir = os.path.join(blocks_dir, block["id"])
        save_json(os.path.join(block_dir, "metadata.json"), block)
        write_text_list(os.path.join(block_dir, "train_cameras.txt"), block["train_cameras"])
        write_text_list(os.path.join(block_dir, "candidate_cameras.txt"), block["candidate_cameras"])
        write_text_list(os.path.join(block_dir, "test_cameras.txt"), block["test_cameras"])
        write_text_list(os.path.join(block_dir, "candidate_test_cameras.txt"), block["candidate_test_cameras"])

    if vis.visualize_blocks:
        reports = visualize_partition_blocks(
            cfg,
            blocks,
            scene_info.train_cameras,
            coarse["xyz"],
            partition_xyz,
            weights,
            partition_output,
        )
        num_images = sum(len(report["images"]) for report in reports)
        print(f"Block visualizations written under {vis.visualize_output or os.path.join(partition_output, 'visualizations')}")
        print(f"Visualization images: {num_images}")

    if vis.visualize_topdown:
        reports = visualize_partition_topdown(cfg, blocks, scene_info.point_cloud, partition_output)
        print(f"Top-down point cloud visualizations written under {vis.visualize_topdown_output or os.path.join(partition_output, 'visualizations_topdown')}")
        print(f"Top-down block images: {len(reports)}")

    print(f"Partition written to {partition_output}")
    print(f"Blocks: {len(blocks)}")
    print(f"Total coarse importance: {weights.sum():.6f}")
    print(f"Derived max block importance: {max_block_importance:.6f}")


if __name__ == "__main__":
    main()
