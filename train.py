#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import math
import torch
import time
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from scene.datasets import CameraDataLoader, GSCameraDataset
from utils.block_depth_mask import BlockDepthMasker
from utils.general_utils import safe_state, get_expon_lr_func
from utils.pseudo_view import make_pseudo_view_camera, pseudo_view_reprojection_loss, warp_pseudo_normal_to_ref
from utils.config_utils import (
    load_yaml_config,
    namespace_from_config,
    save_yaml_config,
    stage_args_from_config,
)
import uuid
from tqdm import tqdm
from utils.image_utils import image_to_cuda_float, psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    import swanlab
    SWANLAB_FOUND = True
except ImportError:
    swanlab = None
    SWANLAB_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


class SwanLabLogger:
    def __init__(self, enabled):
        self.enabled = enabled

    def __bool__(self):
        return self.enabled

    def add_scalar(self, tag, scalar_value, global_step=None):
        if self.enabled:
            swanlab.log({tag: float(scalar_value)}, step=global_step)

    def add_images(self, tag, images, global_step=None):
        if self.enabled:
            swanlab.log({tag: swanlab.Image(images.detach().cpu(), caption=tag)}, step=global_step)

    def add_histogram(self, tag, values, global_step=None):
        if not self.enabled:
            return
        if not isinstance(values, torch.Tensor):
            values = torch.as_tensor(values)
        values = values.detach().float().flatten().cpu()
        if values.numel() == 0:
            return
        min_val = float(values.min())
        max_val = float(values.max())
        if min_val == max_val:
            min_val -= 0.5
            max_val += 0.5
        bins = min(50, max(1, values.numel()))
        hist = torch.histc(values, bins=bins, min=min_val, max=max_val)
        edges = torch.linspace(min_val, max_val, bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        chart = swanlab.echarts.Bar()
        chart.add_xaxis([f"{center:.4f}" for center in centers.tolist()])
        chart.add_yaxis("count", [int(count) for count in hist.tolist()])
        swanlab.log({tag: chart}, step=global_step)

    def close(self):
        if self.enabled:
            swanlab.finish()


def make_validation_dataset(scene, dataset_args, camera_infos, is_test_dataset, indices=None):
    if not camera_infos:
        return []
    if indices is not None:
        camera_infos = [camera_infos[idx] for idx in indices]
    camera_dataset = GSCameraDataset(
        camera_infos,
        dataset_args,
        scene.is_nerf_synthetic,
        is_test_dataset=is_test_dataset,
    )
    return camera_dataset


def depth_tensor_to_image(depth, mask=None):
    if depth is None:
        return None
    depth = depth.detach().float()
    while depth.ndim > 3:
        depth = depth[0]
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
    if depth.ndim != 3:
        return None
    if depth.shape[0] != 1:
        depth = depth[:1]

    valid = torch.isfinite(depth)
    if mask is not None:
        mask = mask.detach().to(device=depth.device).float()
        while mask.ndim > 3:
            mask = mask[0]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3 and mask.shape[0] != 1:
            mask = mask[:1]
        if mask.shape == depth.shape:
            valid = valid & (mask > 0)

    if not bool(valid.any().item()):
        return torch.zeros((3, depth.shape[-2], depth.shape[-1]), device=depth.device, dtype=depth.dtype)

    values = depth[valid]
    min_value = values.min()
    max_value = values.max()
    value_range = max_value - min_value
    if not bool(torch.isfinite(value_range).item()) or float(value_range.abs().item()) < 1e-8:
        return torch.zeros((3, depth.shape[-2], depth.shape[-1]), device=depth.device, dtype=depth.dtype)

    image = ((depth - min_value) / value_range).clamp(0.0, 1.0)
    image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    image[~valid] = 0.0
    return image.expand(3, -1, -1).contiguous()


def log_validation_depth_images(logger, tag_prefix, iteration, render_pkg, viewpoint, log_gt_depth):
    rendered_invdepth = render_pkg.get("depth")
    if rendered_invdepth is None:
        return

    rendered_vis = depth_tensor_to_image(rendered_invdepth)
    if rendered_vis is not None:
        logger.add_images(f"{tag_prefix}/render_inv_depth", rendered_vis[None], global_step=iteration)

    if not getattr(viewpoint, "depth_reliable", False) or viewpoint.invdepthmap is None:
        return

    gt_invdepth = viewpoint.invdepthmap.to(rendered_invdepth.device)
    depth_mask = viewpoint.depth_mask.to(rendered_invdepth.device) if viewpoint.depth_mask is not None else None

    if log_gt_depth:
        gt_vis = depth_tensor_to_image(gt_invdepth, depth_mask)
        if gt_vis is not None:
            logger.add_images(f"{tag_prefix}/gt_inv_depth", gt_vis[None], global_step=iteration)

    invdepth_error = torch.abs(rendered_invdepth.detach() - gt_invdepth)
    if depth_mask is not None:
        invdepth_error = invdepth_error * depth_mask
    error_vis = depth_tensor_to_image(invdepth_error, depth_mask)
    if error_vis is not None:
        logger.add_images(f"{tag_prefix}/inv_depth_error", error_vis[None], global_step=iteration)


def normal_tensor_to_image(normal):
    if normal is None:
        return None
    normal = normal.detach().float()
    while normal.ndim > 3:
        normal = normal[0]
    if normal.ndim != 3 or normal.shape[0] < 3:
        return None
    return (normal[:3] * 0.5 + 0.5).clamp(0.0, 1.0).contiguous()


def world_normal_to_camera(normal, viewpoint, sign=-1.0):
    rotation = viewpoint.world_view_transform[:3, :3].to(device=normal.device, dtype=normal.dtype)
    normal_cam = torch.einsum("ji,jhw->ihw", rotation, normal)
    normal_cam = float(sign) * normal_cam
    return torch.nn.functional.normalize(normal_cam, p=2, dim=0, eps=1e-6)


def prepare_gt_normal(viewpoint):
    if getattr(viewpoint, "normalmap", None) is None:
        return None, None

    gt_normal = viewpoint.normalmap.to("cuda", non_blocking=True).float()
    gt_normal = torch.nan_to_num(gt_normal, nan=0.0, posinf=0.0, neginf=0.0)
    gt_norm = torch.linalg.norm(gt_normal, dim=0, keepdim=True)
    valid = torch.isfinite(gt_normal).all(dim=0, keepdim=True) & (gt_norm > 0.5)

    if viewpoint.normal_mask is not None:
        normal_mask = viewpoint.normal_mask.to("cuda", non_blocking=True).float()
        while normal_mask.ndim > 3:
            normal_mask = normal_mask[0]
        if normal_mask.ndim == 2:
            normal_mask = normal_mask.unsqueeze(0)
        if normal_mask.ndim == 3 and normal_mask.shape[0] != 1:
            normal_mask = normal_mask[:1]
        if normal_mask.shape == valid.shape:
            valid = valid & (normal_mask > 0)

    gt_normal = torch.nn.functional.normalize(gt_normal, p=2, dim=0, eps=1e-6)
    return gt_normal, valid


def parse_normal_confidence_mode(mode):
    mode = str(mode or "none").strip().lower()
    if mode in {"", "0", "false", "none", "off", "disabled"}:
        return mode, set()
    if mode in {"edge", "edges"}:
        return mode, {"edge"}
    if mode in {"depth_normal", "depth-normal", "dn"}:
        return mode, {"depth_normal"}
    if mode in {"depth_normal_edge", "depth-normal-edge", "dn_edge", "dn-edge"}:
        return mode, {"depth_normal", "edge"}
    raise ValueError(
        "--normal_confidence_mode must be one of: none, edge, depth_normal, depth_normal_edge"
    )


def parse_normal_confidence_edge_sources(sources):
    sources = str(sources or "rgb,normal,depth").strip().lower()
    if sources in {"", "all", "default"}:
        return {"rgb", "normal", "depth"}
    normalized = sources.replace("+", ",").replace("|", ",").replace(";", ",")
    parts = {part.strip() for part in normalized.split(",") if part.strip()}
    aliases = {
        "image": "rgb",
        "color": "rgb",
        "colour": "rgb",
        "invdepth": "depth",
        "inverse_depth": "depth",
        "normalmap": "normal",
    }
    resolved = {aliases.get(part, part) for part in parts}
    allowed = {"rgb", "normal", "depth"}
    unknown = resolved - allowed
    if unknown:
        raise ValueError(
            "--normal_confidence_edge_sources must contain only rgb, normal, depth"
        )
    return resolved


def _as_chw(tensor):
    if tensor is None:
        return None
    while tensor.ndim > 3:
        tensor = tensor[0]
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        return None
    return tensor


def _gradient_magnitude_chw(tensor):
    tensor = _as_chw(tensor)
    if tensor is None:
        return None
    tensor = torch.nan_to_num(tensor.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    _, height, width = tensor.shape
    grad = torch.zeros((1, height, width), device=tensor.device, dtype=tensor.dtype)
    if width > 1:
        dx = torch.linalg.norm(tensor[:, :, 1:] - tensor[:, :, :-1], dim=0, keepdim=True)
        grad[:, :, 1:] = torch.maximum(grad[:, :, 1:], dx)
        grad[:, :, :-1] = torch.maximum(grad[:, :, :-1], dx)
    if height > 1:
        dy = torch.linalg.norm(tensor[:, 1:, :] - tensor[:, :-1, :], dim=0, keepdim=True)
        grad[:, 1:, :] = torch.maximum(grad[:, 1:, :], dy)
        grad[:, :-1, :] = torch.maximum(grad[:, :-1, :], dy)
    return grad


def _robust_unit_signal(signal, valid, quantile=0.95):
    signal = torch.nan_to_num(signal.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    valid_values = signal[valid.detach()]
    if valid_values.numel() == 0:
        return torch.zeros_like(signal)
    scale = torch.quantile(valid_values, float(max(0.0, min(1.0, quantile)))).clamp_min(1e-6)
    return (signal / scale).clamp(0.0, 1.0)


def _edge_confidence(gt_image, gt_normal, invdepth, valid, opt):
    edge_parts = []
    edge_sources = parse_normal_confidence_edge_sources(
        getattr(opt, "normal_confidence_edge_sources", "rgb,normal,depth")
    )
    if "rgb" in edge_sources:
        rgb_grad = _gradient_magnitude_chw(gt_image)
        if rgb_grad is not None and rgb_grad.shape == valid.shape:
            edge_parts.append(_robust_unit_signal(rgb_grad, valid))
    if "normal" in edge_sources:
        normal_grad = _gradient_magnitude_chw(gt_normal)
        if normal_grad is not None and normal_grad.shape == valid.shape:
            edge_parts.append(_robust_unit_signal(normal_grad, valid))
    if "depth" in edge_sources:
        invdepth = _as_chw(invdepth)
        if invdepth is not None and invdepth.shape == valid.shape:
            depth_grad = _gradient_magnitude_chw(invdepth)
            if depth_grad is not None:
                edge_parts.append(_robust_unit_signal(depth_grad, valid))

    if not edge_parts:
        return valid.to(gt_normal.dtype), {"threshold": 0.0, "keep": 1.0}

    edge_signal = torch.stack(edge_parts, dim=0).sum(dim=0)
    valid_values = edge_signal[valid.detach()]
    if valid_values.numel() == 0:
        return torch.zeros_like(edge_signal), {"threshold": 0.0, "keep": 0.0}

    edge_quantile = float(getattr(opt, "normal_confidence_edge_quantile", 0.9))
    edge_quantile = max(0.0, min(1.0, edge_quantile))
    threshold = torch.quantile(valid_values.detach(), edge_quantile)
    edge = edge_signal > threshold

    dilate_px = max(0, int(getattr(opt, "normal_confidence_edge_dilate_px", 3)))
    if dilate_px > 0:
        edge = torch.nn.functional.max_pool2d(
            edge.float().unsqueeze(0),
            kernel_size=2 * dilate_px + 1,
            stride=1,
            padding=dilate_px,
        ).squeeze(0) > 0

    edge_floor = float(getattr(opt, "normal_confidence_edge_floor", 0.0))
    edge_floor = max(0.0, min(1.0, edge_floor))
    confidence = torch.where(
        edge,
        torch.full_like(edge_signal, edge_floor),
        torch.ones_like(edge_signal),
    ) * valid.to(gt_normal.dtype)
    keep = confidence.sum() / valid.to(gt_normal.dtype).sum().clamp_min(1.0)
    return confidence, {"threshold": float(threshold.detach().item()), "keep": float(keep.detach().item())}


def _depth_normal_confidence(gt_normal, invdepth, viewpoint, valid, opt):
    invdepth = _as_chw(invdepth)
    if invdepth is None or invdepth.shape != valid.shape:
        return torch.zeros_like(valid, dtype=gt_normal.dtype), {"available": 0.0, "mean": 0.0}

    invdepth = invdepth.to(device=gt_normal.device, dtype=gt_normal.dtype, non_blocking=True)
    invdepth = torch.nan_to_num(invdepth, nan=0.0, posinf=0.0, neginf=0.0)
    _, height, width = invdepth.shape
    if height < 3 or width < 3:
        return torch.zeros_like(valid, dtype=gt_normal.dtype), {"available": 0.0, "mean": 0.0}

    depth_valid = valid & torch.isfinite(invdepth) & (invdepth > 1e-6)
    z = torch.where(depth_valid, 1.0 / invdepth.clamp_min(1e-6), torch.zeros_like(invdepth))
    fx = 0.5 * float(width) / math.tan(float(viewpoint.FoVx) * 0.5)
    fy = 0.5 * float(height) / math.tan(float(viewpoint.FoVy) * 0.5)
    cx = 0.5 * float(width - 1)
    cy = 0.5 * float(height - 1)
    u = torch.arange(width, device=gt_normal.device, dtype=gt_normal.dtype).view(1, 1, width)
    v = torch.arange(height, device=gt_normal.device, dtype=gt_normal.dtype).view(1, height, 1)
    points = torch.cat(((u - cx) / fx * z, (v - cy) / fy * z, z), dim=0)

    dx = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]
    dy = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]
    normal_inner = torch.cross(dx.permute(1, 2, 0), dy.permute(1, 2, 0), dim=-1).permute(2, 0, 1)
    normal_inner = torch.nn.functional.normalize(normal_inner, p=2, dim=0, eps=1e-6)
    inner_valid = (
        depth_valid[:, 1:-1, 1:-1]
        & depth_valid[:, 1:-1, 2:]
        & depth_valid[:, 1:-1, :-2]
        & depth_valid[:, 2:, 1:-1]
        & depth_valid[:, :-2, 1:-1]
        & torch.isfinite(normal_inner).all(dim=0, keepdim=True)
    )

    depth_normal = torch.zeros_like(gt_normal)
    depth_normal[:, 1:-1, 1:-1] = normal_inner
    confidence_valid = torch.zeros_like(valid)
    confidence_valid[:, 1:-1, 1:-1] = inner_valid
    valid_float = confidence_valid.to(gt_normal.dtype)
    dot = (depth_normal * gt_normal).sum(dim=0, keepdim=True).clamp(-1.0, 1.0)
    mean_dot = (dot * valid_float).sum() / valid_float.sum().clamp_min(1.0)
    if mean_dot.detach() < 0:
        dot = -dot

    angle_deg = float(getattr(opt, "normal_confidence_depth_angle_deg", 45.0))
    cos_min = math.cos(math.radians(max(0.0, min(89.0, angle_deg))))
    depth_floor = float(getattr(opt, "normal_confidence_depth_floor", 0.0))
    depth_floor = max(0.0, min(1.0, depth_floor))
    confidence = ((dot - cos_min) / max(1.0 - cos_min, 1e-6)).clamp(0.0, 1.0)
    if depth_floor > 0.0:
        confidence = confidence * (1.0 - depth_floor) + depth_floor
    confidence = confidence * valid_float
    mean_conf = confidence.sum() / valid_float.sum().clamp_min(1.0)
    return confidence, {"available": 1.0, "mean": float(mean_conf.detach().item())}


def compute_normal_confidence(gt_normal, gt_image, viewpoint, valid, mode_parts, opt):
    confidence = valid.to(gt_normal.dtype)
    stats = {
        "depth_available": 0.0,
        "depth_mean": 0.0,
        "edge_threshold": 0.0,
        "edge_keep": 1.0,
    }
    invdepth = getattr(viewpoint, "invdepthmap", None)
    if invdepth is not None:
        invdepth = invdepth.to(device=gt_normal.device, dtype=gt_normal.dtype, non_blocking=True)

    if "depth_normal" in mode_parts:
        if not bool(getattr(viewpoint, "depth_reliable", False)):
            confidence = torch.zeros_like(confidence)
        else:
            depth_conf, depth_stats = _depth_normal_confidence(gt_normal, invdepth, viewpoint, valid, opt)
            confidence = confidence * depth_conf
            stats["depth_available"] = depth_stats["available"]
            stats["depth_mean"] = depth_stats["mean"]

    if "edge" in mode_parts:
        edge_conf, edge_stats = _edge_confidence(gt_image, gt_normal, invdepth, valid, opt)
        confidence = confidence * edge_conf
        stats["edge_threshold"] = edge_stats["threshold"]
        stats["edge_keep"] = edge_stats["keep"]

    min_confidence = float(getattr(opt, "normal_confidence_min", 0.0))
    if min_confidence > 0.0:
        confidence = torch.where(confidence >= min_confidence, confidence, torch.zeros_like(confidence))

    confidence = confidence.detach() * valid.to(gt_normal.dtype)
    valid_float = valid.to(gt_normal.dtype)
    stats["sum"] = float(confidence.sum().detach().item())
    stats["pixels"] = float((confidence > 0).to(gt_normal.dtype).sum().detach().item())
    stats["mean"] = float((confidence.sum() / valid_float.sum().clamp_min(1.0)).detach().item())
    stats["coverage"] = stats["pixels"] / max(float(valid.numel()), 1.0)
    return confidence, stats


def estimate_camera_sample_cuda_bytes(viewpoint):
    if viewpoint is None:
        return 0
    total = 0
    seen = set()
    for attr in (
        "original_image",
        "alpha_mask",
        "invdepthmap",
        "depth_mask",
        "normalmap",
        "normal_mask",
        "world_view_transform",
        "projection_matrix",
        "full_proj_transform",
        "camera_center",
    ):
        value = getattr(viewpoint, attr, None)
        if not isinstance(value, torch.Tensor) or not value.is_cuda:
            continue
        data_ptr = value.data_ptr()
        if data_ptr in seen:
            continue
        seen.add(data_ptr)
        total += value.numel() * value.element_size()
    return int(total)


def auto_cache_num_from_memory(dataset, train_dataset, current_cache_num, sample_bytes):
    if not torch.cuda.is_available() or sample_bytes <= 0:
        return int(current_cache_num)

    reserve_gb = float(getattr(dataset, "auto_cache_reserve_gb", 8.0))
    max_auto_num = int(getattr(dataset, "auto_cache_max_num", 0))
    reserve_bytes = max(0.0, reserve_gb) * (1024 ** 3)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    usable_bytes = max(0, int(free_bytes - reserve_bytes))
    estimated_num = int(usable_bytes // sample_bytes)

    if max_auto_num > 0:
        estimated_num = min(estimated_num, max_auto_num)
    estimated_num = min(estimated_num, len(train_dataset))

    target_num = estimated_num
    target_num = min(max(target_num, 1), len(train_dataset))

    print(
        "[DataLoader] auto cache estimate: "
        f"free={free_bytes / 1024**3:.2f}GB, "
        f"total={total_bytes / 1024**3:.2f}GB, "
        f"reserve={reserve_gb:.2f}GB, "
        f"sample={sample_bytes / 1024**2:.2f}MB, "
        f"current={current_cache_num}, "
        f"target={target_num}"
    )
    return target_num


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, logger):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
    normal_weight_init = float(getattr(opt, "normal_weight_init", 0.0))
    normal_weight_final = float(getattr(opt, "normal_weight_final", 0.0))
    normal_start_iter = max(0, int(getattr(opt, "normal_start_iter", 0)))
    normal_confidence_mode, normal_confidence_parts = parse_normal_confidence_mode(
        getattr(opt, "normal_confidence_mode", "none")
    )
    normal_weight_max_steps = opt.iterations if normal_start_iter <= 0 else max(1, opt.iterations - normal_start_iter + 1)
    if normal_weight_init > 0.0 and normal_weight_final > 0.0:
        normal_weight = get_expon_lr_func(normal_weight_init, normal_weight_final, max_steps=normal_weight_max_steps)
    else:
        normal_weight_constant = max(normal_weight_init, normal_weight_final)
        normal_weight = lambda _step: normal_weight_constant

    pseudo_weight_init = float(getattr(opt, "pseudo_view_weight_init", 0.0))
    pseudo_weight_final = float(getattr(opt, "pseudo_view_weight_final", 0.0))
    pseudo_start_iter = max(0, int(getattr(opt, "pseudo_view_start_iter", 5000)))
    pseudo_end_iter = int(getattr(opt, "pseudo_view_end_iter", 0))
    if pseudo_end_iter <= 0:
        pseudo_end_iter = int(getattr(opt, "densify_until_iter", opt.iterations))
    pseudo_end_iter = min(pseudo_end_iter, int(opt.iterations))
    pseudo_interval = max(1, int(getattr(opt, "pseudo_view_interval", 1)))
    pseudo_weight_max_steps = max(1, pseudo_end_iter - pseudo_start_iter + 1)
    if pseudo_weight_init > 0.0 and pseudo_weight_final > 0.0:
        pseudo_view_weight = get_expon_lr_func(
            pseudo_weight_init,
            pseudo_weight_final,
            max_steps=pseudo_weight_max_steps,
        )
    else:
        pseudo_weight_constant = max(pseudo_weight_init, pseudo_weight_final)
        pseudo_view_weight = lambda _step: pseudo_weight_constant

    pseudo_normal_weight_init = float(getattr(opt, "pseudo_normal_weight_init", 0.0))
    pseudo_normal_weight_final = float(getattr(opt, "pseudo_normal_weight_final", 0.0))
    pseudo_normal_start_iter = max(0, int(getattr(opt, "pseudo_normal_start_iter", 5000)))
    pseudo_normal_end_iter = int(getattr(opt, "pseudo_normal_end_iter", 0))
    if pseudo_normal_end_iter <= 0:
        pseudo_normal_end_iter = int(getattr(opt, "densify_until_iter", opt.iterations))
    pseudo_normal_end_iter = min(pseudo_normal_end_iter, int(opt.iterations))
    pseudo_normal_interval = max(1, int(getattr(opt, "pseudo_normal_interval", 2)))
    pseudo_normal_ref_depth_source = str(getattr(opt, "pseudo_normal_ref_depth_source", "mono")).strip().lower()
    if pseudo_normal_ref_depth_source == "render":
        pseudo_normal_ref_depth_source = "rendered"
    if pseudo_normal_ref_depth_source not in {"mono", "rendered"}:
        raise ValueError("--pseudo_normal_ref_depth_source must be 'mono' or 'rendered'")
    pseudo_normal_weight_max_steps = max(1, pseudo_normal_end_iter - pseudo_normal_start_iter + 1)
    if pseudo_normal_weight_init > 0.0 and pseudo_normal_weight_final > 0.0:
        pseudo_normal_weight = get_expon_lr_func(
            pseudo_normal_weight_init,
            pseudo_normal_weight_final,
            max_steps=pseudo_normal_weight_max_steps,
        )
    else:
        pseudo_normal_weight_constant = max(pseudo_normal_weight_init, pseudo_normal_weight_final)
        pseudo_normal_weight = lambda _step: pseudo_normal_weight_constant

    depth_mask_mode = getattr(opt, "depth_reg_mask_mode", "full") or "full"
    if depth_mask_mode not in {"full", "block_projection"}:
        raise ValueError("--depth_reg_mask_mode must be 'full' or 'block_projection'")
    block_depth_masker = None
    if depth_mask_mode == "block_projection":
        if scene.block_metadata is None:
            raise ValueError("depth_reg_mask_mode=block_projection requires block training")
        block_depth_masker = BlockDepthMasker(
            scene.block_metadata,
            bbox_mode=getattr(opt, "depth_reg_mask_bbox_mode", "expanded"),
            max_points=getattr(opt, "depth_reg_mask_max_points", 100000),
            dilate_px=getattr(opt, "depth_reg_mask_dilate_px", 16),
            cache_masks=getattr(opt, "depth_reg_mask_cache", True),
            cache_max_items=getattr(opt, "depth_reg_mask_cache_max_items", 0),
        )

    max_cache_num = int(getattr(dataset, "max_cache_num", 0))
    release_viewpoint_after_iter = max_cache_num == 0
    train_dataset = GSCameraDataset(
        scene.getTrainCameraInfos(),
        dataset,
        scene.is_nerf_synthetic,
        is_test_dataset=False,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("No training cameras found")
    def build_camera_loader(cache_num):
        print(
            "[DataLoader] "
            f"train_cameras={len(train_dataset)}, "
            f"max_cache_num={cache_num}, "
            f"cache_workers={getattr(dataset, 'image_cache_workers', 0)}"
        )
        return CameraDataLoader(
            train_dataset,
            batch_size=1,
            max_cache_num=cache_num,
            cache_workers=getattr(dataset, "image_cache_workers", 0),
            shuffle=True,
            seed=getattr(dataset, "image_loader_seed", 42),
            num_workers=0,
        )

    camera_loader = build_camera_loader(max_cache_num)
    camera_loader_iter = iter(camera_loader)
    auto_cache_enabled = bool(getattr(dataset, "auto_cache_after_densify", False))
    auto_cache_done = False
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    ema_Lnormal_for_log = 0.0
    ema_Lpseudo_for_log = 0.0
    ema_Lpseudo_normal_for_log = 0.0
    ema_time_render = 0.0
    ema_time_loss = 0.0
    ema_time_densify = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    try:
        for iteration in range(first_iter, opt.iterations + 1):
            iter_start.record()

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            try:
                viewpoint_cam = next(camera_loader_iter)
            except StopIteration:
                camera_loader_iter = iter(camera_loader)
                viewpoint_cam = next(camera_loader_iter)

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            start = time.time()
            normal_active = iteration >= normal_start_iter
            if normal_active:
                normal_weight_step = iteration if normal_start_iter <= 0 else iteration - normal_start_iter + 1
                normal_weight_value = normal_weight(normal_weight_step)
            else:
                normal_weight_value = 0.0
            render_normal = (
                normal_weight_value > 0
                and bool(getattr(viewpoint_cam, "normal_reliable", False))
                and getattr(viewpoint_cam, "normalmap", None) is not None
            )
            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                bg,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
                return_normal=render_normal,
            )
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            end = time.time()
            ema_time_render = 0.4 * (end - start) + 0.6 * ema_time_render

            start = time.time()
            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = image_to_cuda_float(viewpoint_cam.alpha_mask)
                image *= alpha_mask

            # Loss
            gt_image = image_to_cuda_float(viewpoint_cam.original_image)
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            # Depth regularization
            depth_weight_value = depth_l1_weight(iteration)
            depth_reg_enabled = bool(getattr(dataset, "depths", "")) and depth_weight_value > 0
            depth_reliable = bool(getattr(viewpoint_cam, "depth_reliable", False))
            Ll1depth_pure_item = 0.0
            depth_mask_pixels = 0.0
            depth_mask_coverage = 0.0
            depth_mask_enough = True
            depth_reg_time = 0.0
            depth_mask_project_time = 0.0
            depth_mask_transfer_time = 0.0
            depth_mask_cache_hit = False
            depth_mask_cache_size = 0
            depth_reg_start = time.time()
            if depth_weight_value > 0 and depth_reliable and viewpoint_cam.invdepthmap is not None:
                invDepth = render_pkg["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.to("cuda", non_blocking=True)
                depth_mask = None
                if viewpoint_cam.depth_mask is not None:
                    depth_mask = viewpoint_cam.depth_mask.to("cuda", non_blocking=True)
                if block_depth_masker is not None:
                    mask_start = time.time()
                    block_mask_cpu, depth_mask_cache_hit = block_depth_masker.mask_for(viewpoint_cam)
                    if not depth_mask_cache_hit:
                        depth_mask_project_time = time.time() - mask_start
                    depth_mask_cache_size = block_depth_masker.cache_size
                    transfer_start = time.time()
                    block_mask = block_mask_cpu.to(
                        device=mono_invdepth.device,
                        dtype=mono_invdepth.dtype,
                    )
                    depth_mask_transfer_time = time.time() - transfer_start
                    depth_mask = block_mask if depth_mask is None else depth_mask * block_mask

                if depth_mask is None:
                    valid_pixels = mono_invdepth.new_tensor(float(mono_invdepth.numel()))
                    depth_mask_pixels = float(mono_invdepth.numel())
                    depth_mask_coverage = 1.0
                else:
                    valid_pixels = depth_mask.sum()
                    depth_mask_pixels = float(valid_pixels.detach().item())
                    depth_mask_coverage = depth_mask_pixels / max(float(depth_mask.numel()), 1.0)
                min_depth_pixels = float(getattr(opt, "depth_reg_mask_min_pixels", 0))
                depth_mask_enough = depth_mask_pixels >= min_depth_pixels

                if depth_mask_enough:
                    depth_error = torch.abs(invDepth - mono_invdepth)
                    if depth_mask is not None:
                        depth_error = depth_error * depth_mask
                    Ll1depth_pure = depth_error.sum() / valid_pixels.clamp_min(1.0)
                    Ll1depth = depth_weight_value * Ll1depth_pure
                    loss += Ll1depth
                    Ll1depth_pure_item = Ll1depth_pure.item()
                    Ll1depth = Ll1depth.item()
                else:
                    Ll1depth = 0
                depth_reg_time = time.time() - depth_reg_start
            else:
                Ll1depth = 0
            depth_stats = {
                "enabled": depth_reg_enabled,
                "applied": depth_reg_enabled and depth_reliable and depth_mask_enough,
                "reliable": depth_reliable,
                "l1_loss": Ll1depth,
                "l1_loss_pure": Ll1depth_pure_item,
                "weight": depth_weight_value,
                "mask_mode": depth_mask_mode,
                "mask_pixels": depth_mask_pixels,
                "mask_coverage": depth_mask_coverage,
                "mask_enough": depth_mask_enough,
                "mask_cache_hit": depth_mask_cache_hit,
                "mask_cache_size": depth_mask_cache_size,
                "time_depth_reg": depth_reg_time,
                "time_depth_mask_project": depth_mask_project_time,
                "time_depth_mask_transfer": depth_mask_transfer_time,
            }

            # Pseudo-view reprojection regularization, intended to affect pruning-stage floaters.
            pseudo_reg_enabled = max(pseudo_weight_init, pseudo_weight_final) > 0.0 and pseudo_end_iter >= pseudo_start_iter
            pseudo_active = (
                pseudo_reg_enabled
                and pseudo_start_iter <= iteration <= pseudo_end_iter
                and (iteration - pseudo_start_iter) % pseudo_interval == 0
            )
            pseudo_weight_value = 0.0
            Lpseudo = 0.0
            Lpseudo_pure_item = 0.0
            pseudo_mask_pixels = 0.0
            pseudo_mask_coverage = 0.0
            pseudo_mask_enough = False
            pseudo_valid_depth = False
            pseudo_shift = 0.0
            pseudo_reg_time = 0.0
            if pseudo_active:
                pseudo_start = time.time()
                pseudo_weight_step = iteration - pseudo_start_iter + 1
                pseudo_weight_value = pseudo_view_weight(pseudo_weight_step)
                pseudo_camera, pseudo_camera_stats = make_pseudo_view_camera(
                    viewpoint_cam,
                    render_pkg["depth"],
                    shift_ratio=float(getattr(opt, "pseudo_view_shift_ratio", 0.03)),
                    random_sign=bool(getattr(opt, "pseudo_view_random_sign", True)),
                    detach_depth=bool(getattr(opt, "pseudo_view_detach_ref_depth", True)),
                )
                pseudo_shift = pseudo_camera_stats.get("shift", 0.0)
                pseudo_valid_depth = bool(pseudo_camera_stats.get("valid_depth", False))
                if pseudo_camera is not None and pseudo_weight_value > 0.0:
                    pseudo_render_pkg = render(
                        pseudo_camera,
                        gaussians,
                        pipe,
                        bg,
                        use_trained_exp=dataset.train_test_exp,
                        separate_sh=SPARSE_ADAM_AVAILABLE,
                    )
                    Lpseudo_pure, pseudo_loss_stats = pseudo_view_reprojection_loss(
                        viewpoint_cam,
                        pseudo_camera,
                        render_pkg["depth"],
                        pseudo_render_pkg["depth"],
                        pseudo_render_pkg["render"],
                        gt_image,
                        mask_mode=getattr(opt, "pseudo_view_mask_mode", "valid"),
                        depth_rel_thresh=float(getattr(opt, "pseudo_view_depth_rel_thresh", 0.05)),
                        min_pixels=float(getattr(opt, "pseudo_view_min_pixels", 2048)),
                        detach_ref_depth=bool(getattr(opt, "pseudo_view_detach_ref_depth", True)),
                    )
                    pseudo_mask_pixels = pseudo_loss_stats.get("mask_pixels", 0.0)
                    pseudo_mask_coverage = pseudo_loss_stats.get("mask_coverage", 0.0)
                    pseudo_mask_enough = bool(pseudo_loss_stats.get("mask_enough", False))
                    if Lpseudo_pure is not None:
                        Lpseudo_tensor = pseudo_weight_value * Lpseudo_pure
                        loss += Lpseudo_tensor
                        Lpseudo_pure_item = Lpseudo_pure.item()
                        Lpseudo = Lpseudo_tensor.item()
                pseudo_reg_time = time.time() - pseudo_start
            pseudo_stats = {
                "enabled": pseudo_reg_enabled,
                "active": pseudo_active,
                "applied": pseudo_active and pseudo_mask_enough,
                "valid_depth": pseudo_valid_depth,
                "loss": Lpseudo,
                "loss_pure": Lpseudo_pure_item,
                "weight": pseudo_weight_value,
                "start_iter": pseudo_start_iter,
                "end_iter": pseudo_end_iter,
                "interval": pseudo_interval,
                "mask_pixels": pseudo_mask_pixels,
                "mask_coverage": pseudo_mask_coverage,
                "mask_enough": pseudo_mask_enough,
                "shift": pseudo_shift,
                "shift_ratio": float(getattr(opt, "pseudo_view_shift_ratio", 0.03)),
                "depth_rel_thresh": float(getattr(opt, "pseudo_view_depth_rel_thresh", 0.05)),
                "time_pseudo_reg": pseudo_reg_time,
            }

            # Pseudo-normal consistency: render normals from a nearby view and compare after reprojection.
            pseudo_normal_reg_enabled = (
                max(pseudo_normal_weight_init, pseudo_normal_weight_final) > 0.0
                and pseudo_normal_end_iter >= pseudo_normal_start_iter
            )
            pseudo_normal_reliable = (
                bool(getattr(viewpoint_cam, "normal_reliable", False))
                and getattr(viewpoint_cam, "normalmap", None) is not None
            )
            pseudo_normal_active = (
                pseudo_normal_reg_enabled
                and pseudo_normal_reliable
                and pseudo_normal_start_iter <= iteration <= pseudo_normal_end_iter
                and (iteration - pseudo_normal_start_iter) % pseudo_normal_interval == 0
            )
            pseudo_normal_weight_value = 0.0
            Lpseudo_normal = 0.0
            Lpseudo_normal_pure_item = 0.0
            pseudo_normal_mask_pixels = 0.0
            pseudo_normal_mask_coverage = 0.0
            pseudo_normal_mask_enough = False
            pseudo_normal_valid_depth = False
            pseudo_normal_shift = 0.0
            pseudo_normal_depth_from_mono = False
            pseudo_normal_ref_mask = None
            pseudo_normal_reg_time = 0.0
            if pseudo_normal_active:
                pseudo_normal_start = time.time()
                pseudo_normal_weight_step = iteration - pseudo_normal_start_iter + 1
                pseudo_normal_weight_value = pseudo_normal_weight(pseudo_normal_weight_step)
                gt_pseudo_normal, pseudo_normal_valid = prepare_gt_normal(viewpoint_cam)

                if pseudo_normal_ref_depth_source == "mono" and depth_reliable and viewpoint_cam.invdepthmap is not None:
                    pseudo_normal_ref_invdepth = viewpoint_cam.invdepthmap.to("cuda", non_blocking=True).float()
                    pseudo_normal_depth_from_mono = True
                    if viewpoint_cam.depth_mask is not None:
                        pseudo_normal_ref_mask = viewpoint_cam.depth_mask.to("cuda", non_blocking=True)
                else:
                    pseudo_normal_ref_invdepth = render_pkg["depth"]

                if gt_pseudo_normal is not None and pseudo_normal_weight_value > 0.0:
                    pseudo_normal_camera, pseudo_normal_camera_stats = make_pseudo_view_camera(
                        viewpoint_cam,
                        pseudo_normal_ref_invdepth,
                        shift_ratio=float(getattr(opt, "pseudo_normal_shift_ratio", 0.03)),
                        random_sign=bool(getattr(opt, "pseudo_normal_random_sign", True)),
                        detach_depth=bool(getattr(opt, "pseudo_normal_detach_ref_depth", True)),
                    )
                    pseudo_normal_shift = pseudo_normal_camera_stats.get("shift", 0.0)
                    pseudo_normal_valid_depth = bool(pseudo_normal_camera_stats.get("valid_depth", False))
                    if pseudo_normal_camera is not None:
                        pseudo_normal_render_pkg = render(
                            pseudo_normal_camera,
                            gaussians,
                            pipe,
                            bg,
                            use_trained_exp=dataset.train_test_exp,
                            separate_sh=SPARSE_ADAM_AVAILABLE,
                            return_normal=True,
                        )
                        warped_normal, pseudo_normal_mask, pseudo_normal_loss_stats = warp_pseudo_normal_to_ref(
                            viewpoint_cam,
                            pseudo_normal_camera,
                            pseudo_normal_ref_invdepth,
                            pseudo_normal_render_pkg["normal"],
                            pseudo_normal_render_pkg["alpha"],
                            pseudo_normal_valid,
                            ref_valid_mask=pseudo_normal_ref_mask,
                            alpha_min=float(getattr(opt, "pseudo_normal_alpha_min", 0.01)),
                            min_pixels=float(getattr(opt, "pseudo_normal_min_pixels", 2048)),
                            detach_ref_depth=bool(getattr(opt, "pseudo_normal_detach_ref_depth", True)),
                        )
                        pseudo_normal_mask_pixels = pseudo_normal_loss_stats.get("mask_pixels", 0.0)
                        pseudo_normal_mask_coverage = pseudo_normal_loss_stats.get("mask_coverage", 0.0)
                        pseudo_normal_mask_enough = bool(pseudo_normal_loss_stats.get("mask_enough", False))
                        if warped_normal is not None:
                            pred_pseudo_normal = world_normal_to_camera(
                                warped_normal,
                                viewpoint_cam,
                                sign=getattr(opt, "normal_sign", -1.0),
                            )
                            normal_dot = (pred_pseudo_normal * gt_pseudo_normal).sum(dim=0, keepdim=True).clamp(-1.0, 1.0)
                            normal_error = 1.0 - normal_dot
                            Lpseudo_normal_pure = (normal_error * pseudo_normal_mask).sum() / pseudo_normal_mask.sum().clamp_min(1.0)
                            Lpseudo_normal_tensor = pseudo_normal_weight_value * Lpseudo_normal_pure
                            loss += Lpseudo_normal_tensor
                            Lpseudo_normal_pure_item = Lpseudo_normal_pure.item()
                            Lpseudo_normal = Lpseudo_normal_tensor.item()
                pseudo_normal_reg_time = time.time() - pseudo_normal_start
            pseudo_normal_stats = {
                "enabled": pseudo_normal_reg_enabled,
                "active": pseudo_normal_active,
                "applied": pseudo_normal_active and pseudo_normal_mask_enough,
                "reliable": pseudo_normal_reliable,
                "valid_depth": pseudo_normal_valid_depth,
                "loss": Lpseudo_normal,
                "loss_pure": Lpseudo_normal_pure_item,
                "weight": pseudo_normal_weight_value,
                "start_iter": pseudo_normal_start_iter,
                "end_iter": pseudo_normal_end_iter,
                "interval": pseudo_normal_interval,
                "mask_pixels": pseudo_normal_mask_pixels,
                "mask_coverage": pseudo_normal_mask_coverage,
                "mask_enough": pseudo_normal_mask_enough,
                "alpha_min": float(getattr(opt, "pseudo_normal_alpha_min", 0.01)),
                "shift": pseudo_normal_shift,
                "shift_ratio": float(getattr(opt, "pseudo_normal_shift_ratio", 0.03)),
                "ref_depth_is_mono": float(pseudo_normal_depth_from_mono),
                "ref_depth_has_mask": float(pseudo_normal_ref_mask is not None),
                "time_pseudo_normal_reg": pseudo_normal_reg_time,
            }

            # Normal regularization
            normal_reg_enabled = normal_weight_value > 0
            normal_reliable = bool(getattr(viewpoint_cam, "normal_reliable", False))
            Lnormal = 0
            Lnormal_pure_item = 0.0
            normal_mask_pixels = 0.0
            normal_mask_coverage = 0.0
            normal_base_mask_pixels = 0.0
            normal_confidence_sum = 0.0
            normal_confidence_mean = 1.0
            normal_confidence_pixels = 0.0
            normal_confidence_coverage = 0.0
            normal_confidence_depth_available = 0.0
            normal_confidence_depth_mean = 0.0
            normal_confidence_edge_threshold = 0.0
            normal_confidence_edge_keep = 1.0
            normal_mask_enough = True
            normal_reg_time = 0.0
            normal_reg_start = time.time()
            if render_normal and "normal" in render_pkg:
                rendered_normal = render_pkg["normal"]
                gt_normal, valid = prepare_gt_normal(viewpoint_cam)

                rendered_alpha = render_pkg.get("alpha")
                if rendered_alpha is not None:
                    valid = valid & (rendered_alpha.detach() > float(getattr(opt, "normal_alpha_min", 0.001)))

                valid_float = valid.to(gt_normal.dtype)
                valid_pixels = valid_float.sum()
                normal_base_mask_pixels = float(valid_pixels.detach().item())
                normal_confidence = valid_float
                normal_confidence_denom = valid_pixels
                normal_confidence_pixels = normal_base_mask_pixels
                normal_confidence_sum = normal_base_mask_pixels
                normal_confidence_coverage = normal_confidence_pixels / max(float(valid_float.numel()), 1.0)
                normal_confidence_mean = 1.0 if normal_base_mask_pixels > 0 else 0.0

                if normal_base_mask_pixels > 0 and normal_confidence_parts:
                    confidence_stats = {}
                    normal_confidence, confidence_stats = compute_normal_confidence(
                        gt_normal,
                        gt_image,
                        viewpoint_cam,
                        valid,
                        normal_confidence_parts,
                        opt,
                    )
                    normal_confidence_denom = normal_confidence.sum()
                    normal_confidence_sum = float(normal_confidence_denom.detach().item())
                    normal_confidence_pixels = float(
                        (normal_confidence > 0).to(gt_normal.dtype).sum().detach().item()
                    )
                    normal_confidence_coverage = normal_confidence_pixels / max(float(valid_float.numel()), 1.0)
                    normal_confidence_mean = confidence_stats.get("mean", 0.0)
                    normal_confidence_depth_available = confidence_stats.get("depth_available", 0.0)
                    normal_confidence_depth_mean = confidence_stats.get("depth_mean", 0.0)
                    normal_confidence_edge_threshold = confidence_stats.get("edge_threshold", 0.0)
                    normal_confidence_edge_keep = confidence_stats.get("edge_keep", 1.0)

                normal_mask_pixels = normal_confidence_pixels if normal_confidence_parts else normal_base_mask_pixels
                normal_mask_coverage = normal_mask_pixels / max(float(valid_float.numel()), 1.0)
                min_normal_pixels = float(getattr(opt, "normal_min_pixels", 0))
                normal_mask_enough = (
                    normal_mask_pixels >= min_normal_pixels
                    and normal_mask_pixels > 0
                    and float(normal_confidence_denom.detach().item()) > 0.0
                )

                if normal_mask_enough:
                    pred_normal = world_normal_to_camera(
                        rendered_normal,
                        viewpoint_cam,
                        sign=getattr(opt, "normal_sign", -1.0),
                    )
                    normal_dot = (pred_normal * gt_normal).sum(dim=0, keepdim=True).clamp(-1.0, 1.0)
                    normal_error = 1.0 - normal_dot
                    Lnormal_pure = (normal_error * normal_confidence).sum() / normal_confidence_denom.clamp_min(1.0)
                    Lnormal_tensor = normal_weight_value * Lnormal_pure
                    loss += Lnormal_tensor
                    Lnormal_pure_item = Lnormal_pure.item()
                    Lnormal = Lnormal_tensor.item()
                normal_reg_time = time.time() - normal_reg_start
            normal_stats = {
                "enabled": normal_reg_enabled,
                "applied": render_normal and normal_mask_enough,
                "reliable": normal_reliable,
                "loss": Lnormal,
                "loss_pure": Lnormal_pure_item,
                "weight": normal_weight_value,
                "mask_pixels": normal_mask_pixels,
                "mask_coverage": normal_mask_coverage,
                "base_mask_pixels": normal_base_mask_pixels,
                "mask_enough": normal_mask_enough,
                "alpha_min": float(getattr(opt, "normal_alpha_min", 0.001)),
                "sign": float(getattr(opt, "normal_sign", -1.0)),
                "start_iter": normal_start_iter,
                "confidence_mode": normal_confidence_mode,
                "confidence_enabled": bool(normal_confidence_parts),
                "confidence_sum": normal_confidence_sum,
                "confidence_mean": normal_confidence_mean,
                "confidence_pixels": normal_confidence_pixels,
                "confidence_coverage": normal_confidence_coverage,
                "confidence_depth_available": normal_confidence_depth_available,
                "confidence_depth_mean": normal_confidence_depth_mean,
                "confidence_edge_threshold": normal_confidence_edge_threshold,
                "confidence_edge_keep": normal_confidence_edge_keep,
                "confidence_depth_angle_deg": float(getattr(opt, "normal_confidence_depth_angle_deg", 45.0)),
                "confidence_edge_quantile": float(getattr(opt, "normal_confidence_edge_quantile", 0.9)),
                "confidence_edge_dilate_px": int(getattr(opt, "normal_confidence_edge_dilate_px", 3)),
                "confidence_edge_floor": float(getattr(opt, "normal_confidence_edge_floor", 0.0)),
                "confidence_depth_floor": float(getattr(opt, "normal_confidence_depth_floor", 0.0)),
                "confidence_min": float(getattr(opt, "normal_confidence_min", 0.0)),
                "time_normal_reg": normal_reg_time,
            }

            loss.backward()
            end = time.time()
            ema_time_loss = 0.4 * (end - start) + 0.6 * ema_time_loss

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
                ema_Lnormal_for_log = 0.4 * Lnormal + 0.6 * ema_Lnormal_for_log
                ema_Lpseudo_for_log = 0.4 * Lpseudo + 0.6 * ema_Lpseudo_for_log
                ema_Lpseudo_normal_for_log = 0.4 * Lpseudo_normal + 0.6 * ema_Lpseudo_normal_for_log

                if iteration % 10 == 0:
                    progress_bar.set_postfix({
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}",
                        "Normal Loss": f"{ema_Lnormal_for_log:.{7}f}",
                        "Pseudo Loss": f"{ema_Lpseudo_for_log:.{7}f}",
                        "Pseudo Normal": f"{ema_Lpseudo_normal_for_log:.{7}f}",
                    })
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                grads = gaussians.xyz_gradient_accum / gaussians.denom
                grads = torch.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
                ema_time = {
                    "render": ema_time_render,
                    "loss": ema_time_loss,
                    "densify": ema_time_densify,
                    "num_points": radii.shape[0],
                    "mean_grad": grads.mean().item() if grads.numel() > 0 else 0.0,
                }

                # Log and save
                training_report(logger, iteration, Ll1, loss, l1_loss, ema_time, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), radii, visibility_filter, dataset.train_test_exp, dataset, depth_stats, normal_stats, pseudo_stats, pseudo_normal_stats)
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        torch.cuda.empty_cache()
                        points_before = gaussians.get_xyz.shape[0]
                        allocated_before = torch.cuda.memory_allocated()
                        reserved_before = torch.cuda.memory_reserved()
                        try:
                            start = time.time()
                            gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                            end = time.time()
                            ema_time_densify = 0.4 * (end - start) + 0.6 * ema_time_densify
                        except RuntimeError as exc:
                            if "out of memory" in str(exc).lower():
                                print(
                                    "\n[OOM during densify] "
                                    f"iteration={iteration}, "
                                    f"points_before={points_before}, "
                                    f"allocated_before={allocated_before / 1024**3:.2f}GB, "
                                    f"reserved_before={reserved_before / 1024**3:.2f}GB, "
                                    f"allocated_now={torch.cuda.memory_allocated() / 1024**3:.2f}GB, "
                                    f"reserved_now={torch.cuda.memory_reserved() / 1024**3:.2f}GB"
                                )
                                if logger:
                                    try:
                                        logger.add_scalar("oom/iteration", iteration, iteration)
                                        logger.add_scalar("oom/points_before_densify", points_before, iteration)
                                        logger.add_scalar("oom/memory_allocated_gb", torch.cuda.memory_allocated() / 1024**3, iteration)
                                        logger.add_scalar("oom/memory_reserved_gb", torch.cuda.memory_reserved() / 1024**3, iteration)
                                    except Exception as log_exc:
                                        print(f"[OOM logging failed] {log_exc}")
                            raise
                        points_after = gaussians.get_xyz.shape[0]
                        points_delta = points_after - points_before
                        print(f"\n[DENSIFY] iteration={iteration}, points {points_before} -> {points_after}")
                        if logger:
                            logger.add_scalar("densify/points_before", points_before, iteration)
                            logger.add_scalar("densify/points_after", points_after, iteration)
                            logger.add_scalar("densify/points_delta", points_delta, iteration)

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                    if use_sparse_adam:
                        visible = radii > 0
                        gaussians.optimizer.step(visible, radii.shape[0])
                        gaussians.optimizer.zero_grad(set_to_none = True)
                    else:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none = True)

                if (
                    auto_cache_enabled
                    and not auto_cache_done
                    and iteration >= opt.densify_until_iter
                    and iteration < opt.iterations
                ):
                    sample_bytes = estimate_camera_sample_cuda_bytes(viewpoint_cam)
                    if release_viewpoint_after_iter and viewpoint_cam is not None:
                        viewpoint_cam.release_image()
                        viewpoint_cam = None
                    if hasattr(camera_loader_iter, "close"):
                        camera_loader_iter.close()
                    camera_loader.close()
                    target_cache_num = auto_cache_num_from_memory(
                        dataset,
                        train_dataset,
                        camera_loader.max_cache_num,
                        sample_bytes,
                    )
                    if target_cache_num != camera_loader.max_cache_num:
                        print(
                            "[DataLoader] switching cache after densify: "
                            f"{camera_loader.max_cache_num} -> {target_cache_num} "
                            f"at iteration {iteration}"
                        )
                    else:
                        print(f"[DataLoader] keeping cache max_cache_num={target_cache_num} after densify")
                    if logger:
                        logger.add_scalar("data/auto_cache_num", target_cache_num, iteration)
                        logger.add_scalar("data/auto_cache_sample_mb", sample_bytes / 1024**2, iteration)
                    max_cache_num = target_cache_num
                    release_viewpoint_after_iter = max_cache_num == 0
                    camera_loader = build_camera_loader(max_cache_num)
                    camera_loader_iter = iter(camera_loader)
                    auto_cache_done = True

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                if release_viewpoint_after_iter and viewpoint_cam is not None:
                    viewpoint_cam.release_image()
    finally:
        if hasattr(camera_loader_iter, "close"):
            camera_loader_iter.close()
        progress_bar.close()
        logger.close()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    resolved_config = getattr(args, "resolved_config", None)
    if resolved_config:
        save_yaml_config(os.path.join(args.model_path, "resolved_config.yaml"), resolved_config)
    cfg_args = vars(args).copy()
    cfg_args.pop("resolved_config", None)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**cfg_args)))

    # Create SwanLab logger
    logger = SwanLabLogger(enabled=SWANLAB_FOUND)
    if SWANLAB_FOUND:
        try:
            logdir = args.swanlab_logdir or os.path.join(args.model_path, "swanlog")
            os.makedirs(logdir, exist_ok=True)
            swanlab.init(
                project=args.swanlab_project or None,
                workspace=args.swanlab_workspace or None,
                experiment_name=args.swanlab_experiment_name or None,
                mode=args.swanlab_mode or None,
                id=getattr(args, "swanlab_run_id", "") or None,
                resume=getattr(args, "swanlab_resume", "") or None,
                config=cfg_args,
                logdir=logdir,
            )
        except Exception as exc:
            logger.enabled = False
            print(f"SwanLab initialization failed: {exc}. Continuing without SwanLab logging.")
    else:
        print("SwanLab not available: not logging progress")
    return logger

def log_training_observations(logger, iteration, gaussians, radii, visibility_filter):
    if not logger:
        return

    total_points = gaussians.get_xyz.shape[0]
    visible_points = int(visibility_filter.sum().item())
    logger.add_scalar("scene/total_points", total_points, iteration)
    logger.add_scalar("scene/visible_points", visible_points, iteration)
    logger.add_scalar("scene/visibility_ratio", visible_points / max(total_points, 1), iteration)

    if visible_points > 0:
        visible_radii = radii[visibility_filter]
        logger.add_scalar("scene/visible_radii_mean", visible_radii.float().mean().item(), iteration)
        logger.add_scalar("scene/visible_radii_max", visible_radii.float().max().item(), iteration)

    if torch.cuda.is_available():
        logger.add_scalar("gpu/memory_allocated_gb", torch.cuda.memory_allocated() / 1024**3, iteration)
        logger.add_scalar("gpu/memory_reserved_gb", torch.cuda.memory_reserved() / 1024**3, iteration)
        logger.add_scalar("gpu/max_memory_allocated_gb", torch.cuda.max_memory_allocated() / 1024**3, iteration)

    for group in gaussians.optimizer.param_groups:
        group_name = group.get("name", "unknown")
        logger.add_scalar(f"optimizer/lr_{group_name}", group["lr"], iteration)

def training_report(logger, iteration, Ll1, loss, l1_loss, ema_time, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, radii, visibility_filter, train_test_exp, dataset_args=None, depth_stats=None, normal_stats=None, pseudo_stats=None, pseudo_normal_stats=None):
    if logger and iteration % 10 == 0:
        logger.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        logger.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        if depth_stats and depth_stats.get("enabled", False):
            logger.add_scalar('train_loss_patches/depth_l1_loss', depth_stats["l1_loss"], iteration)
            logger.add_scalar('train_loss_patches/depth_l1_loss_pure', depth_stats["l1_loss_pure"], iteration)
            logger.add_scalar('train_loss_patches/depth_l1_weight', depth_stats["weight"], iteration)
            logger.add_scalar('train_loss_patches/depth_reliable', float(depth_stats["reliable"]), iteration)
            logger.add_scalar('train_loss_patches/depth_applied', float(depth_stats["applied"]), iteration)
            logger.add_scalar('train_loss_patches/depth_mask_enough', float(depth_stats["mask_enough"]), iteration)
            logger.add_scalar('train_loss_patches/depth_mask_pixels', depth_stats["mask_pixels"], iteration)
            logger.add_scalar('train_loss_patches/depth_mask_coverage', depth_stats["mask_coverage"], iteration)
            logger.add_scalar('train_loss_patches/depth_mask_cache_hit', float(depth_stats["mask_cache_hit"]), iteration)
            logger.add_scalar('train_loss_patches/depth_mask_cache_size', depth_stats["mask_cache_size"], iteration)
            logger.add_scalar('train_time/depth_reg', depth_stats["time_depth_reg"], iteration)
            logger.add_scalar('train_time/depth_mask_project', depth_stats["time_depth_mask_project"], iteration)
            logger.add_scalar('train_time/depth_mask_transfer', depth_stats["time_depth_mask_transfer"], iteration)
        if normal_stats and normal_stats.get("enabled", False):
            logger.add_scalar('train_loss_patches/normal_loss', normal_stats["loss"], iteration)
            logger.add_scalar('train_loss_patches/normal_loss_pure', normal_stats["loss_pure"], iteration)
            logger.add_scalar('train_loss_patches/normal_weight', normal_stats["weight"], iteration)
            logger.add_scalar('train_loss_patches/normal_reliable', float(normal_stats["reliable"]), iteration)
            logger.add_scalar('train_loss_patches/normal_applied', float(normal_stats["applied"]), iteration)
            logger.add_scalar('train_loss_patches/normal_mask_enough', float(normal_stats["mask_enough"]), iteration)
            logger.add_scalar('train_loss_patches/normal_mask_pixels', normal_stats["mask_pixels"], iteration)
            logger.add_scalar('train_loss_patches/normal_mask_coverage', normal_stats["mask_coverage"], iteration)
            logger.add_scalar('train_loss_patches/normal_base_mask_pixels', normal_stats["base_mask_pixels"], iteration)
            logger.add_scalar('train_loss_patches/normal_alpha_min', normal_stats["alpha_min"], iteration)
            logger.add_scalar('train_loss_patches/normal_sign', normal_stats["sign"], iteration)
            logger.add_scalar('train_loss_patches/normal_start_iter', normal_stats["start_iter"], iteration)
            logger.add_scalar('train_loss_patches/normal_confidence_enabled', float(normal_stats["confidence_enabled"]), iteration)
            logger.add_scalar('train_loss_patches/normal_confidence_sum', normal_stats["confidence_sum"], iteration)
            logger.add_scalar('train_loss_patches/normal_confidence_mean', normal_stats["confidence_mean"], iteration)
            logger.add_scalar('train_loss_patches/normal_confidence_pixels', normal_stats["confidence_pixels"], iteration)
            logger.add_scalar('train_loss_patches/normal_confidence_coverage', normal_stats["confidence_coverage"], iteration)
            logger.add_scalar('train_loss_patches/normal_confidence_depth_available', normal_stats["confidence_depth_available"], iteration)
            logger.add_scalar('train_loss_patches/normal_confidence_depth_mean', normal_stats["confidence_depth_mean"], iteration)
            logger.add_scalar('train_loss_patches/normal_confidence_edge_threshold', normal_stats["confidence_edge_threshold"], iteration)
            logger.add_scalar('train_loss_patches/normal_confidence_edge_keep', normal_stats["confidence_edge_keep"], iteration)
            logger.add_scalar('train_loss_patches/normal_confidence_depth_angle_deg', normal_stats["confidence_depth_angle_deg"], iteration)
            logger.add_scalar('train_loss_patches/normal_confidence_edge_quantile', normal_stats["confidence_edge_quantile"], iteration)
            logger.add_scalar('train_loss_patches/normal_confidence_edge_dilate_px', normal_stats["confidence_edge_dilate_px"], iteration)
            logger.add_scalar('train_loss_patches/normal_confidence_edge_floor', normal_stats["confidence_edge_floor"], iteration)
            logger.add_scalar('train_loss_patches/normal_confidence_depth_floor', normal_stats["confidence_depth_floor"], iteration)
            logger.add_scalar('train_loss_patches/normal_confidence_min', normal_stats["confidence_min"], iteration)
            logger.add_scalar('train_time/normal_reg', normal_stats["time_normal_reg"], iteration)
        if pseudo_stats and pseudo_stats.get("enabled", False):
            logger.add_scalar('train_loss_patches/pseudo_view_loss', pseudo_stats["loss"], iteration)
            logger.add_scalar('train_loss_patches/pseudo_view_loss_pure', pseudo_stats["loss_pure"], iteration)
            logger.add_scalar('train_loss_patches/pseudo_view_weight', pseudo_stats["weight"], iteration)
            logger.add_scalar('train_loss_patches/pseudo_view_active', float(pseudo_stats["active"]), iteration)
            logger.add_scalar('train_loss_patches/pseudo_view_applied', float(pseudo_stats["applied"]), iteration)
            logger.add_scalar('train_loss_patches/pseudo_view_valid_depth', float(pseudo_stats["valid_depth"]), iteration)
            logger.add_scalar('train_loss_patches/pseudo_view_mask_pixels', pseudo_stats["mask_pixels"], iteration)
            logger.add_scalar('train_loss_patches/pseudo_view_mask_coverage', pseudo_stats["mask_coverage"], iteration)
            logger.add_scalar('train_loss_patches/pseudo_view_mask_enough', float(pseudo_stats["mask_enough"]), iteration)
            logger.add_scalar('train_loss_patches/pseudo_view_shift', pseudo_stats["shift"], iteration)
            logger.add_scalar('train_loss_patches/pseudo_view_shift_ratio', pseudo_stats["shift_ratio"], iteration)
            logger.add_scalar('train_loss_patches/pseudo_view_depth_rel_thresh', pseudo_stats["depth_rel_thresh"], iteration)
            logger.add_scalar('train_loss_patches/pseudo_view_start_iter', pseudo_stats["start_iter"], iteration)
            logger.add_scalar('train_loss_patches/pseudo_view_end_iter', pseudo_stats["end_iter"], iteration)
            logger.add_scalar('train_loss_patches/pseudo_view_interval', pseudo_stats["interval"], iteration)
            logger.add_scalar('train_time/pseudo_view_reg', pseudo_stats["time_pseudo_reg"], iteration)
        if pseudo_normal_stats and pseudo_normal_stats.get("enabled", False):
            logger.add_scalar('train_loss_patches/pseudo_normal_loss', pseudo_normal_stats["loss"], iteration)
            logger.add_scalar('train_loss_patches/pseudo_normal_loss_pure', pseudo_normal_stats["loss_pure"], iteration)
            logger.add_scalar('train_loss_patches/pseudo_normal_weight', pseudo_normal_stats["weight"], iteration)
            logger.add_scalar('train_loss_patches/pseudo_normal_active', float(pseudo_normal_stats["active"]), iteration)
            logger.add_scalar('train_loss_patches/pseudo_normal_applied', float(pseudo_normal_stats["applied"]), iteration)
            logger.add_scalar('train_loss_patches/pseudo_normal_reliable', float(pseudo_normal_stats["reliable"]), iteration)
            logger.add_scalar('train_loss_patches/pseudo_normal_valid_depth', float(pseudo_normal_stats["valid_depth"]), iteration)
            logger.add_scalar('train_loss_patches/pseudo_normal_mask_pixels', pseudo_normal_stats["mask_pixels"], iteration)
            logger.add_scalar('train_loss_patches/pseudo_normal_mask_coverage', pseudo_normal_stats["mask_coverage"], iteration)
            logger.add_scalar('train_loss_patches/pseudo_normal_mask_enough', float(pseudo_normal_stats["mask_enough"]), iteration)
            logger.add_scalar('train_loss_patches/pseudo_normal_alpha_min', pseudo_normal_stats["alpha_min"], iteration)
            logger.add_scalar('train_loss_patches/pseudo_normal_shift', pseudo_normal_stats["shift"], iteration)
            logger.add_scalar('train_loss_patches/pseudo_normal_shift_ratio', pseudo_normal_stats["shift_ratio"], iteration)
            logger.add_scalar('train_loss_patches/pseudo_normal_ref_depth_is_mono', pseudo_normal_stats["ref_depth_is_mono"], iteration)
            logger.add_scalar('train_loss_patches/pseudo_normal_ref_depth_has_mask', pseudo_normal_stats["ref_depth_has_mask"], iteration)
            logger.add_scalar('train_loss_patches/pseudo_normal_start_iter', pseudo_normal_stats["start_iter"], iteration)
            logger.add_scalar('train_loss_patches/pseudo_normal_end_iter', pseudo_normal_stats["end_iter"], iteration)
            logger.add_scalar('train_loss_patches/pseudo_normal_interval', pseudo_normal_stats["interval"], iteration)
            logger.add_scalar('train_time/pseudo_normal_reg', pseudo_normal_stats["time_pseudo_normal_reg"], iteration)
        logger.add_scalar('train_time/render', ema_time["render"], iteration)
        logger.add_scalar('train_time/loss', ema_time["loss"], iteration)
        logger.add_scalar('train_time/densify', ema_time["densify"], iteration)
        logger.add_scalar('train_time/num_points', ema_time["num_points"], iteration)
        logger.add_scalar('train_time/mean_grad', ema_time["mean_grad"], iteration)
        logger.add_scalar('iter_time', elapsed, iteration)
        log_training_observations(logger, iteration, scene.gaussians, radii, visibility_filter)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        log_depth_images = bool(getattr(dataset_args, "depths", "") or getattr(dataset_args, "test_depths", ""))
        train_infos = scene.getTrainCameraInfos()
        train_indices = [idx % len(train_infos) for idx in range(5, 30, 5)] if train_infos else []
        validation_configs = (
            {
                'name': 'test',
                'cameras': make_validation_dataset(scene, dataset_args, scene.getTestCameraInfos(), True),
            },
            {
                'name': 'train',
                'cameras': make_validation_dataset(scene, dataset_args, train_infos, False, train_indices),
            },
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    try:
                        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                        gt_image = torch.clamp(image_to_cuda_float(viewpoint.original_image), 0.0, 1.0)
                        if train_test_exp:
                            image = image[..., image.shape[-1] // 2:]
                            gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                        if logger and (idx < 5):
                            tag_prefix = config['name'] + "_view_{}".format(viewpoint.image_name)
                            logger.add_images(tag_prefix + "/render", image[None], global_step=iteration)
                            if log_depth_images:
                                log_validation_depth_images(
                                    logger,
                                    tag_prefix,
                                    iteration,
                                    render_pkg,
                                    viewpoint,
                                    iteration == testing_iterations[0],
                                )
                            if iteration == testing_iterations[0]:
                                logger.add_images(tag_prefix + "/ground_truth", gt_image[None], global_step=iteration)
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                    finally:
                        viewpoint.release_image()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if logger:
                    logger.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    logger.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if logger:
            logger.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            logger.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    cli_args = parser.parse_args(sys.argv[1:])
    args = cli_args
    if cli_args.config:
        defaults = parser.parse_args([])
        cfg = load_yaml_config(cli_args.config, cli_args.override)
        cfg_args = stage_args_from_config(cfg, "train", block_id=cli_args.block_id or None)
        cfg_args["config"] = os.path.abspath(cli_args.config)
        args = namespace_from_config(defaults, cfg_args, resolved_config=cfg)
    if not isinstance(args.save_iterations, list):
        args.save_iterations = list(args.save_iterations)
    args.save_iterations.append(args.iterations)

    logger = prepare_output_and_logger(args)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, logger)

    # All done
    print("\nTraining complete.")
