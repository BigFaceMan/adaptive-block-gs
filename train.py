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
import torch
import time
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from scene.datasets import CameraDataLoader, GSCameraDataset
from utils.block_depth_mask import BlockDepthMasker
from utils.general_utils import safe_state, get_expon_lr_func
from utils.config_utils import (
    load_yaml_config,
    namespace_from_config,
    save_yaml_config,
    stage_args_from_config,
)
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
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
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            end = time.time()
            ema_time_render = 0.4 * (end - start) + 0.6 * ema_time_render

            start = time.time()
            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                image *= alpha_mask

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
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
            if depth_weight_value > 0 and depth_reliable:
                invDepth = render_pkg["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()
                if block_depth_masker is not None:
                    mask_start = time.time()
                    block_mask_cpu, depth_mask_cache_hit = block_depth_masker.mask_for(viewpoint_cam)
                    if not depth_mask_cache_hit:
                        depth_mask_project_time = time.time() - mask_start
                    depth_mask_cache_size = block_depth_masker.cache_size
                    transfer_start = time.time()
                    block_mask = block_mask_cpu.to(
                        device=depth_mask.device,
                        dtype=depth_mask.dtype,
                    )
                    depth_mask_transfer_time = time.time() - transfer_start
                    depth_mask = depth_mask * block_mask

                valid_pixels = depth_mask.sum()
                depth_mask_pixels = float(valid_pixels.detach().item())
                depth_mask_coverage = depth_mask_pixels / max(float(depth_mask.numel()), 1.0)
                min_depth_pixels = float(getattr(opt, "depth_reg_mask_min_pixels", 0))
                depth_mask_enough = depth_mask_pixels >= min_depth_pixels

                if depth_mask_enough:
                    Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).sum() / valid_pixels.clamp_min(1.0)
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

            loss.backward()
            end = time.time()
            ema_time_loss = 0.4 * (end - start) + 0.6 * ema_time_loss

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
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
                training_report(logger, iteration, Ll1, loss, l1_loss, ema_time, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), radii, visibility_filter, dataset.train_test_exp, dataset, depth_stats)
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

def training_report(logger, iteration, Ll1, loss, l1_loss, ema_time, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, radii, visibility_filter, train_test_exp, dataset_args=None, depth_stats=None):
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
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
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
