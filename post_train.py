#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#

import json
import os
import sys
import time
import uuid
from argparse import ArgumentParser, Namespace

import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, load_external_test_scene_info, load_scene_info_from_path
from scene.datasets import CameraDataLoader, GSCameraDataset
from utils.camera_utils import camera_to_JSON
from utils.config_utils import (
    load_yaml_config,
    namespace_from_config,
    save_yaml_config,
    stage_args_from_config,
)
from utils.general_utils import get_expon_lr_func, safe_state
from utils.image_utils import image_to_cuda_float
from utils.loss_utils import l1_loss, ssim
from utils.partition_utils import all_camera_infos, mark_camera_infos_as_test
from utils.post_train_utils import build_boundary_mask, input_ply_path, save_boundary_report

try:
    import swanlab

    SWANLAB_FOUND = True
except ImportError:
    swanlab = None
    SWANLAB_FOUND = False

try:
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


class SwanLabLogger:
    def __init__(self, enabled):
        self.enabled = enabled

    def __bool__(self):
        return self.enabled

    def add_scalar(self, tag, scalar_value, global_step=None):
        if self.enabled:
            swanlab.log({tag: float(scalar_value)}, step=global_step)

    def close(self):
        if self.enabled:
            swanlab.finish()


def prepare_output_and_logger(args):
    if not getattr(args, "output_path", ""):
        args.output_path = getattr(args, "model_path", "")
    if not args.output_path:
        job_id = os.getenv("OAR_JOB_ID")
        unique_str = job_id if job_id else str(uuid.uuid4())
        args.output_path = os.path.join("./output/", unique_str[:10])
    args.model_path = args.output_path

    print(f"Output folder: {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)

    resolved_config = getattr(args, "resolved_config", None)
    if resolved_config:
        save_yaml_config(os.path.join(args.output_path, "resolved_config.yaml"), resolved_config)

    cfg_args = vars(args).copy()
    cfg_args.pop("resolved_config", None)
    with open(os.path.join(args.output_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**cfg_args)))

    logger = SwanLabLogger(enabled=SWANLAB_FOUND)
    if SWANLAB_FOUND:
        try:
            logdir = args.swanlab_logdir or os.path.join(args.output_path, "swanlog")
            os.makedirs(logdir, exist_ok=True)
            swanlab.init(
                project=args.swanlab_project or None,
                workspace=args.swanlab_workspace or None,
                experiment_name=args.swanlab_experiment_name or None,
                mode=args.swanlab_mode or None,
                config=cfg_args,
                logdir=logdir,
            )
        except Exception as exc:
            logger.enabled = False
            print(f"SwanLab initialization failed: {exc}. Continuing without SwanLab logging.")
    else:
        print("SwanLab not available: not logging progress")
    return logger


def load_post_train_scene_info(args):
    scene_info = load_scene_info_from_path(
        args.source_path,
        args.images,
        args.depths,
        getattr(args, "normals", ""),
        args.eval,
        args.train_test_exp,
        args.white_background,
    )
    test_scene_info = load_external_test_scene_info(args)
    if test_scene_info is not None:
        test_cameras = mark_camera_infos_as_test(all_camera_infos(test_scene_info))
        scene_info = scene_info._replace(test_cameras=test_cameras)
        print(f"[External Test] loaded test_cameras={len(test_cameras)} from {args.test_source_path}")
    return scene_info


def write_cameras_json(output_path, scene_info):
    camlist = []
    camlist.extend(scene_info.test_cameras or [])
    camlist.extend(scene_info.train_cameras or [])
    json_cams = [camera_to_JSON(idx, cam) for idx, cam in enumerate(camlist)]
    with open(os.path.join(output_path, "cameras.json"), "w") as file:
        json.dump(json_cams, file)


def resolve_boundary_paths(args):
    merge_report_path = getattr(args, "merge_report_path", "")
    if not merge_report_path:
        merge_report_path = os.path.join(args.input_model_path, "merge_report.json")
    if not os.path.isfile(merge_report_path):
        raise FileNotFoundError(f"merge_report not found: {merge_report_path}")

    partition_path = getattr(args, "partition_path", "")
    if not partition_path:
        with open(merge_report_path, "r") as f:
            partition_path = json.load(f).get("partition_tree", "")
    if not partition_path:
        raise ValueError("--partition_path is required when merge_report has no partition_tree field")
    return partition_path, merge_report_path


def save_model(output_path, iteration, gaussians):
    point_cloud_path = os.path.join(output_path, "point_cloud", f"iteration_{iteration}")
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    exposure_dict = {}
    for image_name in getattr(gaussians, "exposure_mapping", {}):
        exposure_dict[image_name] = gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
    with open(os.path.join(output_path, "exposure.json"), "w") as f:
        json.dump(exposure_dict, f, indent=2)


def scale_grad_rows(param, boundary_mask, internal_mask, boundary_scale=1.0, internal_scale=1.0):
    if param.grad is None:
        return
    if boundary_scale == 1.0 and internal_scale == 1.0:
        return

    row_scale = torch.empty(boundary_mask.shape[0], dtype=param.grad.dtype, device=param.grad.device)
    row_scale[boundary_mask] = boundary_scale
    row_scale[internal_mask] = internal_scale
    view_shape = (row_scale.shape[0],) + (1,) * (param.grad.dim() - 1)
    param.grad.mul_(row_scale.view(view_shape))


def apply_post_train_grad_policy(gaussians, boundary_mask, internal_mask, args):
    geom_scale = float(getattr(args, "boundary_geom_grad_scale", 1.0))
    color_scale = float(getattr(args, "internal_color_grad_scale", 1.0))
    opacity_scale = float(getattr(args, "internal_opacity_grad_scale", 1.0))

    scale_grad_rows(gaussians._xyz, boundary_mask, internal_mask, boundary_scale=geom_scale, internal_scale=0)
    scale_grad_rows(gaussians._scaling, boundary_mask, internal_mask, boundary_scale=geom_scale, internal_scale=0)
    scale_grad_rows(gaussians._rotation, boundary_mask, internal_mask, boundary_scale=geom_scale, internal_scale=0)
    scale_grad_rows(gaussians._features_dc, boundary_mask, internal_mask, internal_scale=color_scale)
    scale_grad_rows(gaussians._features_rest, boundary_mask, internal_mask, internal_scale=color_scale)
    scale_grad_rows(gaussians._opacity, boundary_mask, internal_mask, internal_scale=opacity_scale)


def make_camera_loader(scene_info, dataset_args):
    train_dataset = GSCameraDataset(
        scene_info.train_cameras,
        dataset_args,
        scene_info.is_nerf_synthetic,
        is_test_dataset=False,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("No training cameras found")

    max_cache_num = int(getattr(dataset_args, "max_cache_num", 0))
    print(
        "[DataLoader] "
        f"train_cameras={len(train_dataset)}, "
        f"max_cache_num={max_cache_num}, "
        f"cache_workers={getattr(dataset_args, 'image_cache_workers', 0)}"
    )
    return CameraDataLoader(
        train_dataset,
        batch_size=1,
        max_cache_num=max_cache_num,
        cache_workers=getattr(dataset_args, "image_cache_workers", 0),
        shuffle=True,
        seed=getattr(dataset_args, "image_loader_seed", 42),
        num_workers=0,
    )


def log_iteration(logger, iteration, loss, Ll1, Ll1depth, radii, gaussians, ema_time):
    if not logger or iteration % 10 != 0:
        return
    visible = int((radii > 0).sum().item())
    total = int(gaussians.get_xyz.shape[0])
    logger.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
    logger.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
    logger.add_scalar("train_loss_patches/depth_loss", float(Ll1depth), iteration)
    logger.add_scalar("scene/total_points", total, iteration)
    logger.add_scalar("scene/visible_points", visible, iteration)
    logger.add_scalar("scene/visibility_ratio", visible / max(total, 1), iteration)
    logger.add_scalar("train_time/render", ema_time["render"], iteration)
    logger.add_scalar("train_time/loss", ema_time["loss"], iteration)
    if torch.cuda.is_available():
        logger.add_scalar("gpu/memory_allocated_gb", torch.cuda.memory_allocated() / 1024**3, iteration)
        logger.add_scalar("gpu/memory_reserved_gb", torch.cuda.memory_reserved() / 1024**3, iteration)


def training(dataset, opt, pipe, args, logger):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(
            "Trying to use sparse adam but it is not installed, "
            "please install the correct rasterizer using pip install [3dgs_accel]."
        )

    if int(getattr(opt, "densify_until_iter", 0)) > 0:
        print("[PostTrain] Densify/prune is disabled in post_train.py")

    scene_info = load_post_train_scene_info(dataset)
    write_cameras_json(args.output_path, scene_info)
    cameras_extent = scene_info.nerf_normalization["radius"]

    ply_path = input_ply_path(args.input_model_path, args.input_iteration)
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"Input merged PLY not found: {ply_path}")

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    print(f"[PostTrain] Loading merged Gaussians from {ply_path}")
    gaussians.load_ply(ply_path, dataset.train_test_exp)
    gaussians.prepare_loaded_ply_for_training(scene_info.train_cameras, cameras_extent)
    gaussians.training_setup(opt)
    print(f"[PostTrain] Loaded Gaussians: {gaussians.get_xyz.shape[0]}")

    partition_path, merge_report_path = resolve_boundary_paths(args)
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    boundary_mask_np, boundary_report = build_boundary_mask(
        xyz,
        partition_path=partition_path,
        merge_report_path=merge_report_path,
        band_ratio=args.boundary_band_ratio,
        axes=args.boundary_axes,
    )
    save_boundary_report(os.path.join(args.output_path, "post_train_boundary_report.json"), boundary_report)
    boundary_mask = torch.from_numpy(boundary_mask_np).to(device="cuda", dtype=torch.bool)
    internal_mask = ~boundary_mask
    print(
        "[PostTrain] Boundary mask: "
        f"boundary={boundary_report['boundary_points']} "
        f"internal={boundary_report['internal_points']} "
        f"ratio={boundary_report['boundary_ratio']:.4f}"
    )
    del xyz, boundary_mask_np

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE

    camera_loader = make_camera_loader(scene_info, dataset)
    camera_loader_iter = iter(camera_loader)
    release_viewpoint_after_iter = int(getattr(dataset, "max_cache_num", 0)) == 0
    saving_iterations = set(int(i) for i in args.save_iterations)
    checkpoint_iterations = set(int(i) for i in args.checkpoint_iterations)

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    ema_time_render = 0.0
    ema_time_loss = 0.0
    progress_bar = tqdm(range(1, opt.iterations + 1), desc="Post-training progress")

    try:
        for iteration in range(1, opt.iterations + 1):
            gaussians.update_learning_rate(iteration)

            try:
                viewpoint_cam = next(camera_loader_iter)
            except StopIteration:
                camera_loader_iter = iter(camera_loader)
                viewpoint_cam = next(camera_loader_iter)

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            start = time.time()
            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                bg,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
            )
            image = render_pkg["render"]
            radii = render_pkg["radii"]
            ema_time_render = 0.4 * (time.time() - start) + 0.6 * ema_time_render

            start = time.time()
            if viewpoint_cam.alpha_mask is not None:
                image *= image_to_cuda_float(viewpoint_cam.alpha_mask)

            gt_image = image_to_cuda_float(viewpoint_cam.original_image)
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            Ll1depth = 0.0
            if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                invDepth = render_pkg["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()
                Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
                loss += Ll1depth
                Ll1depth = Ll1depth.item()

            loss.backward()
            apply_post_train_grad_policy(gaussians, boundary_mask, internal_mask, args)
            ema_time_loss = 0.4 * (time.time() - start) + 0.6 * ema_time_loss

            if use_sparse_adam:
                gaussians.optimizer.step(radii > 0, radii.shape[0])
                gaussians.optimizer.zero_grad(set_to_none=True)
            else:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            gaussians.exposure_optimizer.step()
            gaussians.exposure_optimizer.zero_grad(set_to_none=True)

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * float(Ll1depth) + 0.6 * ema_Ll1depth_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.7f}",
                        "Depth Loss": f"{ema_Ll1depth_for_log:.7f}",
                    }
                )
                progress_bar.update(10)

            log_iteration(
                logger,
                iteration,
                loss,
                Ll1,
                Ll1depth,
                radii,
                gaussians,
                {"render": ema_time_render, "loss": ema_time_loss},
            )

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving post-trained Gaussians")
                save_model(args.output_path, iteration, gaussians)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), os.path.join(args.output_path, f"chkpnt{iteration}.pth"))

            if release_viewpoint_after_iter:
                viewpoint_cam.release_image()
    finally:
        if hasattr(camera_loader_iter, "close"):
            camera_loader_iter.close()
        progress_bar.close()
        logger.close()


def parse_args():
    parser = ArgumentParser(description="Post-train merged Gaussian point cloud")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--input_model_path", type=str, default="")
    parser.add_argument("--input_iteration", type=int, default=30000)
    parser.add_argument("--merge_report_path", type=str, default="")
    parser.add_argument("--boundary_band_ratio", type=float, default=0.05)
    parser.add_argument("--boundary_axes", nargs="+", default=["x", "y"])
    parser.add_argument("--boundary_geom_grad_scale", type=float, default=1.0)
    parser.add_argument("--internal_color_grad_scale", type=float, default=1.0)
    parser.add_argument("--internal_opacity_grad_scale", type=float, default=1.0)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    args = parser.parse_args(sys.argv[1:])

    if args.config:
        defaults = parser.parse_args([])
        cfg = load_yaml_config(args.config, args.override)
        cfg_args = stage_args_from_config(cfg, "post_train")
        cfg_args["config"] = os.path.abspath(args.config)
        args = namespace_from_config(defaults, cfg_args, resolved_config=cfg)

    if not args.input_model_path:
        raise ValueError("--input_model_path is required")
    args.input_model_path = os.path.abspath(args.input_model_path)
    if getattr(args, "partition_path", ""):
        args.partition_path = os.path.abspath(args.partition_path)
    if getattr(args, "merge_report_path", ""):
        args.merge_report_path = os.path.abspath(args.merge_report_path)

    args.save_iterations = sorted(set(int(i) for i in list(args.save_iterations) + [int(args.iterations)]))
    args.checkpoint_iterations = [int(i) for i in args.checkpoint_iterations]
    args.test_iterations = [int(i) for i in args.test_iterations]
    return args, lp, op, pp


if __name__ == "__main__":
    args, lp, op, pp = parse_args()
    logger = prepare_output_and_logger(args)
    print("Post-training " + args.output_path)

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(getattr(args, "detect_anomaly", False))
    training(lp.extract(args), op.extract(args), pp.extract(args), args, logger)

    print("\nPost-training complete.")
