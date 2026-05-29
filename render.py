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

import torch
from scene import Scene
import os
import sys
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from scene.datasets import CameraDataLoader, GSCameraDataset
from utils.general_utils import safe_state
from utils.config_utils import (
    load_yaml_config,
    namespace_from_config,
    stage_args_from_config,
)
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        try:
            rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
            gt = view.original_image[0:3, :, :]

            if train_test_exp:
                rendering = rendering[..., rendering.shape[-1] // 2:]
                gt = gt[..., gt.shape[-1] // 2:]

            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        finally:
            view.release_image()


def make_render_loader(scene, dataset, camera_infos, is_test_dataset):
    camera_dataset = GSCameraDataset(
        camera_infos,
        dataset,
        scene.is_nerf_synthetic,
        is_test_dataset=is_test_dataset,
    )
    max_cache_num = int(getattr(dataset, "max_cache_num", 0))
    return CameraDataLoader(
        camera_dataset,
        batch_size=1,
        max_cache_num=max_cache_num,
        cache_workers=getattr(dataset, "image_cache_workers", 0),
        shuffle=False,
        seed=getattr(dataset, "image_loader_seed", 42),
        num_workers=0,
    )


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            train_views = make_render_loader(scene, dataset, scene.getTrainCameraInfos(), False)
            render_set(dataset.model_path, "train", scene.loaded_iter, train_views, gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
            test_views = make_render_loader(scene, dataset, scene.getTestCameraInfos(), True)
            render_set(dataset.model_path, "test", scene.loaded_iter, test_views, gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    cli_args = parser.parse_args(sys.argv[1:])
    if getattr(cli_args, "config", ""):
        defaults = parser.parse_args([])
        cfg = load_yaml_config(cli_args.config, cli_args.override)
        cfg_args = stage_args_from_config(cfg, "render")
        cfg_args["config"] = os.path.abspath(cli_args.config)
        args = namespace_from_config(defaults, cfg_args, resolved_config=cfg)
    else:
        args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)
