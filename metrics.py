#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#

from argparse import ArgumentParser
from pathlib import Path
import json
import os

from PIL import Image
import torch
import torchvision.transforms.functional as tf
from tqdm import tqdm

from lpipsPyTorch.modules.lpips import LPIPS
from utils.config_utils import (
    load_yaml_config,
    namespace_from_config,
    stage_args_from_config,
)
from utils.image_utils import psnr
from utils.loss_utils import ssim


def load_image(path, device):
    with Image.open(path) as image:
        return tf.to_tensor(image).unsqueeze(0)[:, :3, :, :].to(device)


def evaluate(model_paths, test_dir_name="test", lpips_net="vgg"):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lpips_metric = LPIPS(lpips_net).to(device).eval()

    for scene_dir in model_paths:
        scene_path = Path(scene_dir)
        test_dir = scene_path / test_dir_name

        full_dict = {}
        per_view_dict = {}

        print("")
        print("Scene:", scene_dir)

        for method in sorted(os.listdir(test_dir)):
            method_dir = test_dir / method
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"
            if not gt_dir.is_dir() or not renders_dir.is_dir():
                continue

            print("Method:", method)
            image_names = sorted(fname for fname in os.listdir(renders_dir) if (gt_dir / fname).is_file())

            ssims = []
            psnrs = []
            lpipss = []

            with torch.no_grad():
                for name in tqdm(image_names, desc="Metric evaluation progress"):
                    render = load_image(renders_dir / name, device)
                    gt = load_image(gt_dir / name, device)

                    ssims.append(ssim(render, gt).item())
                    psnrs.append(psnr(render, gt).mean().item())
                    lpipss.append(lpips_metric(render, gt).mean().item())

                    del render, gt

            mean_ssim = float(torch.tensor(ssims).mean().item())
            mean_psnr = float(torch.tensor(psnrs).mean().item())
            mean_lpips = float(torch.tensor(lpipss).mean().item())

            print("  SSIM : {:>12.7f}".format(mean_ssim))
            print("  PSNR : {:>12.7f}".format(mean_psnr))
            print("  LPIPS: {:>12.7f}".format(mean_lpips))
            print("")

            full_dict[method] = {
                "SSIM": mean_ssim,
                "PSNR": mean_psnr,
                "LPIPS": mean_lpips,
            }
            per_view_dict[method] = {
                "SSIM": dict(zip(image_names, ssims)),
                "PSNR": dict(zip(image_names, psnrs)),
                "LPIPS": dict(zip(image_names, lpipss)),
            }

        with open(scene_path / "results.json", "w") as fp:
            json.dump(full_dict, fp, indent=True)
        with open(scene_path / "per_view.json", "w") as fp:
            json.dump(per_view_dict, fp, indent=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Metrics script parameters")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--model_paths", "-m", nargs="+", type=str, default=[])
    parser.add_argument("--test_dir_name", "-t", "-d", default="test", type=str)
    parser.add_argument("--lpips_net", default="vgg", choices=["alex", "squeeze", "vgg"])
    cli_args = parser.parse_args()
    args = cli_args
    if cli_args.config:
        defaults = parser.parse_args([])
        cfg = load_yaml_config(cli_args.config, cli_args.override)
        cfg_args = stage_args_from_config(cfg, "metrics")
        cfg_args["config"] = os.path.abspath(cli_args.config)
        args = namespace_from_config(defaults, cfg_args, resolved_config=cfg)
    if not args.model_paths:
        raise ValueError("Missing required metrics argument: model_paths")
    evaluate(args.model_paths, args.test_dir_name, args.lpips_net)
