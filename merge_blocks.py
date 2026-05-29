from __future__ import annotations

import argparse
import os
from argparse import Namespace

from plyfile import PlyData, PlyElement
import numpy as np

from utils.config_utils import (
    load_yaml_config,
    namespace_from_config,
    save_yaml_config,
    stage_args_from_config,
)
from utils.partition_utils import load_partition_tree, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Merge recursively trained Gaussian blocks")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--partition_path", default="")
    parser.add_argument("--blocks_root", default="")
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--output_path", default="")
    parser.add_argument("--allow_missing", action="store_true", default=False)
    parser.add_argument(
        "--cfg_args_source",
        default="",
        help="Optional cfg_args file or model directory to use for the merged model. "
        "If omitted, infer it from partition_tree.coarse_model.",
    )
    cli_args = parser.parse_args()
    args = cli_args
    if cli_args.config:
        defaults = parser.parse_args([])
        cfg = load_yaml_config(cli_args.config, cli_args.override)
        cfg_args = stage_args_from_config(cfg, "merge")
        cfg_args["config"] = os.path.abspath(cli_args.config)
        args = namespace_from_config(defaults, cfg_args, resolved_config=cfg)
    missing = [name for name in ("partition_path", "blocks_root", "iteration", "output_path") if getattr(args, name, None) in ("", None)]
    if missing:
        raise ValueError(f"Missing required merge argument(s): {', '.join(missing)}")
    return args


def block_ply_path(blocks_root: str, block_id: str, iteration: int) -> str:
    return os.path.join(
        blocks_root,
        block_id,
        "point_cloud",
        f"iteration_{iteration}",
        "point_cloud.ply",
    )


def coarse_model_dir_from_ply(path: str) -> str:
    path = os.path.abspath(path)
    if os.path.basename(path) != "point_cloud.ply":
        return path

    iteration_dir = os.path.dirname(path)
    point_cloud_dir = os.path.dirname(iteration_dir)
    if os.path.basename(point_cloud_dir) == "point_cloud":
        return os.path.dirname(point_cloud_dir)
    return os.path.dirname(path)


def cfg_args_candidates(cfg_args_source: str, partition_tree: dict) -> list[str]:
    candidates = []
    if cfg_args_source:
        source = os.path.abspath(cfg_args_source)
        candidates.append(os.path.join(source, "cfg_args") if os.path.isdir(source) else source)

    coarse_model = partition_tree.get("coarse_model") or partition_tree.get("config", {}).get("coarse_model", "")
    if coarse_model:
        candidates.append(os.path.join(coarse_model_dir_from_ply(coarse_model), "cfg_args"))

    return candidates


def read_cfg_args(path: str) -> dict:
    with open(path) as f:
        cfg = eval(f.read(), {"Namespace": Namespace})
    if not isinstance(cfg, Namespace):
        raise ValueError(f"cfg_args did not contain an argparse Namespace: {path}")
    return vars(cfg).copy()


def fallback_cfg_args(partition_tree: dict, output_path: str) -> dict:
    config = partition_tree.get("config", {})
    return {
        "sh_degree": 3,
        "source_path": partition_tree.get("source_path") or config.get("source_path", ""),
        "model_path": output_path,
        "images": config.get("images", "images"),
        "depths": config.get("depths", ""),
        "resolution": -1,
        "white_background": config.get("white_background", False),
        "train_test_exp": config.get("train_test_exp", False),
        "data_device": "cpu",
        "camera_load_workers": 0,
        "partition_path": "",
        "block_id": "",
        "partition_coord_space": "world",
        "contract_aabb": None,
        "partition_bbox_mode": "expanded",
        "partition_init_mode": "cropped",
        "partition_load_test_cameras": False,
        "eval": config.get("eval", False),
        "convert_SHs_python": False,
        "compute_cov3D_python": False,
        "debug": False,
        "antialiasing": False,
    }


def write_merged_cfg_args(
    output_path: str,
    partition_tree: dict,
    cfg_args_source: str,
) -> dict:
    candidates = cfg_args_candidates(cfg_args_source, partition_tree)
    source_path = ""
    cfg = None
    for candidate in candidates:
        if os.path.isfile(candidate):
            source_path = candidate
            cfg = read_cfg_args(candidate)
            break

    if cfg is None:
        cfg = fallback_cfg_args(partition_tree, output_path)
        print("[MERGE] cfg_args source not found; writing fallback cfg_args from partition metadata")
    else:
        print(f"[MERGE] Using cfg_args source: {source_path}")

    cfg["model_path"] = output_path
    cfg["partition_path"] = ""
    cfg["block_id"] = ""
    cfg["partition_coord_space"] = "world"
    cfg["contract_aabb"] = None
    cfg["partition_bbox_mode"] = "expanded"
    cfg["partition_init_mode"] = "cropped"
    cfg["partition_load_test_cameras"] = False

    cfg_output = os.path.join(output_path, "cfg_args")
    with open(cfg_output, "w") as f:
        f.write(str(Namespace(**cfg)))

    return {
        "output": cfg_output,
        "source": source_path,
        "candidates": candidates,
        "sanitized_fields": [
            "model_path",
            "partition_path",
            "block_id",
            "partition_bbox_mode",
            "partition_init_mode",
            "partition_load_test_cameras",
        ],
    }


def main():
    args = parse_args()
    partition_tree, tree_path = load_partition_tree(args.partition_path)
    blocks_root = os.path.abspath(args.blocks_root)
    output_path = os.path.abspath(args.output_path)
    if getattr(args, "resolved_config", None):
        save_yaml_config(os.path.join(output_path, "resolved_config.yaml"), args.resolved_config)

    merged_vertices = []
    report = {
        "partition_tree": tree_path,
        "blocks_root": blocks_root,
        "iteration": args.iteration,
        "blocks": [],
    }

    dtype = None
    for block in partition_tree.get("blocks", []):
        block_id = block["id"]
        ply_path = block_ply_path(blocks_root, block_id, args.iteration)
        if not os.path.isfile(ply_path):
            if args.allow_missing:
                report["blocks"].append(
                    {
                        "id": block_id,
                        "ply_path": ply_path,
                        "status": "missing",
                        "kept_gaussians": 0,
                        "discarded_gaussians": 0,
                    }
                )
                continue
            raise FileNotFoundError(f"Missing block PLY: {ply_path}")

        plydata = PlyData.read(ply_path)
        vertex_data = plydata["vertex"].data
        if dtype is None:
            dtype = vertex_data.dtype
        elif vertex_data.dtype != dtype:
            raise ValueError(f"PLY dtype mismatch in {ply_path}")

        kept = int(vertex_data.shape[0])
        discarded = 0
        if kept > 0:
            merged_vertices.append(vertex_data)
        report["blocks"].append(
            {
                "id": block_id,
                "ply_path": ply_path,
                "status": "merged",
                "kept_gaussians": kept,
                "discarded_gaussians": discarded,
            }
        )
        print(f"[MERGE] {block_id}: appended={kept}")

    if not merged_vertices:
        raise RuntimeError("No Gaussian vertices were merged")

    merged = np.concatenate(merged_vertices)
    point_cloud_dir = os.path.join(output_path, "point_cloud", f"iteration_{args.iteration}")
    os.makedirs(point_cloud_dir, exist_ok=True)
    output_ply = os.path.join(point_cloud_dir, "point_cloud.ply")
    PlyData([PlyElement.describe(merged, "vertex")]).write(output_ply)

    report["output_ply"] = output_ply
    report["total_kept_gaussians"] = int(sum(item["kept_gaussians"] for item in report["blocks"]))
    report["total_discarded_gaussians"] = int(sum(item["discarded_gaussians"] for item in report["blocks"]))
    report["cfg_args"] = write_merged_cfg_args(output_path, partition_tree, args.cfg_args_source)
    save_json(os.path.join(output_path, "merge_report.json"), report)

    print(f"Merged PLY written to {output_ply}")
    print(f"Total kept Gaussians: {report['total_kept_gaussians']}")


if __name__ == "__main__":
    main()
