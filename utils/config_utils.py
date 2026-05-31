import copy
import os
from argparse import Namespace


try:
    import yaml
except ImportError:
    yaml = None


from utils.config_schema import ExperimentConfig, config_from_dict, config_to_dict, load_config


def load_yaml_config(path, overrides=None):
    return config_to_dict(load_config(path, overrides))


def save_yaml_config(path, config):
    if not config or yaml is None:
        return
    payload = config_to_dict(config)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def namespace_from_config(default_args, config_args, resolved_config=None):
    merged = vars(default_args).copy()
    merged.update(config_args)
    if resolved_config is not None:
        merged["resolved_config"] = resolved_config
    return Namespace(**merged)


def stage_args_from_config(config, stage, block_id=None):
    cfg = _ensure_config(config)
    if stage == "partition":
        return partition_args(cfg)
    if stage == "train":
        return train_args(cfg, block_id=block_id)
    if stage == "merge":
        return merge_args(cfg)
    if stage == "render":
        return render_args(cfg)
    if stage == "metrics":
        return metrics_args(cfg)
    raise ValueError(f"Unknown config stage: {stage}")


def config_name(config):
    return _ensure_config(config).experiment.name


def get_in(config, dotted_key, default=None):
    value = config_to_dict(_ensure_config(config))
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def partition_tree_path(config):
    return _ensure_config(config).partition_tree_path


def partition_args(cfg: ExperimentConfig):
    cfg = _ensure_config(cfg)
    args = common_dataset_args(cfg)
    args.update(
        {
            "partition_output": cfg.partition.output_path or os.path.join(cfg.output_root, "partitions"),
            "partition_coord_space": cfg.partition.coord_space,
            "contract_aabb": cfg.partition.contract_aabb,
            "partition_axes": cfg.partition.axes,
            "max_depth": cfg.partition.max_depth,
            "max_blocks": cfg.partition.max_blocks,
            "max_block_importance": cfg.partition.max_block_importance,
            "max_block_density": cfg.partition.max_block_density,
            "min_points": cfg.partition.min_points,
            "min_size": cfg.partition.min_size,
            "expand_ratio": cfg.partition.expand_ratio,
            "num_split_candidates": cfg.partition.num_split_candidates,
            "lambda_boundary": cfg.partition.lambda_boundary,
            "importance": cfg.partition.importance,
            "coarse_model": cfg.model.coarse_model,
            "tau_projection": cfg.camera_assignment.tau_projection,
            "tau_test_projection": cfg.camera_assignment.tau_test_projection,
            "min_cameras": cfg.camera_assignment.min_cameras,
            "min_test_cameras": cfg.camera_assignment.min_test_cameras,
            "supplement_cameras": cfg.camera_assignment.supplement_cameras,
            "camera_projection_max_points": cfg.camera_assignment.projection_max_points,
            "render_difference_cameras": cfg.camera_assignment.render_difference.enabled,
            "render_difference_threshold": cfg.camera_assignment.render_difference.threshold,
            "render_difference_max_candidates_per_block": cfg.camera_assignment.render_difference.max_candidates_per_block,
            "render_difference_max_width": cfg.camera_assignment.render_difference.max_width,
            "render_difference_cache_full": cfg.camera_assignment.render_difference.cache_full,
        }
    )
    args.update(vars(cfg.visualization))
    return args


def train_args(cfg: ExperimentConfig, block_id=None):
    cfg = _ensure_config(cfg)
    args = common_dataset_args(cfg)
    args.update(common_pipeline_args(cfg))
    args.update(vars(cfg.optimization))
    args.update(vars(cfg.training))
    args.update(
        {
            "test_iterations": cfg.block_training.test_iterations,
            "save_iterations": cfg.block_training.save_iterations,
            "checkpoint_iterations": cfg.block_training.checkpoint_iterations,
            "start_checkpoint": cfg.block_training.start_checkpoint,
        }
    )
    resolved_block_id = block_id or cfg.block_training.block_id
    if resolved_block_id:
        args.update(
            {
                "partition_path": cfg.partition_tree_path,
                "partition_bbox_mode": cfg.block_training.partition_bbox_mode,
                "partition_init_mode": cfg.block_training.partition_init_mode,
                "partition_load_test_cameras": cfg.block_training.partition_load_test_cameras,
            }
        )
        args["block_id"] = resolved_block_id
        args["model_path"] = args.get("model_path") or cfg.block_model_path(resolved_block_id)
        args["swanlab_experiment_name"] = args.get("swanlab_experiment_name") or f"{cfg.logging.swanlab_experiment_prefix}-{resolved_block_id}"
    else:
        args["model_path"] = args.get("model_path") or os.path.join(cfg.output_root, "train")
        args["swanlab_experiment_name"] = args.get("swanlab_experiment_name") or cfg.logging.swanlab_experiment_prefix
    return args


def merge_args(cfg: ExperimentConfig):
    cfg = _ensure_config(cfg)
    return {
        "partition_path": cfg.merge.partition_path or cfg.partition_tree_path,
        "blocks_root": cfg.merge.blocks_root or cfg.block_training.blocks_root or os.path.join(cfg.output_root, "blocks"),
        "iteration": cfg.merge.iteration if cfg.merge.iteration is not None else cfg.optimization.iterations,
        "output_path": cfg.merge_output_path,
        "allow_missing": cfg.merge.allow_missing,
        "cfg_args_source": cfg.merge.cfg_args_source,
    }


def render_args(cfg: ExperimentConfig):
    cfg = _ensure_config(cfg)
    args = common_dataset_args(cfg)
    args.update(common_pipeline_args(cfg))
    render_source_path = cfg.render.source_path or cfg.dataset.source_path
    render_depths = cfg.render.depths
    if not render_depths and os.path.abspath(render_source_path) == os.path.abspath(cfg.dataset.source_path):
        render_depths = cfg.dataset.depths
    args.update(
        {
            "model_path": cfg.render.model_path or cfg.merge_output_path,
            "source_path": render_source_path,
            "images": cfg.render.images or cfg.dataset.images,
            "depths": render_depths,
            "iteration": cfg.render.iteration,
            "skip_train": cfg.render.skip_train,
            "skip_test": cfg.render.skip_test,
            "render_depth": cfg.render.render_depth,
            "quiet": cfg.render.quiet,
        }
    )
    return args


def metrics_args(cfg: ExperimentConfig):
    cfg = _ensure_config(cfg)
    model_paths = cfg.metrics.model_paths or [cfg.metrics.model_path or cfg.render.model_path or cfg.merge_output_path]
    return {"model_paths": model_paths}


def common_dataset_args(cfg: ExperimentConfig):
    cfg = _ensure_config(cfg)
    args = vars(cfg.dataset).copy()
    args["sh_degree"] = cfg.model.sh_degree
    return args


def common_pipeline_args(cfg: ExperimentConfig):
    cfg = _ensure_config(cfg)
    args = vars(cfg.pipeline).copy()
    args.update(
        {
            "swanlab_project": cfg.logging.swanlab_project,
            "swanlab_workspace": cfg.logging.swanlab_workspace,
            "swanlab_mode": cfg.logging.swanlab_mode,
            "swanlab_logdir": cfg.logging.swanlab_logdir,
            "swanlab_experiment_name": cfg.logging.swanlab_experiment_name,
        }
    )
    return args


def _ensure_config(config):
    if isinstance(config, ExperimentConfig):
        return config
    return config_from_dict(config)
