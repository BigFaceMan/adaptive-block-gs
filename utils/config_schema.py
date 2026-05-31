import copy
import os
import re


try:
    import yaml
except ImportError:
    yaml = None


_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


class GroupParams:
    pass


class ExperimentConfig(GroupParams):
    @property
    def output_root(self):
        return self.experiment.output_root or os.path.join("output", self.experiment.name)

    @property
    def partition_output_path(self):
        return self.partition.output_path or os.path.join(self.output_root, "partitions")

    @property
    def partition_tree_path(self):
        return self.block_training.partition_path or self.merge.partition_path or os.path.join(
            self.partition_output_path,
            "partition_tree.json",
        )

    @property
    def merge_output_path(self):
        return self.merge.output_path or os.path.join(self.output_root, "merged")

    def block_model_path(self, block_id):
        blocks_root = self.block_training.blocks_root or os.path.join(self.output_root, "blocks")
        return os.path.join(blocks_root, block_id)


def make_group(**kwargs):
    group = GroupParams()
    for key, value in kwargs.items():
        setattr(group, key, value)
    return group


def get_default_config():
    cfg = ExperimentConfig()
    cfg.experiment = make_group(name="experiment", output_root="")
    cfg.dataset = get_default_dataset_params()
    cfg.model = get_default_model_params()
    cfg.pipeline = get_default_pipeline_params()
    cfg.logging = get_default_logging_params()
    cfg.training = get_default_training_params()
    cfg.optimization = get_default_optimization_params()
    cfg.partition = get_default_partition_params()
    cfg.camera_assignment = get_default_camera_assignment_params()
    cfg.visualization = get_default_visualization_params()
    cfg.block_training = get_default_block_training_params()
    cfg.merge = get_default_merge_params()
    cfg.render = get_default_render_params()
    cfg.metrics = get_default_metrics_params()
    cfg.config_path = ""
    return cfg


def get_default_dataset_params():
    return make_group(
        source_path="",
        images="images",
        depths="",
        test_source_path="",
        test_images="",
        test_depths="",
        resolution=-1,
        white_background=False,
        train_test_exp=False,
        data_device="cpu",
        image_loader_seed=42,
        max_cache_num=128,
        image_cache_workers=0,
        eval=False,
    )


def get_default_model_params():
    return make_group(sh_degree=3, coarse_model="")


def get_default_pipeline_params():
    return make_group(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )


def get_default_logging_params():
    return make_group(
        swanlab_project="block-gs",
        swanlab_workspace="",
        swanlab_mode="cloud",
        swanlab_logdir="",
        swanlab_experiment_prefix="experiment",
        swanlab_experiment_name="",
    )


def get_default_training_params():
    return make_group(
        model_path="",
        ip="127.0.0.1",
        port=6009,
        debug_from=-1,
        detect_anomaly=False,
        quiet=False,
        disable_viewer=False,
    )


def get_default_optimization_params():
    return make_group(
        iterations=30_000,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30_000,
        feature_lr=0.0025,
        opacity_lr=0.025,
        scaling_lr=0.005,
        rotation_lr=0.001,
        exposure_lr_init=0.01,
        exposure_lr_final=0.001,
        exposure_lr_delay_steps=0,
        exposure_lr_delay_mult=0.0,
        percent_dense=0.01,
        lambda_dssim=0.2,
        densification_interval=100,
        opacity_reset_interval=3000,
        densify_from_iter=500,
        densify_until_iter=15_000,
        densify_grad_threshold=0.0002,
        depth_l1_weight_init=1.0,
        depth_l1_weight_final=0.01,
        depth_reg_mask_mode="full",
        depth_reg_mask_bbox_mode="expanded",
        depth_reg_mask_dilate_px=16,
        depth_reg_mask_min_pixels=2048,
        depth_reg_mask_max_points=100000,
        depth_reg_mask_cache=True,
        depth_reg_mask_cache_max_items=0,
        random_background=False,
        optimizer_type="default",
    )


def get_default_partition_params():
    return make_group(
        output_path="",
        coord_space="world",
        contract_aabb=None,
        axes=["x", "y"],
        max_depth=5,
        max_blocks=16,
        max_block_importance=0.0,
        max_block_density=0.0,
        min_points=50_000,
        min_size=0.0,
        expand_ratio=0.0,
        num_split_candidates=64,
        lambda_boundary=0.2,
        importance="opacity_scale",
    )


def get_default_camera_assignment_params():
    return make_group(
        tau_projection=0.02,
        tau_test_projection=-1.0,
        min_cameras=20,
        min_test_cameras=0,
        supplement_cameras=False,
        projection_max_points=5000,
        render_difference=get_default_render_difference_params(),
    )


def get_default_render_difference_params():
    return make_group(
        enabled=False,
        threshold=0.03,
        max_candidates_per_block=0,
        max_width=0,
        cache_full=False,
    )


def get_default_visualization_params():
    return make_group(
        visualize_blocks=False,
        visualize_output="",
        visualize_bbox_mode="core",
        visualize_max_cameras_per_block=3,
        visualize_max_points_per_block=50_000,
        visualize_point_radius=1,
        visualize_max_image_width=1600,
        visualize_random_seed=0,
        visualize_topdown=False,
        visualize_topdown_output="",
        visualize_topdown_max_points=0,
        visualize_topdown_image_size=4096,
        visualize_topdown_point_radius=0,
        visualize_topdown_bbox_mode="both",
        visualize_topdown_clip_percentile=0.2,
        visualize_topdown_color="auto",
        visualize_topdown_keep="max",
    )


def get_default_block_training_params():
    return make_group(
        blocks_root="",
        partition_path="",
        block_id="",
        partition_bbox_mode="core",
        partition_init_mode="cropped",
        partition_load_test_cameras=False,
        test_iterations=[7000, 30000],
        save_iterations=[7000, 30000],
        checkpoint_iterations=[],
        start_checkpoint=None,
    )


def get_default_merge_params():
    return make_group(
        partition_path="",
        blocks_root="",
        iteration=None,
        output_path="",
        allow_missing=False,
        cfg_args_source="",
    )


def get_default_render_params():
    return make_group(
        model_path="",
        source_path="",
        images="",
        depths="",
        iteration=-1,
        skip_train=False,
        skip_test=False,
        render_depth=False,
        quiet=False,
    )


def get_default_metrics_params():
    return make_group(model_path="", model_paths=[])


def load_config(path, overrides=None):
    if not path:
        raise ValueError("--config is required for YAML config loading")
    if yaml is None:
        raise ImportError("PyYAML is required for --config support. Install pyyaml in the active environment.")
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")

    data = copy.deepcopy(data)
    data.setdefault("experiment", {})
    data["experiment"].setdefault("name", os.path.splitext(os.path.basename(path))[0])
    data["_config_path"] = os.path.abspath(path)
    data = apply_overrides(data, overrides)
    data = resolve_config_refs(data)
    return config_from_dict(data)


def config_from_dict(data):
    cfg = get_default_config()
    payload = copy.deepcopy(data)
    cfg.config_path = payload.pop("_config_path", "")
    for section, value in payload.items():
        if hasattr(cfg, section):
            current = getattr(cfg, section)
            if isinstance(current, GroupParams) and isinstance(value, dict):
                extract_args(current, value)
            else:
                setattr(cfg, section, _dict_to_group(value) if isinstance(value, dict) else value)
        else:
            setattr(cfg, section, _dict_to_group(value) if isinstance(value, dict) else value)
    return cfg


def extract_args(group, values):
    if values is None:
        return group
    if not isinstance(values, dict):
        raise ValueError(f"Expected mapping for config group, got {type(values).__name__}")
    for key, value in values.items():
        current = getattr(group, key, None)
        if isinstance(current, GroupParams) and isinstance(value, dict):
            extract_args(current, value)
        else:
            setattr(group, key, _dict_to_group(value) if isinstance(value, dict) else value)
    return group


def config_to_dict(config):
    if isinstance(config, GroupParams):
        payload = {}
        for key, value in vars(config).items():
            out_key = "_config_path" if key == "config_path" else key
            payload[out_key] = config_to_dict(value)
        return payload
    if isinstance(config, dict):
        return {key: config_to_dict(value) for key, value in config.items()}
    if isinstance(config, (list, tuple)):
        return [config_to_dict(value) for value in config]
    return copy.deepcopy(config)


def apply_overrides(config, overrides=None):
    config = copy.deepcopy(config)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}', expected key=value")
        key, raw_value = override.split("=", 1)
        _set_dotted(config, key, _parse_scalar(raw_value))
    return config


def resolve_config_refs(config):
    resolved = copy.deepcopy(config)
    for _ in range(8):
        previous = copy.deepcopy(resolved)
        resolved = _resolve_node(resolved, resolved)
        if resolved == previous:
            break
    return resolved


def _dict_to_group(value):
    group = GroupParams()
    extract_args(group, value)
    return group


def _set_dotted(config, dotted_key, value):
    cursor = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        next_value = cursor.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set override '{dotted_key}' through non-dict value")
        cursor = next_value
    cursor[parts[-1]] = value


def _parse_scalar(value):
    if yaml is not None:
        try:
            return yaml.safe_load(value)
        except Exception:
            pass
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _resolve_node(node, root):
    if isinstance(node, dict):
        return {key: _resolve_node(value, root) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_node(value, root) for value in node]
    if isinstance(node, str):
        return _VAR_PATTERN.sub(lambda match: _stringify_ref(_lookup(root, match.group(1))), node)
    return node


def _stringify_ref(value):
    return "" if value is None else str(value)


def _lookup(config, dotted_key):
    value = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value
