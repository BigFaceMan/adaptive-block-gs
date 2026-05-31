import os

import cv2
import numpy as np
import torch

from utils.graphics_utils import fov2focal
from utils.partition_utils import points_in_partition_bbox, read_gaussian_or_point_ply


class BlockDepthMasker:
    def __init__(self, block_metadata, bbox_mode="expanded", max_points=100000, dilate_px=16):
        if bbox_mode not in {"core", "expanded"}:
            raise ValueError("depth_reg_mask_bbox_mode must be 'core' or 'expanded'")

        coarse_model = block_metadata.get("_partition_coarse_model", "")
        if not coarse_model:
            raise ValueError("Block depth mask requires a partition tree with coarse_model")
        if not os.path.isfile(coarse_model):
            raise FileNotFoundError(f"Coarse model not found for block depth mask: {coarse_model}")

        bbox_key = f"{bbox_mode}_bbox"
        coarse = read_gaussian_or_point_ply(coarse_model)
        mask = points_in_partition_bbox(coarse["xyz"], block_metadata, bbox_key)
        points = np.asarray(coarse["xyz"][mask], dtype=np.float32)

        max_points = int(max_points)
        if max_points > 0 and points.shape[0] > max_points:
            rng = np.random.RandomState(0)
            indices = rng.choice(points.shape[0], size=max_points, replace=False)
            points = points[indices]

        self.points = points
        self.dilate_px = max(0, int(dilate_px))
        self.block_id = block_metadata.get("id", "<unknown>")
        self.bbox_mode = bbox_mode
        print(
            "[DepthMask] "
            f"block={self.block_id} bbox={bbox_mode} points={self.points.shape[0]} "
            f"dilate_px={self.dilate_px}"
        )

    def mask_for(self, viewpoint_cam):
        height = int(viewpoint_cam.image_height)
        width = int(viewpoint_cam.image_width)
        mask = np.zeros((height, width), dtype=np.uint8)
        if self.points.shape[0] == 0:
            return torch.from_numpy(mask[None].astype(np.float32))

        camera_points = self.points @ viewpoint_cam.R + viewpoint_cam.T
        z = camera_points[:, 2]
        valid = z > 1e-6
        if not np.any(valid):
            return torch.from_numpy(mask[None].astype(np.float32))

        camera_points = camera_points[valid]
        z = z[valid]
        fx = fov2focal(viewpoint_cam.FoVx, width)
        fy = fov2focal(viewpoint_cam.FoVy, height)
        u = fx * (camera_points[:, 0] / z) + width * 0.5
        v = fy * (camera_points[:, 1] / z) + height * 0.5
        in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not np.any(in_image):
            return torch.from_numpy(mask[None].astype(np.float32))

        u = u[in_image].astype(np.int32)
        v = v[in_image].astype(np.int32)
        mask[v, u] = 1

        if self.dilate_px > 0:
            kernel_size = self.dilate_px * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask = cv2.dilate(mask, kernel, iterations=1)

        return torch.from_numpy(mask[None].astype(np.float32))
