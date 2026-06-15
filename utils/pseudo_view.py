import math

import torch
import torch.nn.functional as F

from utils.graphics_utils import getProjectionMatrix


class PseudoViewCamera:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = torch.inverse(world_view_transform)[3, :3]


def _as_rt(camera, device, dtype):
    R = torch.as_tensor(camera.R, device=device, dtype=dtype)
    T = torch.as_tensor(camera.T, device=device, dtype=dtype)
    return R, T


def _single_channel(tensor):
    while tensor.ndim > 3:
        tensor = tensor[0]
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 3 and tensor.shape[0] != 1:
        tensor = tensor[:1]
    return tensor


def _camera_focal(camera):
    width = int(camera.image_width)
    height = int(camera.image_height)
    fx = width / (2.0 * math.tan(float(camera.FoVx) * 0.5))
    fy = height / (2.0 * math.tan(float(camera.FoVy) * 0.5))
    return fx, fy


def _project_ref_to_pseudo(ref_camera, pseudo_camera, ref_invdepth, detach_ref_depth=True):
    ref_invdepth = _single_channel(ref_invdepth)
    if detach_ref_depth:
        ref_invdepth = ref_invdepth.detach()

    device = ref_invdepth.device
    dtype = ref_invdepth.dtype
    _, height, width = ref_invdepth.shape
    fx, fy = _camera_focal(ref_camera)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    ref_inv = ref_invdepth.squeeze(0)
    ref_valid = torch.isfinite(ref_inv) & (ref_inv > 1e-8)
    safe_ref_inv = torch.where(ref_valid, ref_inv, torch.ones_like(ref_inv))
    ref_depth = 1.0 / safe_ref_inv.clamp_min(1e-8)

    x_cam = (xs - cx) * ref_depth / fx
    y_cam = (ys - cy) * ref_depth / fy
    ref_points = torch.stack((x_cam, y_cam, ref_depth), dim=-1).view(-1, 3)

    ref_R, ref_T = _as_rt(ref_camera, device, dtype)
    pseudo_R, pseudo_T = _as_rt(pseudo_camera, device, dtype)
    world_points = (ref_points - ref_T) @ ref_R.transpose(0, 1)
    pseudo_points = world_points @ pseudo_R + pseudo_T

    pseudo_z = pseudo_points[:, 2].view(height, width)
    pseudo_x = pseudo_points[:, 0].view(height, width)
    pseudo_y = pseudo_points[:, 1].view(height, width)
    u = fx * (pseudo_x / pseudo_z.clamp_min(1e-8)) + cx
    v = fy * (pseudo_y / pseudo_z.clamp_min(1e-8)) + cy
    inside = (pseudo_z > 1e-8) & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)

    grid = torch.stack(
        (
            2.0 * u / max(width - 1, 1) - 1.0,
            2.0 * v / max(height - 1, 1) - 1.0,
        ),
        dim=-1,
    ).unsqueeze(0)
    return grid, ref_valid & inside, pseudo_z


def make_pseudo_view_camera(
    ref_camera,
    ref_invdepth,
    shift_ratio=0.03,
    random_sign=True,
    detach_depth=True,
):
    ref_invdepth = _single_channel(ref_invdepth)
    device = ref_invdepth.device
    dtype = ref_invdepth.dtype
    inv_for_shift = ref_invdepth.detach() if detach_depth else ref_invdepth
    valid = torch.isfinite(inv_for_shift) & (inv_for_shift > 1e-8)
    if not bool(valid.any().item()):
        return None, {"shift": 0.0, "valid_depth": False}

    depth = 1.0 / inv_for_shift[valid].clamp_min(1e-8)
    median_depth = torch.median(depth)
    if not torch.isfinite(median_depth):
        return None, {"shift": 0.0, "valid_depth": False}

    width = int(ref_camera.image_width)
    height = int(ref_camera.image_height)
    fx, _ = _camera_focal(ref_camera)
    shift = float(shift_ratio) * float(width) * median_depth / max(float(fx), 1e-6)
    if random_sign and bool((torch.rand((), device=device) < 0.5).item()):
        shift = -shift

    R, T = _as_rt(ref_camera, device, dtype)
    center = torch.as_tensor(ref_camera.camera_center, device=device, dtype=dtype)
    right_axis = F.normalize(R[:, 0], dim=0, eps=1e-8)
    pseudo_center = center + shift * right_axis
    pseudo_T = -(pseudo_center @ R)

    world_view = torch.eye(4, device=device, dtype=dtype)
    world_view[:3, :3] = R
    world_view[3, :3] = pseudo_T

    projection = getattr(ref_camera, "projection_matrix", None)
    if projection is None:
        projection = getProjectionMatrix(
            znear=ref_camera.znear,
            zfar=ref_camera.zfar,
            fovX=ref_camera.FoVx,
            fovY=ref_camera.FoVy,
        ).transpose(0, 1)
    projection = projection.to(device=device, dtype=dtype)
    full_proj = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)

    pseudo_camera = PseudoViewCamera(
        width,
        height,
        ref_camera.FoVy,
        ref_camera.FoVx,
        ref_camera.znear,
        ref_camera.zfar,
        world_view,
        full_proj,
    )
    pseudo_camera.R = R
    pseudo_camera.T = pseudo_T
    pseudo_camera.image_name = getattr(ref_camera, "image_name", "")
    return pseudo_camera, {"shift": float(torch.abs(shift).detach().item()), "valid_depth": True}


def pseudo_view_reprojection_loss(
    ref_camera,
    pseudo_camera,
    ref_invdepth,
    pseudo_invdepth,
    pseudo_image,
    gt_image,
    mask_mode="valid",
    depth_rel_thresh=0.05,
    min_pixels=2048,
    detach_ref_depth=True,
):
    pseudo_invdepth = _single_channel(pseudo_invdepth)
    pseudo_invdepth_for_mask = pseudo_invdepth.detach()

    grid, mask, pseudo_z = _project_ref_to_pseudo(
        ref_camera,
        pseudo_camera,
        ref_invdepth,
        detach_ref_depth=detach_ref_depth,
    )
    warped_image = F.grid_sample(
        pseudo_image.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)
    warped_invdepth = F.grid_sample(
        pseudo_invdepth_for_mask.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)

    if mask_mode == "depth_consistent":
        warped_depth = 1.0 / warped_invdepth.squeeze(0).clamp_min(1e-8)
        rel_error = torch.abs(warped_depth - pseudo_z) / pseudo_z.clamp_min(1e-8)
        mask = mask & (warped_invdepth.squeeze(0) > 1e-8) & (rel_error < float(depth_rel_thresh))
    elif mask_mode != "valid":
        raise ValueError("pseudo_view_mask_mode must be 'valid' or 'depth_consistent'")

    mask = mask.unsqueeze(0).to(dtype=gt_image.dtype)
    mask_pixels = float(mask.sum().detach().item())
    stats = {
        "mask_pixels": mask_pixels,
        "mask_coverage": mask_pixels / max(float(mask.numel()), 1.0),
        "mask_enough": mask_pixels >= float(min_pixels),
    }
    if not stats["mask_enough"]:
        return None, stats

    image_mask = mask.expand_as(warped_image)
    loss = (torch.abs(warped_image - gt_image) * image_mask).sum() / image_mask.sum().clamp_min(1.0)
    return loss, stats


def warp_pseudo_normal_to_ref(
    ref_camera,
    pseudo_camera,
    ref_invdepth,
    pseudo_normal,
    pseudo_alpha,
    normal_valid,
    ref_valid_mask=None,
    alpha_min=0.01,
    min_pixels=2048,
    detach_ref_depth=True,
):
    grid, mask, _ = _project_ref_to_pseudo(
        ref_camera,
        pseudo_camera,
        ref_invdepth,
        detach_ref_depth=detach_ref_depth,
    )
    warped_normal = F.grid_sample(
        pseudo_normal.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)
    warped_alpha = F.grid_sample(
        _single_channel(pseudo_alpha.detach()).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)

    if ref_valid_mask is not None:
        ref_valid_mask = _single_channel(ref_valid_mask).squeeze(0).to(device=mask.device)
        mask = mask & ref_valid_mask.bool()
    if normal_valid is not None:
        normal_valid = _single_channel(normal_valid).squeeze(0).to(device=mask.device)
        mask = mask & normal_valid.bool()
    normal_norm = torch.linalg.norm(warped_normal, dim=0, keepdim=True)
    mask = mask & (warped_alpha.squeeze(0) > float(alpha_min))
    mask = mask & torch.isfinite(warped_normal).all(dim=0) & (normal_norm.squeeze(0) > 1e-6)

    mask = mask.unsqueeze(0).to(dtype=warped_normal.dtype)
    mask_pixels = float(mask.sum().detach().item())
    stats = {
        "mask_pixels": mask_pixels,
        "mask_coverage": mask_pixels / max(float(mask.numel()), 1.0),
        "mask_enough": mask_pixels >= float(min_pixels),
    }
    if not stats["mask_enough"]:
        return None, mask, stats

    warped_normal = F.normalize(warped_normal, p=2, dim=0, eps=1e-6)
    return warped_normal, mask, stats
