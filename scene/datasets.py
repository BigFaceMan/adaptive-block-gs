from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from scene.image_cache import SharedMmapImageCache
from utils.camera_utils import get_camera_resolution
from utils.general_utils import PILtoTorch
from utils.graphics_utils import getProjectionMatrix, getWorld2View2


class GSCameraSample:
    def __init__(
        self,
        resolution,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image_name,
        uid,
        original_image,
        alpha_mask,
        invdepthmap,
        depth_mask,
        depth_reliable,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
    ):
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.original_image = original_image
        self.alpha_mask = alpha_mask
        self.invdepthmap = invdepthmap
        self.depth_mask = depth_mask
        self.depth_reliable = depth_reliable

        self.image_width = resolution[0]
        self.image_height = resolution[1]
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear,
            zfar=self.zfar,
            fovX=self.FoVx,
            fovY=self.FoVy,
        ).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def release_image(self):
        self.original_image = None
        self.alpha_mask = None
        self.invdepthmap = None
        self.depth_mask = None
        self.depth_reliable = False


def _read_invdepth(depth_path, is_nerf_synthetic):
    if depth_path == "":
        return None
    try:
        if is_nerf_synthetic:
            return cv2.imread(depth_path, -1).astype(np.float32) / 512
        return cv2.imread(depth_path, -1).astype(np.float32) / float(2**16)
    except FileNotFoundError:
        print(f"Error: The depth file at path '{depth_path}' was not found.")
        raise
    except IOError:
        print(f"Error: Unable to open the image file '{depth_path}'. It may be corrupted or an unsupported format.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred when trying to read depth at {depth_path}: {e}")
        raise


def load_camera_sample(args, idx, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset, image_cache=None):
    try:
        data_device = torch.device(args.data_device)
    except Exception as e:
        print(e)
        print(f"[Warning] Custom device {args.data_device} failed, fallback to default cuda device")
        data_device = torch.device("cuda")

    if image_cache is not None:
        gt_image = image_cache.image_tensor(cam_info.image_name)
        image_height = int(gt_image.shape[-2])
        image_width = int(gt_image.shape[-1])
        resolution = (image_width, image_height)
        alpha_mask = image_cache.alpha_tensor(cam_info.image_name)
        if alpha_mask is not None and args.train_test_exp and cam_info.is_test:
            alpha_mask = alpha_mask.clone()
    else:
        with Image.open(cam_info.image_path) as image:
            orig_w, orig_h = image.size
            resolution = get_camera_resolution(args, orig_w, orig_h, resolution_scale)
            resized_image_rgb = PILtoTorch(image, resolution)

        gt_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0).to(data_device)
        if resized_image_rgb.shape[0] == 4:
            alpha_mask = resized_image_rgb[3:4, ...].to(data_device)
        else:
            alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(data_device))

    if args.train_test_exp and cam_info.is_test:
        if alpha_mask is None:
            alpha_mask = torch.full(
                (1, resolution[1], resolution[0]),
                255,
                dtype=torch.uint8,
            )
        if is_test_dataset:
            alpha_mask[..., :alpha_mask.shape[-1] // 2] = 0
        else:
            alpha_mask[..., alpha_mask.shape[-1] // 2:] = 0

    invdepthmap = None
    depth_mask = None
    depth_reliable = False
    raw_invdepthmap = None
    if cam_info.depth_path:
        if image_cache is not None:
            invdepthmap = image_cache.depth_tensor(cam_info.image_name)
            depth_reliable = image_cache.depth_reliable(cam_info.image_name)
        if invdepthmap is None:
            raw_invdepthmap = _read_invdepth(cam_info.depth_path, is_nerf_synthetic)

    if raw_invdepthmap is not None:
        raw_invdepthmap = cv2.resize(raw_invdepthmap, resolution)
        raw_invdepthmap[raw_invdepthmap < 0] = 0
        depth_reliable = True

        if cam_info.depth_params is not None:
            depth_params = cam_info.depth_params
            if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                depth_reliable = False

            if depth_params["scale"] > 0:
                raw_invdepthmap = raw_invdepthmap * depth_params["scale"] + depth_params["offset"]

        if raw_invdepthmap.ndim != 2:
            raw_invdepthmap = raw_invdepthmap[..., 0]
        invdepthmap = torch.from_numpy(raw_invdepthmap[None]).to(data_device)

    return GSCameraSample(
        resolution,
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image_name=cam_info.image_name,
        uid=idx,
        original_image=gt_image,
        alpha_mask=alpha_mask,
        invdepthmap=invdepthmap,
        depth_mask=depth_mask,
        depth_reliable=depth_reliable,
    )


class GSCameraDataset(Dataset):
    def __init__(self, camera_infos, args, is_nerf_synthetic, is_test_dataset=False, scale=1.0):
        self.camera_infos = list(camera_infos)
        self.args = args
        self.is_nerf_synthetic = is_nerf_synthetic
        self.is_test_dataset = is_test_dataset
        self.scale = scale
        self.image_cache = None
        if getattr(args, "image_load_mode", "dataloader") == "shared_mmap":
            cache_dir = getattr(args, "image_mmap_cache_dir", "")
            if not cache_dir:
                raise ValueError("--image_mmap_cache_dir is required when --image_load_mode shared_mmap")
            self.image_cache = SharedMmapImageCache(cache_dir)
            self.image_cache.validate_args(args)
            print(f"[DataLoader] using shared mmap image cache: {cache_dir}")
        if torch.cuda.is_available():
            torch.empty(0, device="cuda")

    def __len__(self):
        return len(self.camera_infos)

    def __getitem__(self, idx):
        return load_camera_sample(
            self.args,
            idx,
            self.camera_infos[idx],
            self.scale,
            self.is_nerf_synthetic,
            self.is_test_dataset,
            self.image_cache,
        )


class CameraDataLoader(DataLoader):
    def __init__(self, dataset, max_cache_num=0, cache_workers=0, shuffle=True, seed=42, **kwargs):
        assert kwargs.get("batch_size", 1) == 1, "only batch_size=1 is supported"
        super().__init__(dataset=dataset, **kwargs)
        self.indices = list(range(len(self.dataset)))
        self.shuffle = shuffle
        self.seed = seed
        self.max_cache_num = int(max_cache_num)
        self.cache_workers = int(cache_workers)
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.cached = None
        self._active_cache = None

        if self.max_cache_num >= len(self.indices) and len(self.indices) > 0:
            self.max_cache_num = -1
        if self.max_cache_num < 0:
            print("cache all images")
            try:
                self.cached = self._load_data(self.indices)
            except RuntimeError as exc:
                if not self._is_cuda_oom(exc) or len(self.indices) <= 1:
                    raise
                fallback_cache_num = max(1, len(self.indices) // 2)
                print(
                    "[DataLoader] OOM while caching all cameras; "
                    f"falling back to chunked cache max_cache_num={fallback_cache_num}"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.max_cache_num = fallback_cache_num
                self.cached = None

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def _is_cuda_oom(exc):
        return "out of memory" in str(exc).lower() and "cuda" in str(exc).lower()

    def _epoch_indices(self):
        if self.shuffle:
            return torch.randperm(len(self.indices), generator=self.generator).tolist()
        return self.indices.copy()

    def _load_data(self, indices):
        if not indices:
            return []
        if self.cache_workers > 0 and len(indices) > 1:
            with ThreadPoolExecutor(max_workers=self.cache_workers) as executor:
                return list(
                    tqdm(
                        executor.map(self.dataset.__getitem__, indices),
                        total=len(indices),
                        desc=f"Caching cameras ({len(indices)})",
                    )
                )
        return [self.dataset[idx] for idx in tqdm(indices, desc=f"Loading cameras ({len(indices)})")]

    @staticmethod
    def _release_cached(cameras, empty_cuda_cache=False):
        if not cameras:
            if empty_cuda_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return
        for camera in cameras:
            release = getattr(camera, "release_image", None)
            if release is not None:
                release()
        if empty_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self):
        self._release_cached(self._active_cache, empty_cuda_cache=True)
        self._active_cache = None

    def __iter__(self):
        if self.max_cache_num < 0:
            indices = self._epoch_indices()
            for idx in indices:
                yield self.cached[idx]
            return

        indices = self._epoch_indices()
        if self.max_cache_num == 0:
            for idx in indices:
                yield self.dataset[idx]
            return

        not_cached = indices.copy()
        while not_cached:
            cache_count = min(self.max_cache_num, len(not_cached))
            while True:
                to_cache = not_cached[:cache_count]
                try:
                    cached = self._load_data(to_cache)
                    break
                except RuntimeError as exc:
                    if not self._is_cuda_oom(exc) or cache_count <= 1:
                        raise
                    self._release_cached(self._active_cache, empty_cuda_cache=True)
                    self._active_cache = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    cache_count = max(1, cache_count // 2)
                    self.max_cache_num = min(self.max_cache_num, cache_count)
                    print(
                        "[DataLoader] OOM while caching cameras; "
                        f"retrying with max_cache_num={self.max_cache_num}"
                    )
            del not_cached[:cache_count]
            self._active_cache = cached
            try:
                for camera in cached:
                    yield camera
            finally:
                self._release_cached(cached)
                if self._active_cache is cached:
                    self._active_cache = None
