import copy
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import os
import time

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
        normalmap,
        normal_mask,
        normal_reliable,
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
        self.normalmap = normalmap
        self.normal_mask = normal_mask
        self.normal_reliable = normal_reliable

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
        self.normalmap = None
        self.normal_mask = None
        self.normal_reliable = False
        self._prefetch_cpu_refs = None


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


def _normal_path_candidates(normal_path):
    if not normal_path:
        return []
    candidates = [normal_path]
    root, ext = os.path.splitext(normal_path)
    if ext.lower() != ".npy":
        candidates.append(root + ".npy")
    for image_ext in (".png", ".jpg", ".jpeg"):
        candidates.append(root + image_ext)
    deduped = []
    seen = set()
    for path in candidates:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def _normal_to_chw(array, encoded_image=False):
    normal = np.asarray(array)
    if normal.ndim != 3:
        raise ValueError(f"Expected normal map with 3 dimensions, got shape={normal.shape}")

    if normal.shape[0] == 3 and normal.shape[-1] != 3:
        normal = normal.astype(np.float32, copy=False)
    elif normal.shape[-1] >= 3:
        normal = np.transpose(normal[..., :3], (2, 0, 1)).astype(np.float32, copy=False)
    else:
        raise ValueError(f"Expected normal map with 3 channels, got shape={normal.shape}")

    finite = np.isfinite(normal)
    normal = np.where(finite, normal, 0.0).astype(np.float32, copy=False)
    if encoded_image and normal.size and normal.max() > 2.0:
        normal = normal / 255.0 * 2.0 - 1.0
    elif encoded_image and normal.size and normal.min() >= 0.0 and normal.max() <= 1.0:
        normal = normal * 2.0 - 1.0
    return normal


def _normalize_normal_chw(normal):
    norm = np.linalg.norm(normal, axis=0, keepdims=True)
    valid = np.isfinite(norm) & (norm > 1e-6)
    normal = np.divide(normal, np.maximum(norm, 1e-6), where=valid, out=np.zeros_like(normal, dtype=np.float32))
    return normal.astype(np.float32, copy=False), valid.astype(np.float32, copy=False)


def _read_normal(normal_path):
    for path in _normal_path_candidates(normal_path):
        if not os.path.exists(path):
            continue
        if path.lower().endswith(".npy"):
            return _normal_to_chw(np.load(path))
        with Image.open(path) as image:
            return _normal_to_chw(np.asarray(image.convert("RGB")), encoded_image=True)
    raise FileNotFoundError(f"Normal map not found for path '{normal_path}'")


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

    normalmap = None
    normal_mask = None
    normal_reliable = False
    raw_normalmap = None
    if image_cache is not None:
        normalmap = image_cache.normal_tensor(cam_info.image_name)
        normal_reliable = image_cache.normal_reliable(cam_info.image_name)
    normal_path = getattr(cam_info, "normal_path", "")
    if normalmap is None and normal_path:
        raw_normalmap = _read_normal(normal_path)

    if raw_normalmap is not None:
        if raw_normalmap.shape[-2:] != (resolution[1], resolution[0]):
            normal_hwc = np.transpose(raw_normalmap, (1, 2, 0))
            normal_hwc = cv2.resize(normal_hwc, resolution, interpolation=cv2.INTER_LINEAR)
            raw_normalmap = np.transpose(normal_hwc, (2, 0, 1))
        raw_normalmap, raw_normal_mask = _normalize_normal_chw(raw_normalmap)
        normalmap = torch.from_numpy(raw_normalmap).to(data_device)
        normal_mask = torch.from_numpy(raw_normal_mask).to(data_device)
        normal_reliable = True

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
        normalmap=normalmap,
        normal_mask=normal_mask,
        normal_reliable=normal_reliable,
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
            missing_count = sum(
                not self.image_cache.contains(cam_info.image_name)
                for cam_info in self.camera_infos
            )
            if missing_count:
                print(
                    "[DataLoader] "
                    f"{missing_count}/{len(self.camera_infos)} cameras are not in the shared mmap cache; "
                    "loading those images from their dataset paths"
                )
        if torch.cuda.is_available():
            torch.empty(0, device="cuda")

    def __len__(self):
        return len(self.camera_infos)

    def __getitem__(self, idx):
        cam_info = self.camera_infos[idx]
        image_cache = self.image_cache
        if image_cache is not None and not image_cache.contains(cam_info.image_name):
            image_cache = None
        return load_camera_sample(
            self.args,
            idx,
            cam_info,
            self.scale,
            self.is_nerf_synthetic,
            self.is_test_dataset,
            image_cache,
        )


class _CameraDataLoaderIterator:
    def __init__(self, loader, state=None):
        self.loader = loader
        if state:
            self.order = [int(idx) for idx in state["order"]]
            self.cursor = int(state.get("cursor", 0))
            generator_state = state.get("generator_state")
            if generator_state is not None:
                self.loader.generator.set_state(generator_state.cpu())
        else:
            self.order = self.loader._epoch_indices()
            self.cursor = 0
        self._chunk = None
        self._chunk_start = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cursor >= len(self.order):
            self.close()
            raise StopIteration

        if self.loader.max_cache_num < 0:
            dataset_index = self.order[self.cursor]
            sample = self.loader.cached[dataset_index]
        elif self.loader.max_cache_num == 0:
            dataset_index = self.order[self.cursor]
            sample = self.loader.dataset[dataset_index]
        else:
            if self._chunk is None or not (
                self._chunk_start <= self.cursor < self._chunk_start + len(self._chunk)
            ):
                self._load_chunk()
            dataset_index = self.order[self.cursor]
            sample = self._chunk[self.cursor - self._chunk_start]

        self.cursor += 1
        sample.dataset_index = dataset_index
        return sample

    def _load_chunk(self):
        self.loader._release_cached(self._chunk)
        self._chunk = None
        cache_count = min(self.loader.max_cache_num, len(self.order) - self.cursor)
        while True:
            indices = self.order[self.cursor:self.cursor + cache_count]
            try:
                self._chunk = self.loader._load_data(indices)
                self._chunk_start = self.cursor
                return
            except RuntimeError as exc:
                if not self.loader._is_cuda_oom(exc) or cache_count <= 1:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cache_count = max(1, cache_count // 2)
                self.loader.max_cache_num = min(self.loader.max_cache_num, cache_count)
                print(
                    "[DataLoader] OOM while caching cameras; "
                    f"retrying with max_cache_num={self.loader.max_cache_num}"
                )

    def state_dict(self, cursor=None):
        return {
            "order": list(self.order),
            "cursor": self.cursor if cursor is None else int(cursor),
            "generator_state": self.loader.generator.get_state(),
        }

    def close(self):
        self.loader._release_cached(self._chunk)
        self._chunk = None


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
        self._resume_state = None

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

    def state_dict(self):
        return {
            "order": [],
            "cursor": 0,
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state):
        self._resume_state = state

    def close(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __iter__(self):
        state = self._resume_state
        self._resume_state = None
        return _CameraDataLoaderIterator(self, state=state)


_IMAGE_TENSOR_ATTRIBUTES = (
    "original_image",
    "alpha_mask",
    "invdepthmap",
    "depth_mask",
    "normalmap",
    "normal_mask",
)


def _pin_camera_sample(viewpoint_cam):
    pinned = copy.copy(viewpoint_cam)
    for attr in _IMAGE_TENSOR_ATTRIBUTES:
        value = getattr(viewpoint_cam, attr, None)
        if isinstance(value, torch.Tensor) and not value.is_cuda and not value.is_pinned():
            value = value.pin_memory()
        setattr(pinned, attr, value)
    return pinned


def _cuda_image_tensor(value, normalize_uint8=False):
    if value is None or not isinstance(value, torch.Tensor):
        return value
    if normalize_uint8 and value.dtype == torch.uint8:
        return value.to(device="cuda", dtype=torch.float32, non_blocking=True).div_(255.0)
    return value.to(device="cuda", non_blocking=True)


def _camera_sample_to_cuda(viewpoint_cam, stream):
    cuda_sample = copy.copy(viewpoint_cam)
    cpu_refs = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        for attr in _IMAGE_TENSOR_ATTRIBUTES:
            value = getattr(viewpoint_cam, attr, None)
            if isinstance(value, torch.Tensor) and not value.is_cuda:
                cpu_refs.append(value)
            cuda_value = _cuda_image_tensor(
                value,
                normalize_uint8=attr in {"original_image", "alpha_mask"},
            )
            setattr(cuda_sample, attr, cuda_value)
        end.record(stream)
    cuda_sample._prefetch_cpu_refs = cpu_refs
    return cuda_sample, start, end, cpu_refs


class CUDACameraPrefetcher:
    """One-camera-ahead pinned-memory/H2D pipeline with a bounded GPU queue."""

    def __init__(self, camera_iterator, enabled=True, prepare_cpu=None):
        self.camera_iterator = camera_iterator
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self.prepare_cpu = prepare_cpu
        self.logical_cursor = int(getattr(camera_iterator, "cursor", 0))
        self.last_data_wait_seconds = 0.0
        self._ready = deque()
        self._inflight_sources = deque()
        self._first_yield = True
        self._exhausted = False
        self._future = None
        self._executor = None
        self._stream = torch.cuda.Stream() if self.enabled else None

        loader = getattr(camera_iterator, "loader", None)
        if self.enabled and loader is not None and loader.cached is not None:
            self._executor = ThreadPoolExecutor(max_workers=1)

        if self.enabled:
            self._submit_cpu()
            self._append_ready()
            self._append_ready()

    def __iter__(self):
        return self

    def __next__(self):
        wait_start = time.perf_counter()
        if not self.enabled:
            sample = next(self.camera_iterator)
            if self.prepare_cpu is not None:
                sample = self.prepare_cpu(sample)
            self.logical_cursor += 1
            self.last_data_wait_seconds = time.perf_counter() - wait_start
            sample.prefetch_stats = {
                "load_cpu_seconds": self.last_data_wait_seconds,
                "prepare_cpu_seconds": float(getattr(sample, "depth_mask_prepare_time", 0.0)),
                "pin_cpu_seconds": 0.0,
            }
            sample.prefetch_h2d_events = (None, None)
            return sample

        if not self._first_yield:
            self._append_ready()
        self._first_yield = False
        if not self._ready:
            self.last_data_wait_seconds = time.perf_counter() - wait_start
            raise StopIteration

        sample, start, end, stats = self._ready.popleft()
        current_stream = torch.cuda.current_stream()
        current_stream.wait_event(end)
        for attr in _IMAGE_TENSOR_ATTRIBUTES:
            value = getattr(sample, attr, None)
            if isinstance(value, torch.Tensor) and value.is_cuda:
                value.record_stream(current_stream)

        self.logical_cursor += 1
        self.last_data_wait_seconds = time.perf_counter() - wait_start
        sample.prefetch_stats = stats
        sample.prefetch_h2d_events = (start, end)
        self._release_completed_sources()
        return sample

    def _prepare_next(self):
        load_start = time.perf_counter()
        sample = next(self.camera_iterator)
        load_seconds = time.perf_counter() - load_start

        prepare_start = time.perf_counter()
        if self.prepare_cpu is not None:
            sample = self.prepare_cpu(sample)
        prepare_seconds = time.perf_counter() - prepare_start

        pin_start = time.perf_counter()
        sample = _pin_camera_sample(sample)
        pin_seconds = time.perf_counter() - pin_start
        return sample, {
            "load_cpu_seconds": load_seconds,
            "prepare_cpu_seconds": prepare_seconds,
            "pin_cpu_seconds": pin_seconds,
        }

    def _submit_cpu(self):
        if self._exhausted or self._future is not None:
            return
        if self._executor is None:
            return
        self._future = self._executor.submit(self._prepare_next)

    def _take_prepared(self):
        if self._exhausted:
            raise StopIteration
        try:
            if self._executor is None:
                return self._prepare_next()
            if self._future is None:
                self._submit_cpu()
            result = self._future.result()
            self._future = None
            self._submit_cpu()
            return result
        except StopIteration:
            self._future = None
            self._exhausted = True
            raise

    def _append_ready(self):
        if self._exhausted:
            return
        try:
            sample, stats = self._take_prepared()
        except StopIteration:
            return
        cuda_sample, start, end, cpu_refs = _camera_sample_to_cuda(sample, self._stream)
        self._ready.append((cuda_sample, start, end, stats))
        self._inflight_sources.append((end, cpu_refs))
        self._release_completed_sources(force_if_full=True)

    def _release_completed_sources(self, force_if_full=False):
        while self._inflight_sources and self._inflight_sources[0][0].query():
            self._inflight_sources.popleft()
        if force_if_full and len(self._inflight_sources) > 4:
            event, _ = self._inflight_sources.popleft()
            event.synchronize()

    def state_dict(self):
        return self.camera_iterator.state_dict(cursor=self.logical_cursor)

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        for sample, _, _, _ in self._ready:
            sample.release_image()
        self._ready.clear()
        for event, _ in self._inflight_sources:
            event.synchronize()
        self._inflight_sources.clear()
        close = getattr(self.camera_iterator, "close", None)
        if close is not None:
            close()
