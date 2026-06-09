import json
import os
from pathlib import Path

import numpy as np
import torch


def image_name_aliases(name):
    path = Path(name)
    aliases = {str(name), path.name, path.stem}
    if path.suffix:
        aliases.add(str(path.with_suffix("")))
    return aliases


class SharedMmapImageCache:
    def __init__(self, cache_dir):
        self.cache_dir = os.path.abspath(cache_dir)
        manifest_path = os.path.join(self.cache_dir, "manifest.json")
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        image_file = self.manifest.get("image_file", "images.uint8.bin")
        self.images = np.memmap(os.path.join(self.cache_dir, image_file), dtype=np.uint8, mode="c")

        alpha_file = self.manifest.get("alpha_file")
        self.alphas = None
        if alpha_file:
            alpha_path = os.path.join(self.cache_dir, alpha_file)
            if os.path.exists(alpha_path):
                self.alphas = np.memmap(alpha_path, dtype=np.uint8, mode="c")

        self.items = self.manifest.get("items", {})
        self._aliases = {}
        for image_name in self.items:
            for alias in image_name_aliases(image_name):
                self._aliases.setdefault(alias, image_name)

    def validate_args(self, args):
        expected_resolution = self.manifest.get("resolution")
        if expected_resolution is not None and int(expected_resolution) != int(args.resolution):
            raise ValueError(
                "Shared image mmap cache resolution mismatch: "
                f"cache={expected_resolution}, args={args.resolution}"
            )

    def _item(self, image_name):
        key = image_name if image_name in self.items else self._aliases.get(image_name)
        if key is None:
            raise KeyError(f"Image '{image_name}' was not found in shared mmap cache")
        return self.items[key]

    @staticmethod
    def _tensor_from_memmap(memmap, offset, shape):
        offset = int(offset)
        shape = tuple(int(dim) for dim in shape)
        size = int(np.prod(shape))
        array = memmap[offset:offset + size].reshape(shape)
        return torch.from_numpy(array)

    def image_tensor(self, image_name):
        item = self._item(image_name)
        return self._tensor_from_memmap(self.images, item["image_offset"], item["image_shape"])

    def alpha_tensor(self, image_name):
        if self.alphas is None:
            return None
        item = self._item(image_name)
        offset = item.get("alpha_offset")
        if offset is None:
            return None
        return self._tensor_from_memmap(self.alphas, offset, item["alpha_shape"])
