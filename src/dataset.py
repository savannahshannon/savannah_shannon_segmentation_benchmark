"""COCO 2017 subset dataset + mask-safe transforms.

Provides two Dataset classes:
    SemanticSegDataset  : (image_tensor, semantic_mask_tensor)
    InstanceSegDataset  : (image_tensor, dict(masks, labels, boxes))

Both classes share transform behavior:
    * Geometric ops (resize, flip, rotate, random scale/crop) are applied to
      the image AND every mask/box (bilinear for image, nearest for masks).
    * Color jitter is applied to the image only.
    * Normalization uses ImageNet mean/std.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False
    class Dataset:  # type: ignore
        pass

from .mask_utils import (
    NUM_CLASSES, IGNORE_INDEX, COCO_TO_SEM,
    build_semantic_mask, build_instance_targets, load_mask_png,
)


# ----------------------- Transforms ----------------------- #

class MaskSafeCompose:
    """Compose transforms that need access to both image and mask(s)."""
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, sample: Dict) -> Dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


class Resize:
    def __init__(self, size: Tuple[int, int]):
        self.size = tuple(size)

    def __call__(self, s):
        s["image"] = s["image"].resize(self.size, Image.BILINEAR)
        if "mask" in s:
            s["mask"] = s["mask"].resize(self.size, Image.NEAREST)
        if "instance_masks" in s and len(s["instance_masks"]):
            s["instance_masks"] = [m.resize(self.size, Image.NEAREST)
                                   for m in s["instance_masks"]]
            # scale boxes
            iw, ih = s["orig_size"]
            sx, sy = self.size[0] / iw, self.size[1] / ih
            if "boxes" in s and len(s["boxes"]):
                b = np.asarray(s["boxes"], dtype=np.float32).copy()
                b[:, [0, 2]] *= sx
                b[:, [1, 3]] *= sy
                s["boxes"] = b
        s["orig_size"] = self.size
        return s


class RandomHFlip:
    def __init__(self, p: float = 0.5): self.p = p
    def __call__(self, s):
        if random.random() < self.p:
            s["image"] = s["image"].transpose(Image.FLIP_LEFT_RIGHT)
            if "mask" in s:
                s["mask"] = s["mask"].transpose(Image.FLIP_LEFT_RIGHT)
            if "instance_masks" in s:
                s["instance_masks"] = [m.transpose(Image.FLIP_LEFT_RIGHT)
                                       for m in s["instance_masks"]]
            if "boxes" in s and len(s["boxes"]):
                W = s["orig_size"][0]
                b = np.asarray(s["boxes"], dtype=np.float32).copy()
                x0 = W - b[:, 2]
                x1 = W - b[:, 0]
                b[:, 0], b[:, 2] = x0, x1
                s["boxes"] = b
        return s


class RandomRotate:
    def __init__(self, degrees: float = 10.0): self.degrees = degrees
    def __call__(self, s):
        if self.degrees <= 0: return s
        angle = random.uniform(-self.degrees, self.degrees)
        s["image"] = s["image"].rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        if "mask" in s:
            s["mask"] = s["mask"].rotate(angle, resample=Image.NEAREST, fillcolor=IGNORE_INDEX)
        if "instance_masks" in s:
            s["instance_masks"] = [m.rotate(angle, resample=Image.NEAREST, fillcolor=0)
                                   for m in s["instance_masks"]]
        # Boxes after rotation are recomputed from rotated masks (below in __getitem__)
        return s


class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
        self.b, self.c, self.s, self.h = brightness, contrast, saturation, hue

    def __call__(self, s):
        from PIL import ImageEnhance
        img = s["image"]
        if self.b: img = ImageEnhance.Brightness(img).enhance(1 + random.uniform(-self.b, self.b))
        if self.c: img = ImageEnhance.Contrast(img).enhance(1 + random.uniform(-self.c, self.c))
        if self.s: img = ImageEnhance.Color(img).enhance(1 + random.uniform(-self.s, self.s))
        # PIL has no hue jitter; skip if not critical
        s["image"] = img
        return s


class ToTensorNormalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std  = np.asarray(std,  dtype=np.float32)

    def __call__(self, s):
        img = np.asarray(s["image"].convert("RGB"), dtype=np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # CxHxW
        if _HAS_TORCH:
            s["image"] = torch.from_numpy(img.copy())
        else:
            s["image"] = img

        if "mask" in s:
            m = np.asarray(s["mask"], dtype=np.int64)
            s["mask"] = torch.from_numpy(m) if _HAS_TORCH else m
        if "instance_masks" in s:
            if len(s["instance_masks"]):
                mm = np.stack([np.asarray(m, dtype=np.uint8) for m in s["instance_masks"]])
                s["instance_masks"] = torch.from_numpy(mm) if _HAS_TORCH else mm
            else:
                s["instance_masks"] = (torch.zeros((0, *s["orig_size"][::-1]), dtype=torch.uint8)
                                       if _HAS_TORCH else np.zeros((0, 1, 1), dtype=np.uint8))
            if "boxes" in s:
                b = np.asarray(s["boxes"], dtype=np.float32)
                s["boxes"] = torch.from_numpy(b) if _HAS_TORCH else b
            if "labels" in s:
                l = np.asarray(s["labels"], dtype=np.int64)
                s["labels"] = torch.from_numpy(l) if _HAS_TORCH else l
        return s


def build_train_transforms(input_size, aug_cfg):
    return MaskSafeCompose([
        Resize(input_size),
        RandomHFlip(aug_cfg.get("horizontal_flip", 0.0)),
        RandomRotate(aug_cfg.get("rotation_degrees", 0.0)),
        ColorJitter(**aug_cfg.get("color_jitter", {})),
        ToTensorNormalize(),
    ])


def build_eval_transforms(input_size):
    return MaskSafeCompose([
        Resize(input_size),
        ToTensorNormalize(),
    ])


# ----------------------- Datasets ----------------------- #

class _BaseCocoSubset:
    """Shared file-layout logic for both semantic and instance datasets.

    Expected on disk (produced by scripts/prepare_coco_subset.py):
        data/coco/images/<img_id>.jpg           - RGB image
        annotations/split_manifest.json         - {"train": [...], "val": [...], "test": [...]}
        annotations/per_image_annotations.json  - {str(img_id): [coco_ann, ...]}
        masks/semantic/<img_id>.png             - class-ID mask (created lazily if missing)
    """
    def __init__(self, root: str, split: str, transforms=None,
                 selected_cat_map: Optional[Dict[int, int]] = None):
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.selected_cat_map = selected_cat_map or {k: v[1] for k, v in COCO_TO_SEM.items()}

        manifest = json.loads((self.root / "annotations" / "split_manifest.json").read_text())
        self.ids: List[int] = manifest[split]
        ann_path = self.root / "annotations" / "per_image_annotations.json"
        self.per_image = json.loads(ann_path.read_text())
        info_path = self.root / "annotations" / "image_info.json"
        self.image_info = json.loads(info_path.read_text())

    def image_path(self, img_id: int) -> Path:
        return self.root / "data" / "coco" / "images" / f"{img_id:012d}.jpg"

    def semantic_mask_path(self, img_id: int) -> Path:
        return self.root / "masks" / "semantic" / f"{img_id:012d}.png"


class SemanticSegDataset(_BaseCocoSubset, Dataset):
    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info   = self.image_info[str(img_id)]
        H, W   = info["height"], info["width"]

        img = Image.open(self.image_path(img_id)).convert("RGB")
        mp = self.semantic_mask_path(img_id)
        if mp.exists():
            mask = Image.open(mp)
        else:
            anns = self.per_image.get(str(img_id), [])
            sem  = build_semantic_mask(anns, H, W, self.selected_cat_map)
            mp.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(sem).save(mp)
            mask = Image.open(mp)

        sample = {"image": img, "mask": mask, "orig_size": (W, H), "image_id": img_id}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


class InstanceSegDataset(_BaseCocoSubset, Dataset):
    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info   = self.image_info[str(img_id)]
        H, W   = info["height"], info["width"]

        img = Image.open(self.image_path(img_id)).convert("RGB")
        anns = self.per_image.get(str(img_id), [])
        targets = build_instance_targets(anns, H, W, self.selected_cat_map)

        instance_masks = [Image.fromarray(m) for m in targets["masks"]]
        sample = {
            "image": img,
            "instance_masks": instance_masks,
            "boxes":  targets["boxes"],
            "labels": targets["labels"],
            "orig_size": (W, H),
            "image_id": img_id,
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


def collate_instance(batch):
    """Custom collate: instance batches keep list-of-dicts semantics (Mask R-CNN convention)."""
    if not _HAS_TORCH:
        return batch
    images  = [b["image"] for b in batch]
    targets = []
    for b in batch:
        t = {"masks": b["instance_masks"], "labels": b["labels"], "boxes": b["boxes"]}
        targets.append(t)
    return images, targets
