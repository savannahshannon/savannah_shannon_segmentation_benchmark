"""Mask utilities for COCO -> semantic/instance conversion.

Policy (documented per assignment section 8):
  * A pixel that belongs to any selected foreground class receives that class's
    integer ID; overlaps between two selected classes are resolved by keeping
    the *smaller* instance on top (small-object preservation).
  * "iscrowd" annotations for selected classes contribute pixels marked with
    the ignore_index and are excluded from both training loss and evaluation.
  * All non-selected classes are treated as background (ID 0).

Class IDs (background + 5 foreground):
    0 background   1 person   2 car   3 bicycle   4 dog   5 cat

The mapping uses the ORIGINAL COCO category IDs so any COCO 2017 annotation
file can be reduced to this task without renaming categories.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# COCO 2017 category ID -> (name, semantic ID in our task)
COCO_TO_SEM: Dict[int, Tuple[str, int]] = {
    1:  ("person",  1),
    3:  ("car",     2),
    2:  ("bicycle", 3),
    18: ("dog",     4),
    17: ("cat",     5),
}
SEM_ID_TO_NAME = {0: "background", 1: "person", 2: "car",
                  3: "bicycle", 4: "dog", 5: "cat"}
NUM_CLASSES = 6
IGNORE_INDEX = 255


def try_import_pycocotools():
    """Lazy import so this module is usable in environments without pycocotools."""
    try:
        from pycocotools import mask as maskUtils  # type: ignore
        return maskUtils
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pycocotools is required for polygon/RLE decoding. "
            "Install with: pip install pycocotools"
        ) from e


def ann_to_binary_mask(ann: dict, height: int, width: int) -> np.ndarray:
    """Decode a COCO annotation (polygon or RLE) to a binary HxW mask (uint8)."""
    maskUtils = try_import_pycocotools()
    seg = ann["segmentation"]
    if isinstance(seg, list):  # polygons
        rles = maskUtils.frPyObjects(seg, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(seg, dict):  # already RLE
        if isinstance(seg["counts"], list):
            rle = maskUtils.frPyObjects([seg], height, width)[0]
        else:
            rle = seg
    else:
        return np.zeros((height, width), dtype=np.uint8)
    m = maskUtils.decode(rle).astype(np.uint8)
    if m.ndim == 3:
        m = m.max(axis=-1)
    return m


def build_semantic_mask(annotations: Sequence[dict],
                        height: int,
                        width: int,
                        selected_cat_map: Dict[int, int] = None,
                        ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    """Combine per-instance masks for one image into a single semantic mask.

    Overlap policy: smaller-area instance on top (preserves small objects).
    Crowd regions of selected classes -> ignore_index.
    """
    if selected_cat_map is None:
        selected_cat_map = {k: v[1] for k, v in COCO_TO_SEM.items()}

    # Filter and sort by area (largest first so smaller overwrites later).
    kept = [a for a in annotations if a["category_id"] in selected_cat_map]
    kept.sort(key=lambda a: a.get("area", 0), reverse=True)

    semantic = np.zeros((height, width), dtype=np.uint8)
    for a in kept:
        m = ann_to_binary_mask(a, height, width)
        cid = selected_cat_map[a["category_id"]]
        if a.get("iscrowd", 0) == 1:
            semantic[m == 1] = ignore_index
        else:
            semantic[m == 1] = cid
    return semantic


def build_instance_targets(annotations: Sequence[dict],
                           height: int,
                           width: int,
                           selected_cat_map: Dict[int, int] = None) -> Dict:
    """Return per-instance masks + labels + boxes suitable for Mask R-CNN."""
    if selected_cat_map is None:
        selected_cat_map = {k: v[1] for k, v in COCO_TO_SEM.items()}

    masks: List[np.ndarray] = []
    labels: List[int] = []
    boxes: List[List[float]] = []
    for a in annotations:
        if a.get("iscrowd", 0) == 1:
            continue
        if a["category_id"] not in selected_cat_map:
            continue
        m = ann_to_binary_mask(a, height, width)
        if m.sum() == 0:
            continue
        ys, xs = np.where(m > 0)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        masks.append(m)
        labels.append(selected_cat_map[a["category_id"]])
        boxes.append([float(x0), float(y0), float(x1), float(y1)])
    return {
        "masks":  np.stack(masks, axis=0) if masks else np.zeros((0, height, width), dtype=np.uint8),
        "labels": np.array(labels, dtype=np.int64),
        "boxes":  np.array(boxes,  dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
    }


def save_mask_png(mask: np.ndarray, path: Path):
    """Write a class-ID mask as an 8-bit PNG (nearest-neighbor safe)."""
    from PIL import Image
    Image.fromarray(mask.astype(np.uint8)).save(str(path))


def load_mask_png(path: Path) -> np.ndarray:
    from PIL import Image
    return np.array(Image.open(str(path)))


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a class-ID mask to an RGB visualization."""
    palette = np.array([
        [  0,   0,   0],   # 0 background
        [220,  20,  60],   # 1 person   crimson
        [ 30, 144, 255],   # 2 car      dodger blue
        [255, 215,   0],   # 3 bicycle  gold
        [ 34, 139,  34],   # 4 dog      forest green
        [148,   0, 211],   # 5 cat      dark violet
    ], dtype=np.uint8)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        out[mask == c] = palette[c]
    # ignore pixels -> gray
    out[mask == IGNORE_INDEX] = [128, 128, 128]
    return out


def save_split_manifest(splits: Dict[str, List[int]], path: Path):
    """Persist the exact image IDs used in every partition (assignment §7)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: sorted(map(int, v)) for k, v in splits.items()}, f, indent=2)


def load_split_manifest(path: Path) -> Dict[str, List[int]]:
    with open(path) as f:
        return json.load(f)
