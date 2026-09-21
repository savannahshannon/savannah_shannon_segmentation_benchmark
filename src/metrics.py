"""Semantic + instance segmentation metrics.

All semantic metrics are computed from an accumulated confusion matrix so that
per-batch results can be averaged consistently across the whole test set.

    * pixel_accuracy = sum(diag) / sum(total)
    * per_class_iou  = TP / (TP + FP + FN)   [NaN if class absent from GT & pred]
    * mean_iou       = mean(per_class_iou over classes that appeared)
    * dice_per_class = 2 TP / (2 TP + FP + FN)  (== F1 pixelwise)
    * precision / recall computed both per class and macro-averaged

Boundary F1 follows Csurka et al. (2013): a predicted boundary pixel counts as
matched if it lies within `tol` pixels of a ground-truth boundary pixel, and
vice versa.

Instance metrics (mask AP, AP50, AP75, AR) use the pycocotools COCOeval
implementation, which is the field-standard reference for COCO benchmarks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ------------ Semantic ------------ #

class ConfusionMatrix:
    """Sparse-safe confusion matrix over K classes, ignore_index-aware."""
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray):
        """pred, target: same shape, integer class IDs; ignore_index skipped."""
        mask = (target != self.ignore_index) & (target < self.num_classes)
        p = pred[mask].astype(np.int64)
        t = target[mask].astype(np.int64)
        idx = t * self.num_classes + p
        bincount = np.bincount(idx, minlength=self.num_classes ** 2)
        self.mat += bincount.reshape(self.num_classes, self.num_classes)

    def reset(self):
        self.mat.fill(0)

    # --- derived metrics --- #
    def pixel_accuracy(self) -> float:
        total = self.mat.sum()
        return float(np.diag(self.mat).sum() / total) if total else 0.0

    def per_class_iou(self) -> np.ndarray:
        tp = np.diag(self.mat).astype(np.float64)
        fp = self.mat.sum(axis=0) - tp
        fn = self.mat.sum(axis=1) - tp
        denom = tp + fp + fn
        iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
        return iou

    def mean_iou(self, include_background: bool = True) -> float:
        iou = self.per_class_iou()
        if not include_background:
            iou = iou[1:]
        return float(np.nanmean(iou))

    def per_class_dice(self) -> np.ndarray:
        tp = np.diag(self.mat).astype(np.float64)
        fp = self.mat.sum(axis=0) - tp
        fn = self.mat.sum(axis=1) - tp
        denom = 2 * tp + fp + fn
        return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), np.nan)

    def macro_dice(self, include_background: bool = True) -> float:
        d = self.per_class_dice()
        if not include_background:
            d = d[1:]
        return float(np.nanmean(d))

    def pixel_precision(self) -> float:
        tp = np.diag(self.mat).astype(np.float64)
        fp = self.mat.sum(axis=0) - tp
        denom = tp + fp
        pc = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
        return float(np.nanmean(pc))

    def pixel_recall(self) -> float:
        tp = np.diag(self.mat).astype(np.float64)
        fn = self.mat.sum(axis=1) - tp
        denom = tp + fn
        rc = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
        return float(np.nanmean(rc))


# ------------ Boundary metrics ------------ #

def boundary_f1(pred: np.ndarray, target: np.ndarray, num_classes: int, tol: int = 2,
                ignore_index: int = 255) -> Dict[str, float]:
    """Boundary precision / recall / F1 (Csurka et al. 2013 style).

    Computed per class and macro-averaged. Uses simple binary edge extraction
    from class masks; the tolerance is realized with a distance transform.
    """
    from scipy.ndimage import distance_transform_edt, binary_erosion  # type: ignore
    valid = (target != ignore_index)
    prs, rcs = [], []
    for c in range(num_classes):
        gt = (target == c) & valid
        pr = (pred  == c) & valid
        if gt.sum() == 0 and pr.sum() == 0:
            continue
        # extract 1-pixel-wide boundaries
        gt_b = gt & ~binary_erosion(gt)
        pr_b = pr & ~binary_erosion(pr)
        if gt_b.sum() == 0 or pr_b.sum() == 0:
            prs.append(0.0); rcs.append(0.0); continue
        dt_gt = distance_transform_edt(~gt_b)
        dt_pr = distance_transform_edt(~pr_b)
        precision = float(((dt_gt <= tol) & pr_b).sum() / max(pr_b.sum(), 1))
        recall    = float(((dt_pr <= tol) & gt_b).sum() / max(gt_b.sum(), 1))
        prs.append(precision); rcs.append(recall)
    if not prs:
        return {"boundary_precision": float("nan"),
                "boundary_recall":    float("nan"),
                "boundary_f1":        float("nan")}
    p = float(np.mean(prs)); r = float(np.mean(rcs))
    f = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"boundary_precision": p, "boundary_recall": r, "boundary_f1": f}


# ------------ Instance ------------ #

def compute_instance_ap(pred_records: List[Dict], gt_records: List[Dict],
                        iou_thresholds: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Mask AP / AP50 / AP75 / AR using pycocotools.

    Inputs match COCO's json format subset:
        gt_records:   [{image_id, category_id, segmentation (RLE), area, iscrowd, id}]
        pred_records: [{image_id, category_id, segmentation (RLE), score}]
    Returns dict of scalar metrics; missing/empty predictions -> zeros.
    """
    from pycocotools.coco import COCO  # type: ignore
    from pycocotools.cocoeval import COCOeval  # type: ignore
    import json, tempfile, os

    if len(pred_records) == 0 or len(gt_records) == 0:
        return {"mask_AP": 0.0, "mask_AP50": 0.0, "mask_AP75": 0.0,
                "mask_AR":  0.0, "box_AP":    0.0}

    # Build in-memory COCO-style dataset for GT
    images = sorted({r["image_id"] for r in gt_records})
    cats   = sorted({r["category_id"] for r in gt_records})
    gt_json = {
        "images":      [{"id": i, "height": 512, "width": 512} for i in images],
        "annotations": [dict(r, id=k+1) for k, r in enumerate(gt_records)],
        "categories":  [{"id": c, "name": f"cat_{c}"} for c in cats],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(gt_json, f); gt_path = f.name
    try:
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(pred_records)
        results = {}
        for iou_type in ("segm", "bbox"):
            e = COCOeval(coco_gt, coco_dt, iou_type)
            e.evaluate(); e.accumulate(); e.summarize()
            stats = e.stats
            if iou_type == "segm":
                results.update({
                    "mask_AP":   float(stats[0]),
                    "mask_AP50": float(stats[1]),
                    "mask_AP75": float(stats[2]),
                    "mask_AR":   float(stats[8]),
                })
            else:
                results["box_AP"] = float(stats[0])
        return results
    finally:
        os.unlink(gt_path)


# ------------ Efficiency helpers ------------ #

@dataclass
class EfficiencyRecord:
    method: str
    task: str
    total_params: int = 0
    trainable_params: int = 0
    checkpoint_size_mb: float = 0.0
    total_training_time_s: float = 0.0
    time_per_epoch_s: float = 0.0
    latency_ms_per_image: float = 0.0
    throughput_images_per_s: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    input_resolution: str = "512x512"
    batch_size: int = 8
    precision: str = "fp32"
    hardware: str = "cpu"
    epochs_completed: int = 0

    def as_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
