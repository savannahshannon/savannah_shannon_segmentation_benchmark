"""Semantic and instance evaluators.

Both produce a dict of scalar metrics keyed exactly as they will appear in
the final CSV columns. Efficiency measurements (params, size, latency,
throughput, peak memory) come from `benchmark_efficiency` below.
"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

from .metrics import ConfusionMatrix, boundary_f1, compute_instance_ap
from .models import count_parameters


def _forward_semantic(model, images):
    out = model(images)
    if isinstance(out, dict) and "out" in out:
        return out["out"]
    if hasattr(out, "logits"):
        import torch.nn.functional as F
        logits = out.logits
        if logits.shape[-1] != images.shape[-1]:
            logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear",
                                   align_corners=False)
        return logits
    return out


def evaluate_semantic(model, data_loader, num_classes: int, device: Optional[str] = None,
                      compute_boundary: bool = True, ignore_index: int = 255,
                      save_predictions_dir: Optional[Path] = None) -> Dict:
    """Full semantic evaluation returning metrics + per-class IoU vector."""
    device = device or ("cuda" if _HAS_TORCH and torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    cm = ConfusionMatrix(num_classes, ignore_index=ignore_index)
    boundary_scores = []
    saved = 0
    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            masks  = batch["mask"].cpu().numpy()
            logits = _forward_semantic(model, images)
            pred = logits.argmax(dim=1).cpu().numpy()
            for i in range(pred.shape[0]):
                cm.update(pred[i], masks[i])
                if compute_boundary:
                    try:
                        b = boundary_f1(pred[i], masks[i], num_classes, tol=2,
                                        ignore_index=ignore_index)
                        boundary_scores.append(b["boundary_f1"])
                    except Exception:
                        pass
                if save_predictions_dir is not None and saved < 12:
                    from PIL import Image
                    save_predictions_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(pred[i].astype(np.uint8)).save(
                        save_predictions_dir / f"pred_{saved:04d}.png")
                    saved += 1

    per_class_iou = cm.per_class_iou()
    return {
        "pixel_accuracy": cm.pixel_accuracy(),
        "mean_iou":       cm.mean_iou(include_background=True),
        "mean_dice":      cm.macro_dice(include_background=True),
        "pixel_precision":cm.pixel_precision(),
        "pixel_recall":   cm.pixel_recall(),
        "per_class_iou":  per_class_iou.tolist(),
        "per_class_dice": cm.per_class_dice().tolist(),
        "confusion_matrix": cm.mat.tolist(),
        "boundary_f1_mean": float(np.nanmean(boundary_scores)) if boundary_scores else float("nan"),
    }


def evaluate_instance_maskrcnn(model, data_loader, device: Optional[str] = None,
                               score_thr: float = 0.05) -> Dict:
    """Run Mask R-CNN over the test set and compute mask AP via pycocotools."""
    device = device or ("cuda" if _HAS_TORCH and torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    pred_records: List[Dict] = []
    gt_records: List[Dict] = []
    import pycocotools.mask as maskUtils  # type: ignore
    inst_id = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for tgt, out in zip(targets, outputs):
                img_id = int(tgt.get("image_id", inst_id))
                # ---- GT records ---- #
                for j in range(len(tgt["labels"])):
                    m = tgt["masks"][j].cpu().numpy().astype(np.uint8)
                    if m.sum() == 0: continue
                    rle = maskUtils.encode(np.asfortranarray(m))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    gt_records.append({
                        "image_id": img_id,
                        "category_id": int(tgt["labels"][j].item()),
                        "segmentation": rle,
                        "area": int(m.sum()),
                        "iscrowd": 0,
                        "bbox": tgt["boxes"][j].cpu().tolist(),
                    })
                # ---- Predictions ---- #
                keep = out["scores"] >= score_thr
                for k in torch.nonzero(keep).flatten().tolist():
                    m = (out["masks"][k, 0] > 0.5).cpu().numpy().astype(np.uint8)
                    if m.sum() == 0: continue
                    rle = maskUtils.encode(np.asfortranarray(m))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    pred_records.append({
                        "image_id": img_id,
                        "category_id": int(out["labels"][k].item()),
                        "segmentation": rle,
                        "score": float(out["scores"][k].item()),
                        "bbox": out["boxes"][k].cpu().tolist(),
                    })
                inst_id += 1

    ap = compute_instance_ap(pred_records, gt_records)
    ap.update({"n_predictions": len(pred_records), "n_ground_truth": len(gt_records)})
    return ap


def benchmark_efficiency(model, sample_input, name: str, task: str,
                         warmup: int = 10, iters: int = 100,
                         device: Optional[str] = None) -> Dict:
    """Measure model params, checkpoint size (fp32 weights), latency, throughput."""
    device = device or ("cuda" if _HAS_TORCH and torch.cuda.is_available() else "cpu")
    result = {"method": name, "task": task}

    # --- parameter counts --- #
    pc = count_parameters(model) if hasattr(model, "parameters") else {"total": 0, "trainable": 0}
    result["total_params"] = pc["total"]
    result["trainable_params"] = pc["trainable"]

    # --- weight-only checkpoint size --- #
    if hasattr(model, "state_dict"):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            size_mb = os.path.getsize(f.name) / (1024 ** 2)
            os.unlink(f.name)
        result["checkpoint_size_mb"] = round(size_mb, 3)
    else:
        result["checkpoint_size_mb"] = 0.0

    # --- latency --- #
    if not hasattr(model, "eval"):
        result["latency_ms_per_image"] = float("nan")
        result["throughput_images_per_s"] = float("nan")
        result["peak_gpu_memory_mb"] = 0.0
        return result

    model.eval().to(device)
    x = sample_input.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x) if task != "instance" else model([x[0]])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(iters):
            _ = model(x) if task != "instance" else model([x[0]])
        if device.startswith("cuda"): torch.cuda.synchronize()
        dt = (time.time() - t0) / iters
    result["latency_ms_per_image"] = round(dt * 1000.0, 3)
    result["throughput_images_per_s"] = round(1.0 / dt, 3)
    result["peak_gpu_memory_mb"] = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                                    if device.startswith("cuda") else 0.0)
    result["hardware"] = device
    result["precision"] = "fp32"
    result["input_resolution"] = f"{x.shape[-2]}x{x.shape[-1]}"
    result["batch_size"] = int(x.shape[0])
    gc.collect()
    return result
