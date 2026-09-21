"""Reusable trainer for semantic segmentation.

Semantic models are trained with CE + Dice loss on (B, C, H, W) logits.
Instance models (Mask R-CNN, YOLO) have their own train drivers; see
`train_maskrcnn` and the YOLO shim in benchmark.py.

`train_model` returns a dict of per-epoch histories plus best-checkpoint info
suitable for saving into results/<model>/history.json.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as e:  # pragma: no cover
    raise ImportError("PyTorch is required for training.") from e

from .metrics import ConfusionMatrix
from .losses import CombinedSegLoss


def _forward_semantic(model, images):
    """Uniform interface: return logits (B, C, H, W)."""
    out = model(images)
    if isinstance(out, dict) and "out" in out:
        return out["out"]
    if hasattr(out, "logits"):  # HuggingFace SegFormer
        logits = out.logits
        # SegFormer outputs at 1/4 resolution; upsample to input
        if logits.shape[-1] != images.shape[-1]:
            logits = nn.functional.interpolate(logits, size=images.shape[-2:],
                                               mode="bilinear", align_corners=False)
        return logits
    return out


def train_model(model,
                train_loader,
                val_loader,
                num_classes: int,
                epochs: int = 25,
                lr: float = 1e-4,
                weight_decay: float = 1e-4,
                ignore_index: int = 255,
                device: Optional[str] = None,
                mixed_precision: bool = True,
                ckpt_path: Optional[Path] = None,
                log_fn: Callable[[str], None] = print) -> Dict:
    """Train a semantic segmentation model and return training history."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = CombinedSegLoss(num_classes, ignore_index=ignore_index)
    scaler = torch.cuda.amp.GradScaler(enabled=(mixed_precision and device.startswith("cuda")))

    history = {"train_loss": [], "val_loss": [], "val_miou": [], "val_dice": [],
               "val_pixel_acc": [], "epoch_time_s": [], "lr": []}
    best_miou, best_state, best_epoch = -1.0, None, -1
    total_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        ep_start = time.time()
        running = 0.0; n = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks  = batch["mask"].to(device,  non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(mixed_precision and device.startswith("cuda"))):
                logits = _forward_semantic(model, images)
                loss = criterion(logits, masks)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            running += loss.item() * images.size(0)
            n += images.size(0)
        train_loss = running / max(n, 1)
        scheduler.step()

        # ------ validation ------ #
        model.eval()
        cm = ConfusionMatrix(num_classes, ignore_index=ignore_index)
        v_loss = 0.0; v_n = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                masks  = batch["mask"].to(device,  non_blocking=True)
                logits = _forward_semantic(model, images)
                v_loss += criterion(logits, masks).item() * images.size(0)
                v_n    += images.size(0)
                pred = logits.argmax(dim=1).cpu().numpy()
                gt   = masks.cpu().numpy()
                for p, t in zip(pred, gt):
                    cm.update(p, t)
        v_loss /= max(v_n, 1)
        miou = cm.mean_iou()
        dice = cm.macro_dice()
        pa   = cm.pixel_accuracy()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(v_loss)
        history["val_miou"].append(miou)
        history["val_dice"].append(dice)
        history["val_pixel_acc"].append(pa)
        history["epoch_time_s"].append(time.time() - ep_start)
        history["lr"].append(opt.param_groups[0]["lr"])

        log_fn(f"[epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={v_loss:.4f} "
               f"miou={miou:.4f} dice={dice:.4f} pa={pa:.4f} "
               f"time={history['epoch_time_s'][-1]:.1f}s")

        if miou > best_miou:
            best_miou = miou; best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            if ckpt_path is not None:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, ckpt_path)

    history["total_training_time_s"] = time.time() - total_start
    history["best_miou"] = best_miou
    history["best_epoch"] = best_epoch
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# ------ Mask R-CNN trainer ------ #
def train_maskrcnn(model, train_loader, val_loader, epochs: int = 25,
                   lr: float = 1e-4, weight_decay: float = 1e-4,
                   device: Optional[str] = None, ckpt_path: Optional[Path] = None,
                   log_fn: Callable[[str], None] = print) -> Dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = {"train_total_loss": [], "epoch_time_s": []}
    total_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        ep_start = time.time(); running = 0.0; n = 0
        for images, targets in train_loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            running += float(loss.item()); n += 1
        scheduler.step()
        history["train_total_loss"].append(running / max(n, 1))
        history["epoch_time_s"].append(time.time() - ep_start)
        log_fn(f"[maskrcnn epoch {epoch:03d}] total_loss={history['train_total_loss'][-1]:.4f} "
               f"time={history['epoch_time_s'][-1]:.1f}s")
        if ckpt_path is not None:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

    history["total_training_time_s"] = time.time() - total_start
    return history
