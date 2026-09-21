"""Plotting utilities.

Produces every figure required by assignment §49:
    1  Semantic model vs mIoU          (miou_comparison.png)
    2  Semantic model vs Dice          (dice_comparison.png)
    3  Model vs parameter count        (parameters.png)
    4  Model vs saved model size       (model_size.png)
    5  Model vs training time          (training_time.png)
    6  Model vs inference imgs/s       (inference_speed.png)
    7  mIoU vs parameter count         (miou_vs_parameters.png)
    8  mIoU vs inference latency       (miou_vs_latency.png)
    9  Per-class IoU across models     (per_class_iou.png)
    10 Mask R-CNN vs YOLO seg AP       (instance_mask_ap.png)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def _bar(ax, x, y, ylabel, title, color="#2b8cbe"):
    bars = ax.bar(x, y, color=color, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, y):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:.3f}" if isinstance(v, float) else f"{v}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(x))); ax.set_xticklabels(x, rotation=30, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def plot_semantic_bars(semantic_df, plots_dir: Path):
    """Figures 1 and 2: mIoU and Dice bar charts."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    x = semantic_df["Model"].tolist()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _bar(ax, x, semantic_df["mIoU"].tolist(), "mean IoU",
         "Figure 1 — Semantic Segmentation mIoU", color="#2b8cbe")
    fig.savefig(plots_dir / "miou_comparison.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _bar(ax, x, semantic_df["Dice"].tolist(), "macro Dice",
         "Figure 2 — Semantic Segmentation Dice", color="#31a354")
    fig.savefig(plots_dir / "dice_comparison.png"); plt.close(fig)


def plot_efficiency_bars(eff_df, plots_dir: Path):
    """Figures 3, 4, 5, 6."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    x = eff_df["method"].tolist()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    _bar(ax, x, (eff_df["total_params"] / 1e6).tolist(),
         "params (millions)", "Figure 3 — Total parameters", color="#e6550d")
    fig.savefig(plots_dir / "parameters.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    _bar(ax, x, eff_df["checkpoint_size_mb"].tolist(),
         "size (MB)", "Figure 4 — Checkpoint size", color="#756bb1")
    fig.savefig(plots_dir / "model_size.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    _bar(ax, x, eff_df["total_training_time_s"].tolist(),
         "training time (s)", "Figure 5 — Total training time", color="#636363")
    fig.savefig(plots_dir / "training_time.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    _bar(ax, x, eff_df["throughput_images_per_s"].tolist(),
         "images / s", "Figure 6 — Inference throughput", color="#3182bd")
    fig.savefig(plots_dir / "inference_speed.png"); plt.close(fig)


def plot_tradeoffs(semantic_df, eff_df, plots_dir: Path):
    """Figures 7 and 8: mIoU vs params, mIoU vs latency."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = semantic_df.merge(eff_df, left_on="Model", right_on="method", how="left")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(df["total_params"] / 1e6, df["mIoU"], s=80, c="#2b8cbe", edgecolor="black")
    for _, r in df.iterrows():
        ax.annotate(r["Model"], (r["total_params"] / 1e6, r["mIoU"]),
                    xytext=(5, 3), textcoords="offset points", fontsize=9)
    ax.set_xlabel("total parameters (M)"); ax.set_ylabel("mean IoU")
    ax.set_title("Figure 7 — mIoU vs parameter count")
    ax.grid(alpha=0.3); fig.savefig(plots_dir / "miou_vs_parameters.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(df["latency_ms_per_image"], df["mIoU"], s=80, c="#31a354", edgecolor="black")
    for _, r in df.iterrows():
        ax.annotate(r["Model"], (r["latency_ms_per_image"], r["mIoU"]),
                    xytext=(5, 3), textcoords="offset points", fontsize=9)
    ax.set_xlabel("latency (ms/image)"); ax.set_ylabel("mean IoU")
    ax.set_title("Figure 8 — mIoU vs inference latency")
    ax.grid(alpha=0.3); fig.savefig(plots_dir / "miou_vs_latency.png"); plt.close(fig)


def plot_per_class_iou(per_class: Dict[str, List[float]], class_names: Sequence[str],
                       plots_dir: Path):
    """Figure 9: grouped bars per class, one bar per model."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    models = list(per_class.keys())
    x = np.arange(len(class_names))
    w = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.get_cmap("tab10").colors
    for i, m in enumerate(models):
        vals = per_class[m]
        ax.bar(x + (i - len(models)/2) * w + w/2, vals, w, label=m, color=colors[i % 10])
    ax.set_xticks(x); ax.set_xticklabels(class_names)
    ax.set_ylabel("IoU"); ax.set_title("Figure 9 — Per-class IoU across semantic models")
    ax.legend(ncol=min(len(models), 4), fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(plots_dir / "per_class_iou.png"); plt.close(fig)


def plot_instance_ap(instance_df, plots_dir: Path):
    """Figure 10: Mask R-CNN vs YOLO seg, grouped bars for AP / AP50 / AP75."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    models = instance_df["Model"].tolist()
    metrics = ["Mask_AP", "AP50", "AP75"]
    x = np.arange(len(metrics)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, m in enumerate(models):
        vals = instance_df[instance_df["Model"] == m][metrics].values.flatten()
        ax.bar(x + (i - len(models)/2) * w + w/2, vals, w, label=m,
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("AP"); ax.set_title("Figure 10 — Instance mask AP: Mask R-CNN vs YOLO Segmentation")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.savefig(plots_dir / "instance_mask_ap.png"); plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], out_path: Path):
    """Class confusion matrix as normalized heat map."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.where(row_sums > 0, cm / np.maximum(row_sums, 1), 0)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right"); ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("ground truth")
    ax.set_title(out_path.stem.replace("_", " "))
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path); plt.close(fig)


def qualitative_grid(images: List[np.ndarray], gt_masks: List[np.ndarray],
                     pred_masks: Dict[str, List[np.ndarray]], out_path: Path):
    """Side-by-side grid: image | GT | pred_model1 | pred_model2 | ..."""
    from .mask_utils import colorize_mask
    n = min(4, len(images))
    models = list(pred_masks.keys())
    cols = 2 + len(models)
    fig, axes = plt.subplots(n, cols, figsize=(3 * cols, 3 * n))
    if n == 1: axes = axes[None, :]
    for r in range(n):
        axes[r, 0].imshow(images[r]); axes[r, 0].set_title("image" if r == 0 else "")
        axes[r, 1].imshow(colorize_mask(gt_masks[r])); axes[r, 1].set_title("ground truth" if r == 0 else "")
        for c, m in enumerate(models, start=2):
            axes[r, c].imshow(colorize_mask(pred_masks[m][r]))
            if r == 0: axes[r, c].set_title(m)
        for a in axes[r]: a.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path); plt.close(fig)
