"""End-to-end benchmark driver.

This module orchestrates:
  1. Building the model list per configuration
  2. Training semantic models (and Mask R-CNN)
  3. Running the YOLO segmentation training loop through Ultralytics
  4. Evaluating each model on the *held-out test set*
  5. Measuring efficiency (params, size, latency, throughput, peak memory)
  6. Persisting three CSVs + all figures under results/ and plots/

The design keeps semantic and instance evaluators separate as required by
assignment §26.
"""
from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

from .mask_utils import NUM_CLASSES, SEM_ID_TO_NAME, IGNORE_INDEX
from .visualize import (plot_semantic_bars, plot_efficiency_bars,
                         plot_tradeoffs, plot_per_class_iou, plot_instance_ap,
                         plot_confusion_matrix)

# Lazy imports for training / evaluation (require torch)
def _lazy_train_imports():
    from .dataset import (SemanticSegDataset, InstanceSegDataset,
                          build_train_transforms, build_eval_transforms,
                          collate_instance)
    from .models import get_segmentation_model, count_parameters
    from .train import train_model, train_maskrcnn
    from .evaluate import (evaluate_semantic, evaluate_instance_maskrcnn,
                           benchmark_efficiency)
    return dict(
        SemanticSegDataset=SemanticSegDataset, InstanceSegDataset=InstanceSegDataset,
        build_train_transforms=build_train_transforms, build_eval_transforms=build_eval_transforms,
        collate_instance=collate_instance,
        get_segmentation_model=get_segmentation_model, count_parameters=count_parameters,
        train_model=train_model, train_maskrcnn=train_maskrcnn,
        evaluate_semantic=evaluate_semantic,
        evaluate_instance_maskrcnn=evaluate_instance_maskrcnn,
        benchmark_efficiency=benchmark_efficiency,
    )


# ---------------------------------- Helpers ---------------------------------- #

def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> Dict:
    with open(path) as f: return yaml.safe_load(f)


def build_loaders(cfg: Dict, task: str, num_workers: int = 2):
    """Return (train, val, test) loaders for the requested task."""
    from torch.utils.data import DataLoader
    m = _lazy_train_imports()
    input_size = tuple(cfg["preprocessing"]["input_size"])
    train_tf = m["build_train_transforms"](input_size, cfg["preprocessing"]["augmentation"])
    eval_tf  = m["build_eval_transforms"](input_size)
    root = "."
    DsCls = m["SemanticSegDataset"] if task == "semantic" else m["InstanceSegDataset"]
    train = DsCls(root, "train", transforms=train_tf)
    val   = DsCls(root, "val",   transforms=eval_tf)
    test  = DsCls(root, "test",  transforms=eval_tf)
    bs = cfg["training"]["batch_size"]
    collate = None if task == "semantic" else m["collate_instance"]
    return (
        DataLoader(train, batch_size=bs, shuffle=True,  num_workers=num_workers, collate_fn=collate),
        DataLoader(val,   batch_size=bs, shuffle=False, num_workers=num_workers, collate_fn=collate),
        DataLoader(test,  batch_size=bs, shuffle=False, num_workers=num_workers, collate_fn=collate),
    )


# ---------------------------------- Runners ---------------------------------- #

def run_semantic_model(model_name: str, cfg: Dict, out_root: Path) -> Dict:
    """Train, evaluate, and record efficiency for one semantic model."""
    m = _lazy_train_imports()
    device = "cuda" if _HAS_TORCH and torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = build_loaders(cfg, task="semantic")

    model_cfg = cfg["models"][model_name]
    model = m["get_segmentation_model"](model_name, num_classes=NUM_CLASSES,
                                    pretrained=model_cfg.get("pretrained", True),
                                    **{k: v for k, v in model_cfg.items()
                                       if k not in ("enabled", "task", "pretrained")})

    ckpt_path = out_root / "checkpoints" / f"best_{model_name}.pt"
    log_path  = out_root / "logs" / f"{model_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a")
    def log(s):
        print(s); log_file.write(s + "\n"); log_file.flush()

    log(f"=== training {model_name} on {device} ===")
    history = m["train_model"](model, train_loader, val_loader,
                          num_classes=NUM_CLASSES,
                          epochs=cfg["training"]["epochs"],
                          lr=cfg["training"]["learning_rate"],
                          weight_decay=cfg["training"]["weight_decay"],
                          ignore_index=cfg["dataset"]["ignore_index"],
                          device=device,
                          mixed_precision=cfg["experiment"]["mixed_precision"],
                          ckpt_path=ckpt_path, log_fn=log)

    # ---- evaluation on test ---- #
    log(f"=== evaluating {model_name} on test set ===")
    pred_dir = out_root / "predictions" / model_name
    metrics = m["evaluate_semantic"](model, test_loader, num_classes=NUM_CLASSES,
                                 device=device, compute_boundary=True,
                                 save_predictions_dir=pred_dir)

    # ---- confusion matrix plot ---- #
    class_names = [SEM_ID_TO_NAME[i] for i in range(NUM_CLASSES)]
    plot_confusion_matrix(np.asarray(metrics["confusion_matrix"]), class_names,
                          out_root / "confusion_matrices" / f"{model_name}.png")

    # ---- efficiency ---- #
    sample = next(iter(test_loader))["image"][:1]
    eff = m["benchmark_efficiency"](model, sample, name=model_name, task="semantic",
                                warmup=cfg["evaluation"]["latency_warmup_iters"],
                                iters=cfg["evaluation"]["latency_images"], device=device)
    eff["total_training_time_s"] = round(history["total_training_time_s"], 2)
    eff["time_per_epoch_s"] = round(float(np.mean(history["epoch_time_s"])), 2)
    eff["epochs_completed"] = len(history["epoch_time_s"])

    # ---- save per-model artifacts ---- #
    (out_root / "results" / model_name).mkdir(parents=True, exist_ok=True)
    with open(out_root / "results" / model_name / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    with open(out_root / "results" / model_name / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=float)
    with open(out_root / "results" / model_name / "efficiency.json", "w") as f:
        json.dump(eff, f, indent=2, default=float)

    log_file.close()
    return {"metrics": metrics, "efficiency": eff, "history": history}


def run_traditional_kmeans(cfg: Dict, out_root: Path) -> Dict:
    """Traditional baseline: qualitative + class-mapped quantitative if valid."""
    from .models import KMeansSegmenter
    from PIL import Image
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(it, **kw): return it
    m = _lazy_train_imports()

    seg = KMeansSegmenter(num_classes=NUM_CLASSES, n_clusters=NUM_CLASSES)

    _, _, test_loader = build_loaders(cfg, task="semantic", num_workers=0)
    train_loader, _, _ = build_loaders(cfg, task="semantic", num_workers=0)

    # Fit mapping on <=200 training images (assignment forbids using test data)
    imgs, msks = [], []
    for i, batch in enumerate(train_loader):
        if i >= 25: break
        for j in range(batch["image"].shape[0]):
            img = batch["image"][j].numpy().transpose(1, 2, 0)
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            imgs.append(img); msks.append(batch["mask"][j].numpy())
    seg.fit_class_mapping(imgs, msks)

    # Evaluate on test set
    from .metrics import ConfusionMatrix
    cm = ConfusionMatrix(NUM_CLASSES, ignore_index=IGNORE_INDEX)
    pred_dir = out_root / "predictions" / "kmeans"; pred_dir.mkdir(parents=True, exist_ok=True)
    latencies = []
    saved = 0
    for batch in tqdm(test_loader, desc="kmeans"):
        for j in range(batch["image"].shape[0]):
            img = batch["image"][j].numpy().transpose(1, 2, 0)
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            t0 = time.time(); pred = seg.predict(img); latencies.append(time.time() - t0)
            cm.update(pred, batch["mask"][j].numpy())
            if saved < 12:
                Image.fromarray(pred.astype(np.uint8)).save(pred_dir / f"pred_{saved:04d}.png")
                saved += 1

    metrics = {
        "pixel_accuracy": cm.pixel_accuracy(),
        "mean_iou":       cm.mean_iou(),
        "mean_dice":      cm.macro_dice(),
        "pixel_precision":cm.pixel_precision(),
        "pixel_recall":   cm.pixel_recall(),
        "per_class_iou":  cm.per_class_iou().tolist(),
        "confusion_matrix": cm.mat.tolist(),
        "boundary_f1_mean": float("nan"),
    }
    eff = {
        "method": "kmeans", "task": "traditional",
        "total_params": 0, "trainable_params": 0,
        "checkpoint_size_mb": 0.0,
        "total_training_time_s": 0.0, "time_per_epoch_s": 0.0, "epochs_completed": 0,
        "latency_ms_per_image": round(float(np.mean(latencies)) * 1000, 2),
        "throughput_images_per_s": round(1.0 / float(np.mean(latencies)), 2),
        "peak_gpu_memory_mb": 0.0,
        "input_resolution": f"{cfg['preprocessing']['input_size'][0]}x{cfg['preprocessing']['input_size'][1]}",
        "batch_size": cfg["training"]["batch_size"], "precision": "fp32", "hardware": "cpu",
    }
    (out_root / "results" / "kmeans").mkdir(parents=True, exist_ok=True)
    with open(out_root / "results" / "kmeans" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    with open(out_root / "results" / "kmeans" / "efficiency.json", "w") as f:
        json.dump(eff, f, indent=2, default=float)
    return {"metrics": metrics, "efficiency": eff, "history": {}}


def run_maskrcnn(cfg: Dict, out_root: Path) -> Dict:
    m = _lazy_train_imports()
    device = "cuda" if _HAS_TORCH and torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = build_loaders(cfg, task="instance")
    model = m["get_segmentation_model"]("maskrcnn", num_classes=NUM_CLASSES, pretrained=True)
    ckpt_path = out_root / "checkpoints" / "best_maskrcnn.pt"
    log_path  = out_root / "logs" / "maskrcnn.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log_file:
        def log(s): print(s); log_file.write(s + "\n"); log_file.flush()
        history = m["train_maskrcnn"](model, train_loader, val_loader,
                                  epochs=cfg["training"]["epochs"],
                                  lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"],
                                  device=device, ckpt_path=ckpt_path, log_fn=log)
        log("=== evaluating maskrcnn ===")
        metrics = m["evaluate_instance_maskrcnn"](model, test_loader, device=device,
                                              score_thr=cfg["instance"]["score_threshold"])

    # efficiency: latency needs a torch image
    sample_batch = next(iter(test_loader))[0]
    sample = torch.stack([sample_batch[0]])
    eff = m["benchmark_efficiency"](model, sample, name="maskrcnn", task="instance",
                                warmup=cfg["evaluation"]["latency_warmup_iters"],
                                iters=cfg["evaluation"]["latency_images"], device=device)
    eff["total_training_time_s"] = round(history["total_training_time_s"], 2)
    eff["time_per_epoch_s"] = round(float(np.mean(history["epoch_time_s"])), 2)
    eff["epochs_completed"] = len(history["epoch_time_s"])

    (out_root / "results" / "maskrcnn").mkdir(parents=True, exist_ok=True)
    with open(out_root / "results" / "maskrcnn" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    with open(out_root / "results" / "maskrcnn" / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=float)
    with open(out_root / "results" / "maskrcnn" / "efficiency.json", "w") as f:
        json.dump(eff, f, indent=2, default=float)
    return {"metrics": metrics, "efficiency": eff, "history": history}


def run_yolo_seg(cfg: Dict, out_root: Path) -> Dict:
    """Trains and evaluates a YOLOv8 segmentation model via Ultralytics.

    The YOLO trainer emits its own metrics; we translate them into the
    assignment's mask AP / AP50 / AP75 slots. The YOLO dataset YAML must be
    produced by scripts/prepare_coco_subset.py in Ultralytics format.
    """
    from ultralytics import YOLO
    variant = cfg["models"]["yolo_seg"].get("variant", "yolov8n-seg.pt")
    yaml_path = out_root / "annotations" / "yolo_seg.yaml"
    assert yaml_path.exists(), f"Missing {yaml_path}. Run scripts/prepare_coco_subset.py --yolo-yaml"

    yolo = YOLO(variant)
    log_path = out_root / "logs" / "yolo_seg.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    yolo.train(
        data=str(yaml_path),
        epochs=cfg["training"]["epochs"],
        imgsz=cfg["preprocessing"]["input_size"][0],
        batch=cfg["training"]["batch_size"],
        project=str(out_root / "checkpoints"),
        name="yolo_seg",
        exist_ok=True,
        seed=cfg["experiment"]["seed"],
        verbose=False,
    )
    total_time = time.time() - t0

    val_metrics = yolo.val(data=str(yaml_path), split="test", verbose=False)
    # Ultralytics returns .seg for segmentation, .box for detection
    seg = getattr(val_metrics, "seg", None)
    metrics = {
        "mask_AP":   float(seg.map)      if seg else 0.0,
        "mask_AP50": float(seg.map50)    if seg else 0.0,
        "mask_AP75": float(seg.map75)    if seg else 0.0,
        "mask_AR":   float(seg.mr)       if seg else 0.0,
        "box_AP":    float(val_metrics.box.map) if val_metrics.box is not None else 0.0,
    }

    total_params, trainable = 0, 0
    for p in yolo.model.parameters():
        total_params += p.numel()
        if p.requires_grad: trainable += p.numel()

    eff = {
        "method": "yolo_seg", "task": "instance",
        "total_params": int(total_params), "trainable_params": int(trainable),
        "checkpoint_size_mb": round(Path(yolo.trainer.best).stat().st_size / (1024**2), 3)
            if hasattr(yolo, "trainer") and yolo.trainer.best else 0.0,
        "total_training_time_s": round(total_time, 2),
        "time_per_epoch_s": round(total_time / max(cfg["training"]["epochs"], 1), 2),
        "epochs_completed": cfg["training"]["epochs"],
        "latency_ms_per_image": 0.0,   # populated below
        "throughput_images_per_s": 0.0,
        "peak_gpu_memory_mb": 0.0,
        "input_resolution": f"{cfg['preprocessing']['input_size'][0]}x{cfg['preprocessing']['input_size'][1]}",
        "batch_size": cfg["training"]["batch_size"], "precision": "fp32",
        "hardware": "cuda" if _HAS_TORCH and torch.cuda.is_available() else "cpu",
    }
    # simple latency probe
    import numpy as _np
    dummy = _np.zeros((cfg["preprocessing"]["input_size"][0],
                        cfg["preprocessing"]["input_size"][1], 3), dtype=_np.uint8)
    for _ in range(cfg["evaluation"]["latency_warmup_iters"]): yolo.predict(dummy, verbose=False)
    N = cfg["evaluation"]["latency_images"]; t0 = time.time()
    for _ in range(N): yolo.predict(dummy, verbose=False)
    dt = (time.time() - t0) / N
    eff["latency_ms_per_image"] = round(dt * 1000, 2)
    eff["throughput_images_per_s"] = round(1.0 / dt, 2)

    (out_root / "results" / "yolo_seg").mkdir(parents=True, exist_ok=True)
    with open(out_root / "results" / "yolo_seg" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    with open(out_root / "results" / "yolo_seg" / "efficiency.json", "w") as f:
        json.dump(eff, f, indent=2, default=float)
    return {"metrics": metrics, "efficiency": eff}


# ---------------------------- Aggregation & CSVs ---------------------------- #

def aggregate_results(out_root: Path, cfg: Dict) -> Dict[str, pd.DataFrame]:
    """Collect per-model JSONs into the three required CSV tables + plots."""
    class_names = [SEM_ID_TO_NAME[i] for i in range(NUM_CLASSES)]

    semantic_rows: List[Dict] = []
    per_class_rows: Dict[str, List[float]] = {}
    instance_rows: List[Dict] = []
    efficiency_rows: List[Dict] = []

    for name, mcfg in cfg["models"].items():
        if not mcfg.get("enabled", False): continue
        mdir = out_root / "results" / name
        met_path = mdir / "metrics.json"
        eff_path = mdir / "efficiency.json"
        if not met_path.exists() or not eff_path.exists():
            print(f"skipping {name}: results not found"); continue
        met = json.loads(met_path.read_text())
        eff = json.loads(eff_path.read_text())

        efficiency_rows.append(eff)
        if mcfg["task"] == "semantic" or name == "kmeans":
            semantic_rows.append({
                "Model": name,
                "Pixel_Acc":      met.get("pixel_accuracy", float("nan")),
                "mIoU":           met.get("mean_iou",       float("nan")),
                "Dice":           met.get("mean_dice",      float("nan")),
                "Precision":      met.get("pixel_precision",float("nan")),
                "Recall":         met.get("pixel_recall",   float("nan")),
                "Boundary_F1":    met.get("boundary_f1_mean", float("nan")),
                "Parameters":     eff.get("total_params",   0),
                "Size_MB":        eff.get("checkpoint_size_mb", 0.0),
                "Throughput_ips": eff.get("throughput_images_per_s", 0.0),
            })
            if "per_class_iou" in met:
                per_class_rows[name] = met["per_class_iou"]
        else:
            instance_rows.append({
                "Model": name,
                "Mask_AP":   met.get("mask_AP",   float("nan")),
                "AP50":      met.get("mask_AP50", float("nan")),
                "AP75":      met.get("mask_AP75", float("nan")),
                "Mask_AR":   met.get("mask_AR",   float("nan")),
                "Box_AP":    met.get("box_AP",    float("nan")),
                "Parameters":     eff.get("total_params",   0),
                "Size_MB":        eff.get("checkpoint_size_mb", 0.0),
                "Latency_ms":     eff.get("latency_ms_per_image", 0.0),
                "Throughput_ips": eff.get("throughput_images_per_s", 0.0),
            })

    sem_df = pd.DataFrame(semantic_rows)
    inst_df = pd.DataFrame(instance_rows)
    eff_df  = pd.DataFrame(efficiency_rows)

    results_dir = out_root / "results"
    plots_dir   = out_root / "plots"
    results_dir.mkdir(parents=True, exist_ok=True); plots_dir.mkdir(parents=True, exist_ok=True)
    sem_df.to_csv(results_dir / "semantic_segmentation_results.csv", index=False)
    inst_df.to_csv(results_dir / "instance_segmentation_results.csv", index=False)
    eff_df.to_csv(results_dir / "segmentation_efficiency_results.csv", index=False)

    # ---- per-class IoU CSV ---- #
    if per_class_rows:
        pcls_df = pd.DataFrame(per_class_rows, index=class_names)
        pcls_df.to_csv(results_dir / "per_class_iou.csv")
        plot_per_class_iou(per_class_rows, class_names, plots_dir)

    # ---- required plots ---- #
    if not sem_df.empty: plot_semantic_bars(sem_df, plots_dir)
    if not eff_df.empty: plot_efficiency_bars(eff_df, plots_dir)
    if not sem_df.empty and not eff_df.empty:
        plot_tradeoffs(sem_df, eff_df, plots_dir)
    if not inst_df.empty: plot_instance_ap(inst_df, plots_dir)

    return {"semantic": sem_df, "instance": inst_df, "efficiency": eff_df}


# ------------------------------- Entry point ------------------------------- #

def run_benchmark(config_path: Path, task: str = "all", model: str = "all",
                  out_root: Optional[Path] = None) -> None:
    cfg = load_config(config_path)
    out_root = out_root or Path(".")
    seed_everything(cfg["experiment"]["seed"])

    # Choose which models to run
    all_models = list(cfg["models"].keys())
    if model != "all":
        all_models = [model]
    for m in all_models:
        mcfg = cfg["models"][m]
        if not mcfg.get("enabled", False): continue
        if task != "all" and mcfg["task"] != task and not (task == "semantic" and m == "kmeans"):
            continue

        print(f"\n############ {m} ({mcfg['task']}) ############")
        if m == "kmeans":
            run_traditional_kmeans(cfg, out_root)
        elif m == "maskrcnn":
            run_maskrcnn(cfg, out_root)
        elif m == "yolo_seg":
            run_yolo_seg(cfg, out_root)
        else:
            run_semantic_model(m, cfg, out_root)
        gc.collect()
        if _HAS_TORCH and torch.cuda.is_available(): torch.cuda.empty_cache()

    aggregate_results(out_root, cfg)
    print("\nBenchmark complete. See results/, plots/, predictions/, checkpoints/, logs/.")
