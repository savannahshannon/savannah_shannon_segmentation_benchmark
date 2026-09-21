# Comprehensive Image Segmentation Benchmarking

**Author:** Savannah Shannon — Ph.D. student, Clark Atlanta University
**Course:** Computer Science, AI concentration

This repository implements the assignment's full benchmark of 10 image
segmentation architectures on a common COCO 2017 subset. It contains a
reusable framework (`src/`), a data-preparation script (`scripts/`),
per-model outputs (`predictions/`, `checkpoints/`, `logs/`,
`confusion_matrices/`, `results/`), aggregate tables (`results/`), and every
figure required by the assignment specification (`plots/`).

## What is compared

| # | Model                       | Track       |
|---|-----------------------------|-------------|
| 1 | K-Means color segmentation  | Traditional |
| 2 | FCN-ResNet50                | Semantic    |
| 3 | U-Net                       | Semantic    |
| 4 | U-Net++                     | Semantic    |
| 5 | SegNet                      | Semantic    |
| 6 | DeepLabV3-ResNet50          | Semantic    |
| 7 | PSPNet                      | Semantic    |
| 8 | Mask R-CNN                  | Instance    |
| 9 | YOLOv8 Segmentation         | Instance    |
| 10| SegFormer-B0                | Semantic    |

## Directory layout

```
Savannah_Shannon_Segmentation_Benchmark/
├── data/coco/images/                     # Prepared COCO 2017 subset (jpg)
├── annotations/                          # split_manifest.json, per_image_annotations.json,
│                                         # image_info.json, yolo_seg.yaml
├── masks/semantic/                       # Cached class-ID masks (8-bit PNG)
├── checkpoints/                          # best_<model>.pt (semantic models)
│                                         # yolo_seg/ (Ultralytics runs)
├── confusion_matrices/<model>.png        # per-model confusion matrix heatmaps
├── logs/<model>.log                      # per-model training/eval logs
├── predictions/<model>/                  # sample predicted masks (PNG)
├── results/                              # Auto-generated tables
│   ├── semantic_segmentation_results.csv
│   ├── instance_segmentation_results.csv
│   ├── segmentation_efficiency_results.csv
│   ├── per_class_iou.csv
│   └── <model>/{metrics.json, efficiency.json, history.json}
├── plots/                                # All 10 required figures (Figures 1–10)
├── scripts/prepare_coco_subset.py        # Downloads COCO 2017 and builds the subset
├── scripts/generate_reference_results.py # Populates literature-based reference values
├── src/
│   ├── dataset.py    # SemanticSegDataset, InstanceSegDataset, mask-safe transforms
│   ├── mask_utils.py # polygon/RLE → mask, semantic assembly, ignore-index policy
│   ├── models.py     # Model factory covering all 10 architectures
│   ├── losses.py     # 0.5·CE + 0.5·Dice combined loss
│   ├── metrics.py    # ConfusionMatrix, boundary F1, mask AP via pycocotools
│   ├── train.py      # semantic + Mask R-CNN trainers
│   ├── evaluate.py   # test-set evaluators and efficiency probes
│   ├── visualize.py  # every required plot (Figures 1–10)
│   └── benchmark.py  # orchestration, CSV aggregation, plot generation
├── run_benchmark.py                      # CLI: --task {semantic,instance,all} --model <name>
├── configuration.yaml                    # single source of truth for seeds, sizes, hyper-params
├── requirements.txt
├── notebooks/colab_full_benchmark.ipynb  # ready-to-run Colab GPU notebook
└── README.md
```

## Quick start (Colab or workstation with NVIDIA GPU)

### 1. Install
```
pip install -r requirements.txt
```

### 2. Prepare the dataset (SEED=42)
```
python scripts/prepare_coco_subset.py --train 5000 --val 1000 --test 1000 --yolo-yaml
```
This downloads COCO 2017, keeps only images containing at least one of
{person, car, bicycle, dog, cat}, produces the split_manifest.json (used by
*every* model), caches per-image annotations, and emits a YOLO segmentation
dataset YAML for the Ultralytics trainer. **The same manifest is loaded by all
models — no per-architecture splits.**

### 3. Run the benchmark
```
# Everything
python run_benchmark.py --task all --model all

# Single semantic model
python run_benchmark.py --task semantic --model deeplabv3

# Instance segmentation only
python run_benchmark.py --task instance --model all

# Only regenerate the aggregated CSVs and plots after editing per-model JSONs
python run_benchmark.py --aggregate-only
```

Outputs land in `results/`, `plots/`, `predictions/`, `checkpoints/`,
`confusion_matrices/`, `logs/`. The three headline CSVs are named
`semantic_segmentation_results.csv`, `instance_segmentation_results.csv`, and
`segmentation_efficiency_results.csv` exactly as required by §27–§29.

### 4. Reference-values mode (no training)
For quick previews of table/plot layout without launching a training run:
```
python scripts/generate_reference_results.py
```
This writes literature-grounded placeholder values to `results/<model>/*.json`
and regenerates the CSVs + all 10 figures. **Overwrite these by running the
real training** in step 3.

## Experimental protocol (§13)

| Setting | Value |
|---------|-------|
| Random seed | 42 (numpy, torch, python) |
| Input resolution | 512 × 512 (256 × 256 for compute-limited runs) |
| Training epochs | 25 (semantic + Mask R-CNN); 25 for YOLO |
| Batch size | 8 |
| Optimizer / LR | AdamW / 1e-4 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Mixed precision | AMP on CUDA |
| Semantic loss | 0.5·CE + 0.5·Dice (both ignore_index-aware) |
| Ignore index | 255 (crowd regions of the 5 classes) |
| Checkpoint metric | val mIoU (semantic); Mask R-CNN uses its own val loss |

Dataset policy (§8):
* One consistent policy across every model.
* Selected classes: `background, person, car, bicycle, dog, cat` (IDs 0–5).
* Overlaps between two selected classes: smaller-area instance wins
  (preserves small objects).
* `iscrowd=1` regions of the selected classes: assigned `ignore_index=255`
  and excluded from both loss and test evaluation.
* Non-selected COCO classes: treated as background.
* Split manifest under `annotations/split_manifest.json` — every model loads
  the same file.

## Metrics reported

Semantic (§22):
* Pixel accuracy (over labeled pixels)
* Per-class IoU (background reported separately + in mIoU by default)
* Mean IoU
* Macro Dice
* Pixel-level precision and recall
* Boundary precision / recall / F1 at 2-pixel tolerance (Csurka et al. 2013)

Instance (§25, COCO-style via pycocotools):
* Mask AP averaged over IoU 0.50:0.95:0.05 (primary)
* Mask AP50, Mask AP75
* Mask AR (max detections = 100)
* Box AP (secondary)

Efficiency (§29):
* Total & trainable parameters
* Weight-only checkpoint size (MB)
* Total training time (s), per-epoch time (s)
* Latency (ms/image, batch=1) & throughput (images/s) after 10 warm-up iters
* Peak GPU memory (MB) when running on CUDA
* Input resolution, batch size, precision, hardware string — logged with the
  measurements so numbers stay comparable across environments

## Interpreting the reference outputs shipped with this repo

The version of `results/` and `plots/` checked into the repo was produced by
`scripts/generate_reference_results.py` using literature-grounded placeholder
values (see the header of that script for citations). This lets reviewers
see the report scaffold, the CSV formats, and the figure layout at a glance.
On any GPU-equipped machine, `python run_benchmark.py --task all --model all`
overwrites every per-model JSON with actual measurements; running
`python run_benchmark.py --aggregate-only` afterward rebuilds the CSVs and
figures from the fresh JSONs.

## Colab notebook

`notebooks/colab_full_benchmark.ipynb` is a self-contained notebook that
installs dependencies, downloads a subset, and calls the same CLI end-to-end
on Colab's T4 GPU. Recommended when a local GPU is not available.

## References

* Long et al. 2015 — Fully Convolutional Networks
* Ronneberger et al. 2015 — U-Net
* Badrinarayanan et al. 2017 — SegNet
* Zhou et al. 2018 — UNet++
* Chen et al. 2017 — DeepLabV3, dilated convolutions & ASPP
* Zhao et al. 2017 — PSPNet, pyramid pooling
* He et al. 2017 — Mask R-CNN, RoI Align
* Xie et al. 2021 — SegFormer
* Ultralytics YOLOv8-seg
* Csurka, Larlus, Perronnin 2013 — Boundary F1 measure
* Lin et al. 2014 — Microsoft COCO
