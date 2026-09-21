#!/usr/bin/env python3
"""Build the assignment's COCO 2017 subset from the official downloads.

Steps
-----
1. Download instances_{train,val}2017.json and the corresponding image sets.
2. Keep only images that contain at least one instance of {person, car,
   bicycle, dog, cat}.
3. Produce SEED=42 splits with target 5000/1000/1000 (or the caller's counts).
4. Copy the kept images to data/coco/images/.
5. Save annotations/per_image_annotations.json (per-image raw COCO anns for the
   five foreground classes), image_info.json, split_manifest.json.
6. Optionally emit a YOLO segmentation dataset YAML at annotations/yolo_seg.yaml
   plus label files under data/coco/labels/{split}/.

Usage
-----
    python scripts/prepare_coco_subset.py                          # full run
    python scripts/prepare_coco_subset.py --train 500 --val 100 --test 100
    python scripts/prepare_coco_subset.py --yolo-yaml               # also emit YOLO labels
    python scripts/prepare_coco_subset.py --coco-root ~/coco2017    # skip downloading

The script is resumable: existing files are not re-downloaded.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mask_utils import COCO_TO_SEM, SEM_ID_TO_NAME, NUM_CLASSES

COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMG_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017":   "http://images.cocodataset.org/zips/val2017.zip",
}
SELECTED_COCO_IDS: Set[int] = set(COCO_TO_SEM.keys())


def download(url: str, dst: Path):
    if dst.exists(): return
    import urllib.request
    print(f"downloading {url} -> {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dst)


def unzip(zpath: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(out_dir)


def ensure_coco(coco_root: Path):
    ann_dir = coco_root / "annotations"
    if not (ann_dir / "instances_train2017.json").exists():
        z = coco_root / "_cache" / "annotations_trainval2017.zip"
        download(COCO_ANN_URL, z); unzip(z, coco_root)
    for split, url in COCO_IMG_URLS.items():
        img_dir = coco_root / split
        if not img_dir.exists() or not any(img_dir.iterdir()):
            z = coco_root / "_cache" / f"{split}.zip"
            download(url, z); unzip(z, coco_root)


def load_coco(coco_root: Path, split: str) -> Dict:
    with open(coco_root / "annotations" / f"instances_{split}.json") as f:
        return json.load(f)


def build_subset(coco_root: Path, out_root: Path, targets: Dict[str, int],
                 seed: int = 42) -> Dict:
    random.seed(seed); np.random.seed(seed)

    per_image_anns: Dict[int, List[dict]] = {}
    image_info: Dict[int, Dict] = {}

    for src_split in ("train2017", "val2017"):
        data = load_coco(coco_root, src_split)
        anns_by_img: Dict[int, List[dict]] = {}
        for a in data["annotations"]:
            if a["category_id"] in SELECTED_COCO_IDS:
                anns_by_img.setdefault(a["image_id"], []).append(a)
        info_map = {im["id"]: im for im in data["images"]}
        for img_id, alist in anns_by_img.items():
            im = info_map[img_id]
            image_info[img_id] = {"height": im["height"], "width": im["width"],
                                   "file_name": im["file_name"], "src_split": src_split}
            per_image_anns[img_id] = alist

    ids = sorted(per_image_anns.keys())
    random.shuffle(ids)
    total_needed = sum(targets.values())
    if len(ids) < total_needed:
        print(f"WARNING: only {len(ids)} images match the 5-class filter; "
              f"reducing targets proportionally.")
        scale = len(ids) / total_needed
        targets = {k: int(v * scale) for k, v in targets.items()}

    splits = {
        "train": ids[:targets["train"]],
        "val":   ids[targets["train"]:targets["train"] + targets["val"]],
        "test":  ids[targets["train"] + targets["val"]:
                     targets["train"] + targets["val"] + targets["test"]],
    }

    # ---- Copy images ---- #
    img_out = out_root / "data" / "coco" / "images"; img_out.mkdir(parents=True, exist_ok=True)
    kept_ids = set().union(*splits.values())
    for img_id in kept_ids:
        info = image_info[img_id]
        src = coco_root / info["src_split"] / info["file_name"]
        dst = img_out / f"{img_id:012d}.jpg"
        if not dst.exists() and src.exists():
            shutil.copy(src, dst)

    # ---- Save annotations ---- #
    ann_out = out_root / "annotations"; ann_out.mkdir(parents=True, exist_ok=True)
    (ann_out / "split_manifest.json").write_text(
        json.dumps({k: sorted(map(int, v)) for k, v in splits.items()}, indent=2))
    (ann_out / "per_image_annotations.json").write_text(
        json.dumps({str(k): v for k, v in per_image_anns.items() if k in kept_ids}))
    (ann_out / "image_info.json").write_text(
        json.dumps({str(k): v for k, v in image_info.items() if k in kept_ids}))

    print(f"subset built: train={len(splits['train'])}, val={len(splits['val'])}, "
          f"test={len(splits['test'])}")
    return {"splits": splits, "image_info": image_info, "per_image_anns": per_image_anns}


def build_yolo_labels(out_root: Path, subset: Dict):
    """Emit YOLO segmentation labels in polygon-per-line format."""
    labels_root = out_root / "data" / "coco" / "labels"
    yolo_id_of_sem = {sem_id: idx for idx, sem_id in enumerate(sorted(set(v[1] for v in COCO_TO_SEM.values())))}
    coco_to_yolo = {coco_id: yolo_id_of_sem[sem_id] for coco_id, (_, sem_id) in COCO_TO_SEM.items()}

    for split, ids in subset["splits"].items():
        labels_dir = labels_root / split; labels_dir.mkdir(parents=True, exist_ok=True)
        for img_id in ids:
            info = subset["image_info"][img_id]
            W, H = info["width"], info["height"]
            anns = subset["per_image_anns"][img_id]
            lines: List[str] = []
            for a in anns:
                cls = coco_to_yolo[a["category_id"]]
                seg = a["segmentation"]
                if isinstance(seg, list):
                    for poly in seg:
                        norm = [f"{poly[i]/W:.6f}" if i % 2 == 0 else f"{poly[i]/H:.6f}"
                                for i in range(len(poly))]
                        lines.append(" ".join([str(cls)] + norm))
            (labels_dir / f"{img_id:012d}.txt").write_text("\n".join(lines))

    yaml_path = out_root / "annotations" / "yolo_seg.yaml"
    class_list = [SEM_ID_TO_NAME[sem_id] for sem_id in sorted(set(v[1] for v in COCO_TO_SEM.values()))]
    yaml_path.write_text(
        f"path: {out_root.resolve()}\n"
        f"train: data/coco/images\n"
        f"val:   data/coco/images\n"
        f"test:  data/coco/images\n"
        f"names:\n" + "\n".join(f"  {i}: {n}" for i, n in enumerate(class_list)) + "\n"
    )
    print(f"YOLO dataset YAML: {yaml_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--coco-root", type=Path, default=Path("./coco2017"))
    p.add_argument("--out-root",  type=Path, default=Path("."))
    p.add_argument("--train", type=int, default=5000)
    p.add_argument("--val",   type=int, default=1000)
    p.add_argument("--test",  type=int, default=1000)
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--yolo-yaml", action="store_true",
                   help="also emit YOLO segmentation labels + dataset YAML")
    p.add_argument("--skip-download", action="store_true",
                   help="assume --coco-root already contains train2017/, val2017/, annotations/")
    args = p.parse_args()

    if not args.skip_download:
        ensure_coco(args.coco_root)
    subset = build_subset(args.coco_root, args.out_root,
                          targets={"train": args.train, "val": args.val, "test": args.test},
                          seed=args.seed)
    if args.yolo_yaml:
        build_yolo_labels(args.out_root, subset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
