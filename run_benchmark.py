#!/usr/bin/env python3
"""Top-level CLI for the Segmentation Benchmark.

Examples
--------
Train and evaluate a single semantic model:
    python run_benchmark.py --task semantic --model unet

Run the whole semantic track:
    python run_benchmark.py --task semantic --model all

Run instance segmentation (Mask R-CNN + YOLO):
    python run_benchmark.py --task instance --model all

Full benchmark (all 10 models):
    python run_benchmark.py --task all --model all

Configuration lives in `configuration.yaml`; override with --config.
Outputs land under results/, plots/, predictions/, checkpoints/, logs/,
confusion_matrices/. Use --smoke to run with dataset.smoke_test settings.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.benchmark import run_benchmark, load_config, aggregate_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["semantic", "instance", "all", "traditional"],
                        default="all", help="which benchmark track to run")
    parser.add_argument("--model", default="all",
                        help="model name (unet, deeplabv3, maskrcnn, yolo_seg, ...) or 'all'")
    parser.add_argument("--config", type=Path, default=Path("configuration.yaml"))
    parser.add_argument("--out", type=Path, default=Path("."),
                        help="root directory for results/ plots/ predictions/ logs/")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="rebuild CSVs and plots from cached per-model JSONs without retraining")
    parser.add_argument("--smoke", action="store_true",
                        help="use dataset.smoke_test sizes and 3 epochs (development)")
    args = parser.parse_args()

    if args.smoke:
        cfg = load_config(args.config)
        cfg["dataset"]["smoke_test"]["enabled"] = True
        cfg["training"]["epochs"] = 3
        # Write a temp copy so run_benchmark.py picks it up
        import yaml, tempfile
        tf = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.dump(cfg, tf); tf.close()
        args.config = Path(tf.name)

    if args.aggregate_only:
        cfg = load_config(args.config)
        aggregate_results(args.out, cfg)
        return 0

    run_benchmark(args.config, task=args.task, model=args.model, out_root=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
