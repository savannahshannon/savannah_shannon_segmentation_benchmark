"""Segmentation Benchmark source package.

Modules
-------
dataset      : COCO subset loading, mask-safe transforms, semantic & instance adapters
mask_utils   : polygon/RLE -> mask conversion, semantic mask assembly, ignore-index handling
models       : model factory covering all 10 required architectures
losses       : combined CE + Dice for semantic; wrappers for instance losses
metrics      : semantic (IoU, mIoU, Dice, pixel accuracy, boundary F1) and instance (mask AP) metrics
train        : reusable trainer with checkpointing and history logging
evaluate     : semantic and instance evaluators producing rows for CSV tables
visualize    : qualitative comparison figures, per-class IoU heat map, confusion matrices
benchmark    : orchestrates a full model list, produces the three required CSVs + all plots
"""
