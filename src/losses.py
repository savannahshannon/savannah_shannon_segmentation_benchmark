"""Semantic loss = 0.5 * CE + 0.5 * Dice.

Both terms respect ignore_index (crowd / non-labeled). Dice is macro-averaged
across foreground classes; background is included optionally.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    raise ImportError("PyTorch is required for training losses.") from e


class DiceLoss(nn.Module):
    """Soft Dice, averaged over classes."""
    def __init__(self, num_classes: int, ignore_index: int = 255,
                 include_background: bool = True, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.include_background = include_background
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W), targets: (B, H, W) long
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)
        # Build one-hot ignoring ignore_index
        valid = (targets != self.ignore_index)
        tgt = targets.clone()
        tgt[~valid] = 0
        one_hot = F.one_hot(tgt, num_classes=C).permute(0, 3, 1, 2).float()
        valid_f = valid.unsqueeze(1).float()

        dims = (0, 2, 3)
        inter = (probs * one_hot * valid_f).sum(dim=dims)
        denom = (probs * valid_f).sum(dim=dims) + (one_hot * valid_f).sum(dim=dims)
        dice  = (2 * inter + self.eps) / (denom + self.eps)
        if not self.include_background:
            dice = dice[1:]
        return 1.0 - dice.mean()


class CombinedSegLoss(nn.Module):
    """Weighted CE + Dice (default 0.5 / 0.5)."""
    def __init__(self, num_classes: int, ce_weight: float = 0.5, dice_weight: float = 0.5,
                 ignore_index: int = 255, class_weights=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, logits, targets):
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)
