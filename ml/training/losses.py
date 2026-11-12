"""Loss functions: cross-entropy with ignore-index, optionally combined with soft Dice."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..classes import IGNORE_INDEX, NUM_CLASSES


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = IGNORE_INDEX,
                   eps: float = 1.0) -> torch.Tensor:
    """Multi-class soft Dice loss over valid (labelled) pixels, averaged over classes present in the batch."""
    num_classes = logits.shape[1]
    valid = (target != ignore_index)
    if not valid.any():
        return logits.sum() * 0.0
    probs = F.softmax(logits, dim=1)
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    onehot = F.one_hot(safe_target, num_classes).permute(0, 3, 1, 2).to(probs.dtype)
    mask = valid.unsqueeze(1).to(probs.dtype)
    probs, onehot = probs * mask, onehot * mask
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    present = onehot.sum(dims) > 0
    if not present.any():
        return logits.sum() * 0.0
    return 1.0 - dice[present].mean()


class SegmentationLoss(nn.Module):
    """``CE(ignore_index) + dice_weight * Dice``. Unlabelled pixels contribute nothing to either term."""

    def __init__(self, dice_weight: float = 0.0, class_weights: torch.Tensor | None = None,
                 ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.ignore_index = ignore_index
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not (target != self.ignore_index).any():
            # A batch with no labelled pixels has no supervised signal: return a zero that keeps the graph.
            return logits.sum() * 0.0
        loss = F.cross_entropy(logits, target, weight=self.class_weights, ignore_index=self.ignore_index)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * soft_dice_loss(logits, target, self.ignore_index)
        return loss


def class_weights_from_counts(counts: list[int], mode: str = "none") -> torch.Tensor | None:
    """Optional class re-weighting for the heavy AI4Mars imbalance (big rock is <1 % of pixels).

    ``inverse_sqrt``: w_c proportional to 1/sqrt(frequency_c), normalised to mean 1 and clipped to [0.25, 8].
    """
    if mode == "none":
        return None
    if mode != "inverse_sqrt":
        raise ValueError("class weighting must be 'none' or 'inverse_sqrt'")
    c = torch.tensor(counts, dtype=torch.float64)
    if c.sum() <= 0:
        return None
    freq = (c / c.sum()).clamp_min(1e-6)
    w = 1.0 / freq.sqrt()
    w = w / w.mean()
    assert len(w) == NUM_CLASSES
    return w.clamp(0.25, 8.0).float()
