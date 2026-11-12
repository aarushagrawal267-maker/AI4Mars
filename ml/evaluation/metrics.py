"""Segmentation metrics computed from a confusion matrix.

Every metric ignores pixels whose ground-truth value is ``ignore_index`` (255,
unlabelled). Rows of the confusion matrix are ground truth, columns predictions.
A class that is absent from both ground truth and predictions has an undefined
score, reported as ``None`` and excluded from the mean instead of counted as 0 or 1.
"""
from __future__ import annotations

import numpy as np
import torch

from ..classes import CLASS_KEYS, IGNORE_INDEX, NUM_CLASSES


def confusion_matrix(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray,
                     num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    """(num_classes, num_classes) int64 matrix; rows = ground truth, cols = prediction."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    pred = np.asarray(pred).reshape(-1).astype(np.int64)
    target = np.asarray(target).reshape(-1).astype(np.int64)
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same number of pixels")
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    idx = target[valid] * num_classes + pred[valid]
    return np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def _safe_div(num: np.ndarray, den: np.ndarray) -> list[float | None]:
    return [float(n / d) if d > 0 else None for n, d in zip(num, den)]


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def metrics_from_confusion(cm: np.ndarray, class_keys: list[str] = CLASS_KEYS) -> dict:
    """Pixel accuracy, per-class IoU / Dice / precision / recall and their means."""
    cm = np.asarray(cm, dtype=np.int64)
    tp = np.diag(cm).astype(np.float64)
    gt = cm.sum(axis=1).astype(np.float64)       # pixels whose true class is c
    pd = cm.sum(axis=0).astype(np.float64)       # pixels predicted as c
    total = float(cm.sum())
    iou = _safe_div(tp, gt + pd - tp)
    dice = _safe_div(2 * tp, gt + pd)
    precision = _safe_div(tp, pd)
    recall = _safe_div(tp, gt)
    # Mean IoU / Dice are averaged over classes that occur in the ground truth, the usual
    # convention: a class the evaluation set never contains cannot be scored.
    present = gt > 0
    miou = _mean([v for v, p in zip(iou, present) if p])
    mdice = _mean([v for v, p in zip(dice, present) if p])
    return {
        "pixel_accuracy": float(tp.sum() / total) if total > 0 else None,
        "mean_iou": miou,
        "mean_dice": mdice,
        "mean_precision": _mean([v for v, p in zip(precision, present) if p]),
        "mean_recall": _mean([v for v, p in zip(recall, present) if p]),
        "per_class": {
            key: {
                "iou": iou[i], "dice": dice[i], "precision": precision[i], "recall": recall[i],
                "support_pixels": int(gt[i]), "predicted_pixels": int(pd[i]),
            } for i, key in enumerate(class_keys)
        },
        "confusion_matrix": cm.tolist(),
        "evaluated_pixels": int(total),
        "classes_present": [class_keys[i] for i in range(len(class_keys)) if present[i]],
    }
