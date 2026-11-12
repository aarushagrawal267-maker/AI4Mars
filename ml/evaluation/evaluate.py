"""Evaluate a trained model on held-out labelled data.

Predictions are made at the model's training resolution, the class probabilities
are upsampled to the label's native resolution, and the comparison is done there
against the original (un-resized) annotation. Unlabelled pixels (255) are ignored.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from ..classes import NUM_CLASSES
from ..datasets.cache import read_gray, read_label
from ..preprocessing.transforms import prepare_inference_input
from .metrics import confusion_matrix, metrics_from_confusion


def records_for(index: dict, split: str, variant: str | None) -> tuple[list[dict], str]:
    """Records and label key for a split. ``test`` needs an agreement variant (e.g. ``min3``)."""
    if split not in ("test", "val"):
        raise ValueError("Evaluation runs on the 'test' or 'val' split only (never on training data)")
    recs = [r for r in index["records"] if r["split"] == split]
    if split == "val":
        return recs, "train"
    variants = sorted({v for r in recs for v in r["labels"]})
    if not variants:
        raise ValueError("The dataset index contains no test labels")
    if variant is None:
        variant = "min3" if "min3" in variants else variants[-1]
    if variant not in variants:
        raise ValueError(f"Unknown test label variant {variant!r}; available: {variants}")
    return [r for r in recs if variant in r["labels"]], variant


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, resolution: int, index: dict, raw_root: Path, split: str = "test",
                   variant: str | None = None, device: torch.device | str = "cpu", max_images: int | None = None,
                   progress: Callable[[float, str], None] | None = None,
                   should_stop: Callable[[], bool] | None = None) -> dict:
    raw_root = Path(raw_root)
    device = torch.device(device)
    recs, label_key = records_for(index, split, variant)
    if max_images is not None:
        recs = recs[:max_images]
    if not recs:
        raise ValueError(f"No labelled images available for split {split!r}")
    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    forward_ms: list[float] = []
    total_ms: list[float] = []
    for i, r in enumerate(recs):
        if should_stop and should_stop():
            raise InterruptedError("Evaluation cancelled")
        gray = read_gray(raw_root / r["image"])
        label = read_label(raw_root / r["labels"][label_key]["path"])
        t0 = time.perf_counter()
        tensor, _ = prepare_inference_input(gray, resolution)
        tensor = tensor.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        logits = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        probs = F.interpolate(F.softmax(logits.float(), dim=1), size=label.shape, mode="bilinear", align_corners=False)
        pred = probs.argmax(1)[0].cpu().numpy()
        t3 = time.perf_counter()
        forward_ms.append((t2 - t1) * 1000)
        total_ms.append((t3 - t0) * 1000)
        cm += confusion_matrix(pred, label)
        if progress and (i % 5 == 0 or i == len(recs) - 1):
            progress((i + 1) / len(recs), f"Evaluated {i + 1}/{len(recs)} images")
    metrics = metrics_from_confusion(cm)
    # The first image includes one-off warm-up cost, so it is left out of the timing statistics.
    f = np.asarray(forward_ms[1:] or forward_ms)
    t = np.asarray(total_ms[1:] or total_ms)
    metrics.update({
        "split": split,
        "label_variant": label_key if split == "test" else "train-merged",
        "images": len(recs),
        "resolution": resolution,
        "evaluated_at_native_label_resolution": True,
        "inference": {
            "device": str(device),
            "forward_ms_mean": float(f.mean()), "forward_ms_median": float(np.median(f)),
            "forward_ms_p95": float(np.percentile(f, 95)),
            "end_to_end_ms_mean": float(t.mean()),
            "images_per_second": float(1000.0 / f.mean()) if f.mean() > 0 else None,
            "timed_images": int(len(f)),
            "note": "Single-image forward pass at the model's input resolution on the listed device; "
                    "end-to-end adds preprocessing and upsampling to native resolution.",
        },
    })
    return metrics
