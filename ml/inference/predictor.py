"""Run a trained model on one image and return class map, confidence and uncertainty."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..classes import CLASS_KEYS, NUM_CLASSES
from ..preprocessing.transforms import prepare_inference_input


def output_size(height: int, width: int, max_side: int) -> tuple[int, int]:
    """Output resolution: the original size, capped so the long side is at most ``max_side``."""
    scale = min(1.0, max_side / max(height, width))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


@torch.no_grad()
def predict(model: torch.nn.Module, resolution: int, rgb: np.ndarray, device: torch.device | str = "cpu",
            max_output_side: int = 1536) -> dict:
    """Segment one RGB (H, W, 3) or grayscale (H, W) uint8 image.

    Returns
    -------
    dict with
      ``class_map``    (h, w) uint8, values 0-3
      ``confidence``   (h, w) float32, max softmax probability of the predicted class
      ``uncertainty``  (h, w) float32, normalised predictive entropy in [0, 1]
      ``mean_probs``   mean softmax probability per class over the image
      ``input_size`` / ``output_size`` / ``inference_ms``

    Confidence is the model's own softmax score. It is not calibrated and is not
    a measurement of accuracy: a confident prediction can be wrong.
    """
    model.eval()
    tensor, gray = prepare_inference_input(rgb, resolution)
    tensor = tensor.to(device)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = model(tensor)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - t0) * 1000.0
    if logits.shape[1] != NUM_CLASSES:
        raise RuntimeError(f"Model produced {logits.shape[1]} channels, expected {NUM_CLASSES}")
    oh, ow = output_size(gray.shape[0], gray.shape[1], max_output_side)
    probs = F.softmax(logits.float(), dim=1)
    probs = F.interpolate(probs, size=(oh, ow), mode="bilinear", align_corners=False)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    conf, cls = probs.max(dim=1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1) / math.log(NUM_CLASSES)
    return {
        "class_map": cls[0].to(torch.uint8).cpu().numpy(),
        "confidence": conf[0].cpu().numpy().astype(np.float32),
        "uncertainty": entropy[0].clamp(0, 1).cpu().numpy().astype(np.float32),
        "mean_probs": {k: float(probs[0, i].mean()) for i, k in enumerate(CLASS_KEYS)},
        "input_size": [int(tensor.shape[2]), int(tensor.shape[3])],
        "output_size": [oh, ow],
        "original_size": [int(gray.shape[0]), int(gray.shape[1])],
        "inference_ms": inference_ms,
    }
