"""Rendering helpers: colour-coded masks, overlays, heat maps and side-by-side panels."""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from ..classes import palette_array


def colorize(class_map: np.ndarray, colors: dict[str, str] | None = None) -> np.ndarray:
    """(H, W) class ids (255 = unlabelled) -> (H, W, 3) RGB using the display palette."""
    return palette_array(colors)[class_map]


def to_rgb(image: np.ndarray) -> np.ndarray:
    return np.repeat(image[:, :, None], 3, axis=2) if image.ndim == 2 else image


def overlay(image: np.ndarray, class_map: np.ndarray, colors: dict[str, str] | None = None,
            alpha: float = 0.5) -> np.ndarray:
    """Blend the colour mask over the image. Unlabelled pixels (255) are left untouched."""
    base = to_rgb(image).astype(np.float32)
    mask = colorize(class_map, colors).astype(np.float32)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = base * (1 - alpha) + mask * alpha
    keep = class_map == 255
    out[keep] = base[keep]
    return np.clip(out, 0, 255).astype(np.uint8)


def heatmap(values: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Float map in [0, 1] -> RGB heat map."""
    v = np.clip(values, 0.0, 1.0)
    bgr = cv2.applyColorMap((v * 255).astype(np.uint8), colormap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def side_by_side(*panels: np.ndarray, gap: int = 8, background: tuple[int, int, int] = (17, 19, 23)) -> np.ndarray:
    """Place equally sized RGB panels next to each other."""
    panels = tuple(to_rgb(p) for p in panels)
    h = panels[0].shape[0]
    w = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = background
    x = 0
    for p in panels:
        canvas[:, x:x + p.shape[1]] = p
        x += p.shape[1] + gap
    return canvas


def png_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG", optimize=False, compress_level=3)
    return buf.getvalue()
