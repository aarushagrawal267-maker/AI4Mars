"""Terrain analysis derived from a segmentation result.

Everything here is arithmetic on the predicted class map and the softmax
confidence/entropy maps: class percentages, the dominant class, where each class
sits in the frame, and how confident the model was. Nothing is estimated or
invented, and nothing here speaks to whether terrain is safe to drive on.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..classes import CLASS_KEYS, CLASS_LABELS, IGNORE_INDEX, NUM_CLASSES

ZONE_ROWS = ("upper", "middle", "lower")
ZONE_COLS = ("left", "centre", "right")


def _pct(n: float, d: float) -> float:
    return round(100.0 * n / d, 4) if d else 0.0


def terrain_statistics(class_map: np.ndarray, confidence: np.ndarray | None = None,
                       uncertainty: np.ndarray | None = None, bands: int = 12) -> dict:
    """Compute composition, spatial and confidence statistics for a class map.

    ``class_map`` may contain 255 for pixels without a class (ground-truth masks);
    those are excluded and counted separately.
    """
    h, w = class_map.shape
    valid = class_map != IGNORE_INDEX
    n_valid = int(valid.sum())
    counts = np.bincount(class_map[valid].ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    percentages = {k: _pct(int(counts[i]), n_valid) for i, k in enumerate(CLASS_KEYS)}
    dominant = CLASS_KEYS[int(np.argmax(counts))] if n_valid else None

    ys, xs = np.mgrid[0:h, 0:w]
    row_edges = np.linspace(0, h, 4).astype(int)
    col_edges = np.linspace(0, w, 4).astype(int)
    band_edges = np.linspace(0, h, bands + 1).astype(int)
    spatial: dict[str, dict] = {}
    for i, key in enumerate(CLASS_KEYS):
        m = class_map == i
        n = int(counts[i])
        entry: dict = {"pixels": n}
        if n:
            entry["centroid"] = [round(float(ys[m].mean() / h), 4), round(float(xs[m].mean() / w), 4)]  # (row, col) 0-1
            rr, cc = np.where(m.any(axis=1))[0], np.where(m.any(axis=0))[0]
            entry["bbox"] = [round(rr[0] / h, 4), round(cc[0] / w, 4), round((rr[-1] + 1) / h, 4),
                             round((cc[-1] + 1) / w, 4)]
            n_comp, _lab, stats, _cent = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
            areas = stats[1:, cv2.CC_STAT_AREA]
            entry["regions"] = int(n_comp - 1)
            entry["largest_region_percent_of_class"] = _pct(int(areas.max()), n)
            entry["largest_region_percent_of_image"] = _pct(int(areas.max()), n_valid)
        else:
            entry.update({"centroid": None, "bbox": None, "regions": 0,
                          "largest_region_percent_of_class": 0.0, "largest_region_percent_of_image": 0.0})
        # Share of this class's pixels that falls in each 3x3 zone of the frame.
        zones = {}
        for r, rname in enumerate(ZONE_ROWS):
            for c, cname in enumerate(ZONE_COLS):
                cell = m[row_edges[r]:row_edges[r + 1], col_edges[c]:col_edges[c + 1]]
                zones[f"{rname}_{cname}"] = _pct(int(cell.sum()), n)
        entry["zone_share_percent"] = zones
        spatial[key] = entry

    # Composition of each 3x3 zone, and of horizontal bands from top (far) to bottom (near).
    zone_composition = {}
    for r, rname in enumerate(ZONE_ROWS):
        for c, cname in enumerate(ZONE_COLS):
            cell = class_map[row_edges[r]:row_edges[r + 1], col_edges[c]:col_edges[c + 1]]
            cv = cell[cell != IGNORE_INDEX]
            cc = np.bincount(cv.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
            zone_composition[f"{rname}_{cname}"] = {
                "percent": {k: _pct(int(cc[i]), cv.size) for i, k in enumerate(CLASS_KEYS)},
                "dominant": CLASS_KEYS[int(np.argmax(cc))] if cv.size else None,
            }
    band_profile = []
    for b in range(bands):
        strip = class_map[band_edges[b]:band_edges[b + 1]]
        sv = strip[strip != IGNORE_INDEX]
        bc = np.bincount(sv.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
        band_profile.append({"band": b + 1, "from_top_percent": round(100 * band_edges[b] / h, 1),
                             **{k: _pct(int(bc[i]), sv.size) for i, k in enumerate(CLASS_KEYS)}})

    result: dict = {
        "height": h, "width": w,
        "valid_pixels": n_valid, "unlabeled_pixels": int(h * w - n_valid),
        "class_pixels": {k: int(counts[i]) for i, k in enumerate(CLASS_KEYS)},
        "class_percent": percentages,
        "dominant_class": dominant,
        "spatial": spatial,
        "zone_composition": zone_composition,
        "band_profile": band_profile,
    }

    if confidence is not None:
        conf_valid = confidence[valid]
        hist, edges = np.histogram(conf_valid, bins=10, range=(0.0, 1.0))
        result["confidence"] = {
            "mean": round(float(conf_valid.mean()), 6) if n_valid else None,
            "median": round(float(np.median(conf_valid)), 6) if n_valid else None,
            "min": round(float(conf_valid.min()), 6) if n_valid else None,
            "per_class_mean": {k: (round(float(confidence[class_map == i].mean()), 6) if counts[i] else None)
                               for i, k in enumerate(CLASS_KEYS)},
            "histogram": [{"from": round(float(edges[j]), 1), "to": round(float(edges[j + 1]), 1),
                           "pixels": int(hist[j]), "percent": _pct(int(hist[j]), n_valid)} for j in range(10)],
            "low_confidence_percent": _pct(int((conf_valid < 0.5).sum()), n_valid),
            "high_confidence_percent": _pct(int((conf_valid >= 0.9).sum()), n_valid),
            "calibrated": False,
            "note": "Softmax confidence of the predicted class. Not calibrated; it is not a measure of accuracy.",
        }
    if uncertainty is not None:
        unc_valid = uncertainty[valid]
        result["uncertainty"] = {
            "mean_entropy": round(float(unc_valid.mean()), 6) if n_valid else None,
            "high_uncertainty_percent": _pct(int((unc_valid > 0.5).sum()), n_valid),
            "per_class_mean_entropy": {k: (round(float(uncertainty[class_map == i].mean()), 6) if counts[i] else None)
                                       for i, k in enumerate(CLASS_KEYS)},
            "definition": "Normalised predictive entropy of the softmax distribution (0 = one class certain, "
                          "1 = all four classes equally likely).",
        }
    result["summary"] = describe(result)
    return result


def describe(stats: dict) -> str:
    """Plain-language description built only from the computed numbers."""
    if not stats["valid_pixels"] or stats["dominant_class"] is None:
        return "No classified pixels."
    label = dict(zip(CLASS_KEYS, CLASS_LABELS))
    pct = stats["class_percent"]
    ranked = sorted(CLASS_KEYS, key=lambda k: pct[k], reverse=True)
    parts = [f"{label[k].lower()} {pct[k]:.1f}%" for k in ranked if pct[k] >= 0.05]
    dom = stats["dominant_class"]
    text = f"The predicted terrain is dominated by {label[dom].lower()} ({pct[dom]:.1f}% of classified pixels). "
    text += "Composition: " + ", ".join(parts) + ". "
    absent = [label[k].lower() for k in CLASS_KEYS if pct[k] < 0.05]
    if absent:
        text += "Essentially absent (<0.05%): " + ", ".join(absent) + ". "
    zc = stats["spatial"][dom]["zone_share_percent"]
    top_zone = max(zc, key=zc.get)
    text += f"Most {label[dom].lower()} pixels lie in the {top_zone.replace('_', ' ')} part of the frame. "
    conf = stats.get("confidence")
    if conf and conf["mean"] is not None:
        text += (f"Mean softmax confidence is {conf['mean'] * 100:.1f}%, with {conf['low_confidence_percent']:.1f}% "
                 f"of pixels below 50% confidence. Confidence is uncalibrated and does not measure accuracy.")
    return text


def class_grid(class_map: np.ndarray, rows: int, cols: int, uncertainty: np.ndarray | None = None) -> dict:
    """Downsample a class map to a ``rows x cols`` grid by majority vote within each cell.

    Cells with no classified pixel get -1. Also returns the mean uncertainty per cell when provided.
    """
    h, w = class_map.shape
    rows, cols = min(rows, h), min(cols, w)
    r_edges = np.linspace(0, h, rows + 1).astype(int)
    c_edges = np.linspace(0, w, cols + 1).astype(int)
    grid = np.full((rows, cols), -1, dtype=np.int16)
    unc = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        strip = class_map[r_edges[r]:r_edges[r + 1]]
        ustrip = uncertainty[r_edges[r]:r_edges[r + 1]] if uncertainty is not None else None
        for c in range(cols):
            cell = strip[:, c_edges[c]:c_edges[c + 1]]
            vals = cell[cell != IGNORE_INDEX]
            if vals.size:
                grid[r, c] = int(np.bincount(vals.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES].argmax())
            if ustrip is not None:
                unc[r, c] = float(ustrip[:, c_edges[c]:c_edges[c + 1]].mean())
    return {"rows": rows, "cols": cols, "cells": grid.tolist(),
            "uncertainty": np.round(unc, 4).tolist() if uncertainty is not None else None}
