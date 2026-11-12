"""Resolution-specific preprocessing cache.

Decoding and resizing 1024x1024 images every epoch dominates CPU training time.
We resize each train/validation pair once to the requested resolution and store
the result as memory-mapped ``uint8`` arrays, so later epochs (and later
experiments at the same resolution) read straight from disk with very little RAM.

Images are resized with area interpolation; labels with nearest-neighbour so that
class ids are never blended and 255 (unlabelled) is preserved.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

log = logging.getLogger(__name__)


def read_gray(path: Path) -> np.ndarray:
    """Read an image file as a 2-D uint8 grayscale array (handles non-ASCII Windows paths)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not decode image: {path}")
    return img


def read_label(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    lab = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if lab is None or lab.ndim != 2:
        raise IOError(f"Could not decode label: {path}")
    return lab


def cache_dir(processed_root: Path, resolution: int, fingerprint: str) -> Path:
    return Path(processed_root) / f"cache_{resolution}_{fingerprint}"


def ensure_cache(index: dict, raw_root: Path, processed_root: Path, resolution: int, split: str,
                 progress: Callable[[float, str], None] | None = None, workers: int = 4,
                 should_stop: Callable[[], bool] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return memory-mapped (images, labels, ids) for ``split`` at ``resolution``, building if needed."""
    if split not in ("train", "val"):
        raise ValueError("The preprocessing cache only covers the train and val splits")
    raw_root = Path(raw_root)
    cdir = cache_dir(processed_root, resolution, index["fingerprint"])
    img_path, lab_path, ids_path = cdir / f"{split}_images.npy", cdir / f"{split}_labels.npy", cdir / f"{split}_ids.json"
    records = [r for r in index["records"] if r["split"] == split]
    ids = [r["id"] for r in records]
    if img_path.exists() and lab_path.exists() and ids_path.exists():
        try:
            if json.loads(ids_path.read_text()) == ids:
                return np.load(img_path, mmap_mode="r"), np.load(lab_path, mmap_mode="r"), ids
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    cdir.mkdir(parents=True, exist_ok=True)
    n = len(records)
    shape = (n, resolution, resolution)
    img_tmp, lab_tmp = img_path.with_suffix(".tmp.npy"), lab_path.with_suffix(".tmp.npy")
    imgs = np.lib.format.open_memmap(img_tmp, mode="w+", dtype=np.uint8, shape=shape)
    labs = np.lib.format.open_memmap(lab_tmp, mode="w+", dtype=np.uint8, shape=shape)

    def work(i: int) -> int:
        r = records[i]
        img = read_gray(raw_root / r["image"])
        lab = read_label(raw_root / r["labels"]["train"]["path"])
        imgs[i] = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
        labs[i] = cv2.resize(lab, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
        return i

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for _ in pool.map(work, range(n)):
            done += 1
            if should_stop and done % 50 == 0 and should_stop():
                pool.shutdown(wait=False, cancel_futures=True)
                raise InterruptedError("Cache build cancelled")
            if progress and (done % 100 == 0 or done == n):
                progress(done / max(n, 1), f"Caching {split} split at {resolution}px: {done}/{n}")
    imgs.flush()
    labs.flush()
    del imgs, labs
    img_tmp.replace(img_path)
    lab_tmp.replace(lab_path)
    ids_path.write_text(json.dumps(ids))
    return np.load(img_path, mmap_mode="r"), np.load(lab_path, mmap_mode="r"), ids
