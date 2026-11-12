"""Build and validate the dataset index.

The index is the single description of what is on disk: every image/label pair,
which split it belongs to, the per-class pixel counts of its label, and a
validation report (missing pairs, unreadable files, size mismatches, unexpected
label values). Everything shown in the Dataset Explorer comes from this file;
nothing is estimated.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from ..classes import CLASS_KEYS, IGNORE_INDEX, NUM_CLASSES, VALID_LABEL_VALUES
from .layout import IMAGE_EXT, DatasetLayout, find_layout
from .naming import parse_msl_name

log = logging.getLogger(__name__)
INDEX_VERSION = 3
MAX_LISTED = 200  # cap for problem lists stored in the report


def _label_stem(name: str) -> str:
    """Label filename -> image stem (``X_merged.png`` -> ``X``; ``X.png`` -> ``X``)."""
    stem = Path(name).stem
    return stem[:-len("_merged")] if stem.endswith("_merged") else stem


def _inspect_label(path: Path) -> dict:
    """Open one label and report its size, pixel-value histogram and any problems."""
    try:
        with Image.open(path) as im:
            mode, size = im.mode, im.size
            arr = np.asarray(im)
    except Exception as exc:  # unreadable / truncated file
        return {"error": f"{type(exc).__name__}: {exc}"}
    if arr.ndim != 2:
        return {"error": f"label is not single-channel (mode {mode}, shape {arr.shape})"}
    hist = np.bincount(arr.ravel().astype(np.int64), minlength=256)
    present = {int(v) for v in np.nonzero(hist)[0]}
    return {
        "width": size[0], "height": size[1],
        "counts": [int(hist[c]) for c in range(NUM_CLASSES)],
        "unlabeled": int(hist[IGNORE_INDEX]),
        "unexpected": sorted(present - VALID_LABEL_VALUES),
    }


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def assign_validation_split(records: list[dict], val_fraction: float, seed: int) -> None:
    """Mark ~``val_fraction`` of training records as validation, grouped by rover position.

    All frames sharing a site+drive key go to the same split so near-duplicate
    views of one location never straddle train and validation.
    """
    train = [r for r in records if r["split"] in ("train", "val")]
    groups: dict[str, list[dict]] = {}
    for r in train:
        r["split"] = "train"
        groups.setdefault(r["group"], []).append(r)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target = int(round(len(train) * val_fraction))
    taken = 0
    for k in keys:
        if taken >= target:
            break
        for r in groups[k]:
            r["split"] = "val"
        taken += len(groups[k])


def build_index(raw_root: Path, val_fraction: float = 0.1, seed: int = 42,
                progress: Callable[[float, str], None] | None = None, workers: int = 4) -> dict:
    """Scan ``raw_root``, pair images with labels, validate every label and assign splits."""
    t0 = time.time()
    raw_root = Path(raw_root)
    layout: DatasetLayout | None = find_layout(raw_root)
    if layout is None:
        raise FileNotFoundError(
            "No AI4Mars MSL NavCam dataset found. Expected a folder containing images/edr and labels/train."
        )

    def rel(p: Path) -> str:
        return str(p.relative_to(raw_root)).replace("\\", "/")

    images = {p.stem: p for p in sorted(layout.images_dir.iterdir()) if p.suffix.lower() in IMAGE_EXT}
    train_labels = {_label_stem(p.name): p for p in sorted(layout.train_labels_dir.iterdir())
                    if p.suffix.lower() == ".png"}
    test_labels: dict[str, dict[str, Path]] = {}
    for variant, d in layout.test_label_dirs.items():
        for p in sorted(d.iterdir()):
            if p.suffix.lower() == ".png":
                test_labels.setdefault(_label_stem(p.name), {})[variant] = p

    report = {
        "images_found": len(images),
        "train_labels_found": len(train_labels),
        "test_images_with_labels": 0,
        "test_label_variants": {v: 0 for v in layout.test_label_dirs},
        "valid_pairs": 0,
        "labels_without_image": [], "labels_without_image_count": 0,
        "images_without_label_count": 0,
        "unreadable": [], "unreadable_count": 0,
        "size_mismatch": [], "size_mismatch_count": 0,
        "unexpected_values": [], "unexpected_values_count": 0,
        "empty_labels_count": 0,
        "train_test_filename_overlap": sorted(set(train_labels) & set(test_labels))[:MAX_LISTED],
        "observed_label_values": [],
    }

    def note(key: str, item) -> None:
        report[key + "_count"] += 1
        if len(report[key]) < MAX_LISTED:
            report[key].append(item)

    jobs: list[tuple[str, str, str, Path]] = []  # (stem, split, variant, label path)
    for stem, lp in train_labels.items():
        if stem in test_labels:
            continue  # never train on an image that is part of the expert test set
        if stem not in images:
            note("labels_without_image", lp.name)
            continue
        jobs.append((stem, "train", "train", lp))
    for stem, variants in test_labels.items():
        if stem not in images:
            for lp in variants.values():
                note("labels_without_image", lp.name)
            continue
        for variant, lp in variants.items():
            jobs.append((stem, "test", variant, lp))
    report["images_without_label_count"] = len(set(images) - set(train_labels) - set(test_labels))

    total = max(len(jobs), 1)
    done = 0
    records: dict[str, dict] = {}
    observed: set[int] = set()

    def work(job):
        stem, split, variant, lp = job
        return job, _inspect_label(lp), _image_size(images[stem])

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for (stem, split, variant, lp), info, img_size in pool.map(work, jobs):
            done += 1
            if progress and (done % 100 == 0 or done == total):
                progress(done / total, f"Validated {done}/{total} labels")
            if "error" in info:
                note("unreadable", {"file": lp.name, "error": info["error"]})
                continue
            if img_size is None:
                note("unreadable", {"file": images[stem].name, "error": "image could not be opened"})
                continue
            if img_size != (info["width"], info["height"]):
                note("size_mismatch", {"image": images[stem].name, "image_size": list(img_size),
                                       "label_size": [info["width"], info["height"]]})
                continue
            if info["unexpected"]:
                note("unexpected_values", {"file": lp.name, "values": info["unexpected"]})
                continue
            observed.update(c for c in range(NUM_CLASSES) if info["counts"][c])
            if info["unlabeled"]:
                observed.add(IGNORE_INDEX)
            name = parse_msl_name(stem)
            rec = records.get(stem)
            if rec is None:
                rec = records[stem] = {
                    "id": stem, "split": split, "image": rel(images[stem]),
                    "width": info["width"], "height": info["height"],
                    "camera": name.camera, "sclk": name.sclk, "site": name.site, "drive": name.drive,
                    "group": name.group, "labels": {},
                }
            rec["labels"][variant] = {"path": rel(lp), "counts": info["counts"], "unlabeled": info["unlabeled"]}
            if split == "test":
                report["test_label_variants"][variant] = report["test_label_variants"].get(variant, 0) + 1

    ordered: list[dict] = []
    for rec in sorted(records.values(), key=lambda r: (r["sclk"], r["id"])):
        if rec["split"] == "train":
            lab = rec["labels"]["train"]
            if sum(lab["counts"]) == 0:
                report["empty_labels_count"] += 1
                continue  # nothing to supervise: fully unlabelled
        ordered.append(rec)
    assign_validation_split(ordered, val_fraction, seed)
    report["valid_pairs"] = len(ordered)
    report["test_images_with_labels"] = sum(1 for r in ordered if r["split"] == "test")
    report["observed_label_values"] = sorted(observed)

    train_groups = {r["group"] for r in ordered if r["split"] == "train"}
    val_groups = {r["group"] for r in ordered if r["split"] == "val"}
    test_recs = [r for r in ordered if r["split"] == "test"]
    leakage = {
        "train_val_shared_positions": len(train_groups & val_groups),
        "test_images_sharing_position_with_train": sum(1 for r in test_recs if r["group"] in train_groups),
        "test_images": len(test_recs),
        "note": "Train/validation are split by rover position (site+drive). The test set is the official "
                "expert-labelled partition and is used unchanged; any position overlap with training data "
                "is reported here rather than hidden.",
    }
    index = {
        "version": INDEX_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scan_seconds": round(time.time() - t0, 2),
        "layout": layout.describe(raw_root),
        "split_config": {"val_fraction": val_fraction, "seed": seed, "grouping": "site+drive"},
        "classes": CLASS_KEYS,
        "ignore_index": IGNORE_INDEX,
        "validation": report,
        "leakage_check": leakage,
        "records": ordered,
    }
    index["statistics"] = compute_statistics(index)
    index["fingerprint"] = fingerprint(index)
    return index


def compute_statistics(index: dict) -> dict:
    """Per-split image counts, class pixel distribution and unlabelled percentage."""
    variants = sorted({v for r in index["records"] if r["split"] == "test" for v in r["labels"]})
    stats: dict = {"splits": {}, "test_variants": variants}

    def summarise(recs: list[dict], variant: str) -> dict:
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        unlabeled = 0
        present = np.zeros(NUM_CLASSES, dtype=np.int64)
        n = 0
        for r in recs:
            lab = r["labels"].get(variant)
            if lab is None:
                continue
            n += 1
            c = np.asarray(lab["counts"], dtype=np.int64)
            counts += c
            present += (c > 0)
            unlabeled += lab["unlabeled"]
        labeled = int(counts.sum())
        total = labeled + unlabeled
        return {
            "images": n,
            "pixels_total": total,
            "pixels_labeled": labeled,
            "unlabeled_percent": round(100 * unlabeled / total, 4) if total else None,
            "class_pixels": {k: int(counts[i]) for i, k in enumerate(CLASS_KEYS)},
            "class_percent_of_labeled": {k: (round(100 * int(counts[i]) / labeled, 4) if labeled else None)
                                         for i, k in enumerate(CLASS_KEYS)},
            "images_containing_class": {k: int(present[i]) for i, k in enumerate(CLASS_KEYS)},
        }

    for split in ("train", "val"):
        stats["splits"][split] = summarise([r for r in index["records"] if r["split"] == split], "train")
    test = [r for r in index["records"] if r["split"] == "test"]
    stats["splits"]["test"] = {v: summarise(test, v) for v in variants}
    stats["total_images"] = len(index["records"])
    stats["cameras"] = {}
    for r in index["records"]:
        stats["cameras"][r["camera"]] = stats["cameras"].get(r["camera"], 0) + 1
    stats["rover_positions"] = len({r["group"] for r in index["records"]})
    return stats


def fingerprint(index: dict) -> str:
    """Stable hash of the record ids and splits; caches are keyed on it."""
    h = hashlib.sha256()
    for r in index["records"]:
        h.update(f"{r['id']}|{r['split']}\n".encode())
    return h.hexdigest()[:16]


def save_index(index: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(index), encoding="utf-8")
    tmp.replace(path)


def load_index(path: Path) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if data.get("version") == INDEX_VERSION else None
