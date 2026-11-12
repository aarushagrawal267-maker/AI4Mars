"""Dataset management for the API: status, acquisition jobs, index access and sample rendering."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from ml.classes import CLASS_KEYS
from ml.datasets import acquire, sources
from ml.datasets.cache import read_gray, read_label
from ml.datasets.index import build_index, load_index, save_index
from ml.datasets.layout import directory_tree, find_layout
from ml.inference.visualize import colorize, overlay, png_bytes

from ..core.config import Settings
from .jobs import Job, JobRunner

log = logging.getLogger(__name__)
JOB_KIND = "dataset"


class DatasetService:
    def __init__(self, settings: Settings, jobs: JobRunner):
        self.settings = settings
        self.jobs = jobs
        self._index: dict | None = None
        self._index_mtime: float | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ index
    def index(self) -> dict | None:
        """Return the cached dataset index, reloading it when the file on disk changed."""
        path = self.settings.index_path
        with self._lock:
            mtime = path.stat().st_mtime if path.is_file() else None
            if mtime != self._index_mtime:
                self._index = load_index(path) if mtime else None
                self._index_mtime = mtime
            return self._index

    def require_index(self) -> dict:
        idx = self.index()
        if idx is None:
            raise LookupError("No dataset has been loaded yet. Download or import AI4Mars, then load it.")
        return idx

    def record(self, sample_id: str) -> dict:
        for r in self.require_index()["records"]:
            if r["id"] == sample_id:
                return r
        raise KeyError(f"Unknown sample id: {sample_id}")

    # ----------------------------------------------------------------- status
    def local_archives(self) -> list[dict]:
        d = self.settings.imports_dir
        if not d.is_dir():
            return []
        return [{"name": p.name, "bytes": p.stat().st_size} for p in sorted(d.iterdir())
                if p.is_file() and p.suffix.lower() == ".zip"]

    def status(self) -> dict:
        layout = find_layout(self.settings.raw_dir)
        idx = self.index()
        job = self.jobs.latest(JOB_KIND)
        return {
            "raw_dir": str(self.settings.raw_dir),
            "imports_dir": str(self.settings.imports_dir),
            "dataset_found": layout is not None,
            "layout": layout.describe(self.settings.raw_dir) if layout else None,
            "indexed": idx is not None,
            "index_created": idx["created"] if idx else None,
            "fingerprint": idx["fingerprint"] if idx else None,
            "total_images": idx["statistics"]["total_images"] if idx else 0,
            "split_counts": self._split_counts(idx),
            "needs_index": layout is not None and idx is None,
            "job": job.to_dict() if job else None,
            "local_archives": self.local_archives(),
            "source": {
                "name": "AI4Mars: A Dataset for Terrain-Aware Autonomous Driving on Mars (merged v0.6)",
                "record_url": sources.ZENODO_RECORD_URL, "doi": sources.ZENODO_DOI,
                "archive": sources.MERGED_ZIP_NAME, "archive_bytes": sources.MERGED_ZIP_SIZE,
                "license": "CC-BY-4.0", "kaggle_mirror": sources.KAGGLE_MIRROR_URL,
                "subset": "Curiosity (MSL) NavCam: msl/ncam/images/edr + labels",
            },
        }

    @staticmethod
    def _split_counts(idx: dict | None) -> dict:
        if not idx:
            return {"train": 0, "val": 0, "test": 0}
        out = {"train": 0, "val": 0, "test": 0}
        for r in idx["records"]:
            out[r["split"]] += 1
        return out

    # ------------------------------------------------------------------- jobs
    def _scan(self, job: Job) -> dict:
        job.update(0.0, "Scanning dataset folders", "index")
        idx = build_index(self.settings.raw_dir, self.settings.val_fraction, self.settings.split_seed,
                          progress=lambda f, m: job.update(f, m, "index"))
        save_index(idx, self.settings.index_path)
        return {"valid_pairs": idx["validation"]["valid_pairs"], "fingerprint": idx["fingerprint"]}

    def start_scan(self) -> Job:
        if find_layout(self.settings.raw_dir) is None:
            raise LookupError("No AI4Mars data found under the dataset directory. Download or import it first.")
        return self.jobs.submit(JOB_KIND, self._scan, {"action": "scan"})

    def start_download(self, mode: str, train_count: int | None) -> Job:
        def run(job: Job) -> dict:
            cb = lambda stage, frac, msg: job.update(frac, msg, stage)  # noqa: E731
            stop = job.cancel_event.is_set
            if mode == "full":
                summary = acquire.download_full(self.settings.raw_dir, progress=cb, should_stop=stop)
            else:
                summary = acquire.download_subset(self.settings.raw_dir, train_count, seed=self.settings.split_seed,
                                                  progress=cb, should_stop=stop)
            summary["index"] = self._scan(job)
            return summary

        return self.jobs.submit(JOB_KIND, run, {"action": "download", "mode": mode, "train_count": train_count})

    def start_import(self, archive_name: str) -> Job:
        # Only archives the user placed in data/raw/imports can be chosen, by bare file name.
        allowed = {a["name"] for a in self.local_archives()}
        if archive_name not in allowed:
            raise LookupError(f"Archive {archive_name!r} was not found in the imports folder")
        path = self.settings.imports_dir / archive_name

        def run(job: Job) -> dict:
            summary = acquire.extract_archive(path, self.settings.raw_dir,
                                              progress=lambda s, f, m: job.update(f, m, s),
                                              should_stop=job.cancel_event.is_set)
            summary["index"] = self._scan(job)
            return summary

        return self.jobs.submit(JOB_KIND, run, {"action": "import", "archive": archive_name})

    def cancel(self) -> bool:
        job = self.jobs.active(JOB_KIND)
        if job is None:
            return False
        job.cancel_event.set()
        return True

    # ---------------------------------------------------------------- queries
    def structure(self) -> list[dict]:
        return directory_tree(self.settings.raw_dir)

    def samples(self, split: str | None, offset: int, limit: int, contains: str | None) -> dict:
        idx = self.require_index()
        recs = idx["records"]
        if split:
            recs = [r for r in recs if r["split"] == split]
        if contains:
            ci = CLASS_KEYS.index(contains)
            recs = [r for r in recs if any(l["counts"][ci] > 0 for l in r["labels"].values())]
        page = recs[offset:offset + limit]
        return {"total": len(recs), "offset": offset, "limit": limit, "items": [self._public(r) for r in page]}

    @staticmethod
    def _public(r: dict) -> dict:
        labels = {}
        for variant, lab in r["labels"].items():
            labeled = sum(lab["counts"])
            total = labeled + lab["unlabeled"]
            labels[variant] = {
                "class_percent": {k: (round(100 * lab["counts"][i] / labeled, 2) if labeled else 0.0)
                                  for i, k in enumerate(CLASS_KEYS)},
                "unlabeled_percent": round(100 * lab["unlabeled"] / total, 2) if total else 0.0,
            }
        return {"id": r["id"], "split": r["split"], "width": r["width"], "height": r["height"],
                "camera": r["camera"], "site": r["site"], "drive": r["drive"], "sclk": r["sclk"],
                "variants": sorted(r["labels"]), "labels": labels}

    def sample(self, sample_id: str) -> dict:
        return self._public(self.record(sample_id))

    # -------------------------------------------------------------- rendering
    def _label_array(self, rec: dict, variant: str | None) -> np.ndarray:
        if variant is None:
            variant = "train" if "train" in rec["labels"] else ("min3" if "min3" in rec["labels"]
                                                                else sorted(rec["labels"])[0])
        if variant not in rec["labels"]:
            raise KeyError(f"Sample has no label variant {variant!r}")
        return read_label(self.settings.raw_dir / rec["labels"][variant]["path"])

    def image_array(self, sample_id: str) -> np.ndarray:
        return read_gray(self.settings.raw_dir / self.record(sample_id)["image"])

    def render(self, sample_id: str, kind: str, variant: str | None, size: int | None, colors: dict[str, str],
               alpha: float = 0.5) -> bytes:
        rec = self.record(sample_id)
        if kind == "image":
            out = read_gray(self.settings.raw_dir / rec["image"])
            interp = cv2.INTER_AREA
        elif kind == "label":
            out = colorize(self._label_array(rec, variant), colors)
            interp = cv2.INTER_NEAREST
        elif kind == "overlay":
            out = overlay(read_gray(self.settings.raw_dir / rec["image"]), self._label_array(rec, variant), colors, alpha)
            interp = cv2.INTER_AREA
        else:
            raise KeyError(f"Unknown render kind {kind!r}")
        if size and size < max(out.shape[:2]):
            scale = size / max(out.shape[:2])
            out = cv2.resize(out, (max(1, round(out.shape[1] * scale)), max(1, round(out.shape[0] * scale))),
                             interpolation=interp)
        return png_bytes(out)
