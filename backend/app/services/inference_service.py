"""Runs local model inference on an image and stores the result as an 'analysis'."""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ml.inference.predictor import predict
from ml.inference.terrain import class_grid, terrain_statistics
from ml.inference.visualize import colorize, heatmap, overlay, png_bytes, side_by_side

from ..core.config import Settings
from ..database.db import Database
from ..database.models import Analysis
from .model_registry import ModelRegistry

log = logging.getLogger(__name__)
RENDER_KINDS = ("original", "mask", "overlay", "side_by_side", "confidence", "uncertainty", "class_ids", "confidence_raw")


class InferenceService:
    def __init__(self, settings: Settings, db: Database, registry: ModelRegistry):
        self.settings, self.db, self.registry = settings, db, registry

    def _dir(self, analysis_id: str) -> Path:
        return self.settings.analyses_dir / analysis_id

    def analyze(self, rgb: np.ndarray, experiment_id: str, filename: str, source: str,
                sample_id: str | None = None) -> dict:
        model, meta, device, info = self.registry.load(experiment_id)
        with self.registry.inference_lock:
            try:
                result = predict(model, int(meta["resolution"]), rgb, device, self.settings.max_output_side)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise MemoryError("Not enough memory to run this model on this image.") from exc
                raise
        class_map = result["class_map"]
        oh, ow = result["output_size"]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
        original = rgb if (rgb.shape[0], rgb.shape[1]) == (oh, ow) else cv2.resize(
            rgb, (ow, oh), interpolation=cv2.INTER_AREA)
        stats = terrain_statistics(class_map, result["confidence"], result["uncertainty"])
        analysis_id = uuid.uuid4().hex[:16]
        out = self._dir(analysis_id)
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(original).save(out / "original.png", compress_level=3)
        # class ids are stored unmodified (0-3) so colours can change without touching predictions
        Image.fromarray(class_map, mode="L").save(out / "class_ids.png", compress_level=3)
        Image.fromarray((result["confidence"] * 255).round().astype(np.uint8), mode="L").save(out / "confidence.png")
        Image.fromarray((result["uncertainty"] * 255).round().astype(np.uint8), mode="L").save(out / "uncertainty.png")
        run_meta = {
            "input_size": result["input_size"], "output_size": result["output_size"],
            "original_size": result["original_size"], "inference_ms": round(result["inference_ms"], 3),
            "device": str(device), "resolution": int(meta["resolution"]), "mean_probs": result["mean_probs"],
            "model_parameters": meta.get("parameters"),
            "color_input": bool(rgb.ndim == 3 and not (np.array_equal(rgb[..., 0], rgb[..., 1])
                                                       and np.array_equal(rgb[..., 1], rgb[..., 2]))),
            "grayscale_mean": round(float(gray.mean()), 2),
        }
        with self.db.session() as s:
            s.add(Analysis(id=analysis_id, experiment_id=info["id"], model_name=info["name"], arch=info["arch"],
                           source=source, filename=filename[:300], sample_id=sample_id,
                           is_fixture=info["is_fixture"], stats=stats, meta=run_meta))
        if self.settings.public_mode:
            self._prune(self.settings.public_max_analyses)
        return self.get(analysis_id)

    def _prune(self, keep: int) -> None:
        """Public mode: bound disk use by removing the oldest analyses beyond ``keep``."""
        with self.db.session() as s:
            old = s.query(Analysis).order_by(Analysis.created_at.desc()).offset(keep).all()
            ids = [a.id for a in old]
            for a in old:
                s.delete(a)
        for i in ids:
            shutil.rmtree(self._dir(i), ignore_errors=True)

    # ---------------------------------------------------------------- queries
    @staticmethod
    def _public(a: Analysis, full: bool = True) -> dict:
        d = {"id": a.id, "created_at": a.created_at.isoformat(), "experiment_id": a.experiment_id,
             "model_name": a.model_name, "arch": a.arch, "source": a.source, "filename": a.filename,
             "sample_id": a.sample_id, "is_fixture": a.is_fixture, "meta": a.meta,
             "dominant_class": a.stats.get("dominant_class"), "class_percent": a.stats.get("class_percent"),
             "mean_confidence": (a.stats.get("confidence") or {}).get("mean")}
        if full:
            d["stats"] = a.stats
        return d

    def get(self, analysis_id: str) -> dict:
        with self.db.session() as s:
            a = s.get(Analysis, analysis_id)
            if a is None:
                raise KeyError(analysis_id)
            return self._public(a)

    def list(self, limit: int = 50) -> list[dict]:
        with self.db.session() as s:
            rows = s.query(Analysis).order_by(Analysis.created_at.desc()).limit(limit).all()
            return [self._public(a, full=False) for a in rows]

    def delete(self, analysis_id: str) -> None:
        with self.db.session() as s:
            a = s.get(Analysis, analysis_id)
            if a is None:
                raise KeyError(analysis_id)
            s.delete(a)
        shutil.rmtree(self._dir(analysis_id), ignore_errors=True)

    # -------------------------------------------------------------- rendering
    def _load(self, analysis_id: str, name: str) -> np.ndarray:
        self.get(analysis_id)  # 404 for unknown ids; ids are looked up, never used as raw paths
        path = self._dir(analysis_id) / name
        if not path.is_file():
            raise KeyError(f"{analysis_id}/{name}")
        return np.asarray(Image.open(path))

    def class_map(self, analysis_id: str) -> np.ndarray:
        return self._load(analysis_id, "class_ids.png")

    def render(self, analysis_id: str, kind: str, colors: dict[str, str], alpha: float = 0.5) -> bytes:
        if kind not in RENDER_KINDS:
            raise KeyError(kind)
        if kind == "original":
            return png_bytes(self._load(analysis_id, "original.png"))
        if kind == "class_ids":
            return png_bytes(self._load(analysis_id, "class_ids.png"))
        if kind == "confidence_raw":  # greyscale, value = confidence * 255 (read by the pixel inspector)
            return png_bytes(self._load(analysis_id, "confidence.png"))
        if kind == "confidence":
            return png_bytes(heatmap(self._load(analysis_id, "confidence.png") / 255.0, cv2.COLORMAP_VIRIDIS))
        if kind == "uncertainty":
            return png_bytes(heatmap(self._load(analysis_id, "uncertainty.png") / 255.0, cv2.COLORMAP_INFERNO))
        cmap = self.class_map(analysis_id)
        if kind == "mask":
            return png_bytes(colorize(cmap, colors))
        original = self._load(analysis_id, "original.png")
        if kind == "overlay":
            return png_bytes(overlay(original, cmap, colors, alpha))
        return png_bytes(side_by_side(original, colorize(cmap, colors), overlay(original, cmap, colors, alpha)))

    def grid(self, analysis_id: str, size: int) -> dict:
        """Downsample the prediction to a navigation grid whose long side has ``size`` cells."""
        cmap = self.class_map(analysis_id)
        unc = self._load(analysis_id, "uncertainty.png").astype(np.float32) / 255.0
        h, w = cmap.shape
        scale = size / max(h, w)
        rows, cols = max(2, round(h * scale)), max(2, round(w * scale))
        return class_grid(cmap, rows, cols, unc)
