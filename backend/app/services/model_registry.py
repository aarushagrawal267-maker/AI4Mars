"""Lists trained checkpoints and loads them (with a small cache) for inference and evaluation."""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict

import torch

from ml.architectures.factory import build_model, load_checkpoint, resolve_device, save_checkpoint

from ..core.config import Settings
from ..database.db import Database
from ..database.models import Experiment, utcnow

log = logging.getLogger(__name__)
FIXTURE_ID = "fixture-untrained-unet"
FIXTURE_NAME = "TEST FIXTURE - untrained U-Net (not a real model)"


class ModelNotAvailable(LookupError):
    pass


class ModelRegistry:
    def __init__(self, settings: Settings, db: Database, cache_size: int = 2):
        self.settings, self.db = settings, db
        self._cache: OrderedDict[tuple, tuple] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()
        self.inference_lock = threading.Lock()  # one forward pass at a time keeps laptop memory bounded

    def checkpoint_path(self, exp_id: str, which: str = "best"):
        return self.settings.checkpoint_dir / exp_id / f"{which}.pt"

    def device(self) -> torch.device:
        return resolve_device("auto")

    def list_models(self) -> list[dict]:
        """Every experiment that has a usable best checkpoint on disk."""
        out = []
        with self.db.session() as s:
            for exp in s.query(Experiment).order_by(Experiment.created_at.desc()).all():
                path = self.checkpoint_path(exp.id)
                if not path.is_file():
                    continue
                out.append({
                    "id": exp.id, "name": exp.name, "arch": exp.arch, "status": exp.status,
                    "is_fixture": exp.is_fixture, "created_at": exp.created_at.isoformat(),
                    "resolution": exp.config.get("resolution"), "model_config": exp.config.get("model_config"),
                    "epochs_completed": exp.epochs_completed, "best_epoch": exp.best_epoch,
                    "best_val_miou": exp.best_val_miou, "parameters": exp.parameters,
                    "n_train": exp.n_train, "checkpoint_bytes": path.stat().st_size,
                    "pretrained_loaded": exp.pretrained_loaded,
                    "latest_test_evaluation": self._latest_eval(exp),
                })
        return out

    @staticmethod
    def _latest_eval(exp: Experiment) -> dict | None:
        tests = [e for e in exp.evaluations if e.split == "test"]
        if not tests:
            return None
        e = max(tests, key=lambda x: x.created_at)
        return {"id": e.id, "label_variant": e.label_variant, "images": e.images,
                "mean_iou": e.metrics.get("mean_iou"), "pixel_accuracy": e.metrics.get("pixel_accuracy"),
                "created_at": e.created_at.isoformat()}

    def load(self, exp_id: str) -> tuple[torch.nn.Module, dict, torch.device, dict]:
        """Return (model, checkpoint meta, device, experiment info). Raises ModelNotAvailable."""
        with self.db.session() as s:
            exp = s.get(Experiment, exp_id)
            if exp is None:
                raise ModelNotAvailable(f"Unknown model: {exp_id}")
            info = {"id": exp.id, "name": exp.name, "arch": exp.arch, "is_fixture": exp.is_fixture}
        path = self.checkpoint_path(exp_id)
        if not path.is_file():
            raise ModelNotAvailable("This experiment has no saved checkpoint (it never completed an epoch).")
        device = self.device()
        key = (exp_id, path.stat().st_mtime_ns, str(device))
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                model, meta = self._cache[key]
                return model, meta, device, info
        try:
            model, meta = load_checkpoint(path, device)
        except Exception as exc:  # corrupt/incompatible file: report, never crash the server
            log.exception("Failed to load checkpoint for %s", exp_id)
            raise ModelNotAvailable(f"The checkpoint could not be loaded: {exc}") from exc
        with self._lock:
            self._cache[key] = (model, meta)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return model, meta, device, info

    def forget(self, exp_id: str) -> None:
        with self._lock:
            for key in [k for k in self._cache if k[0] == exp_id]:
                del self._cache[key]

    def ensure_fixture(self) -> None:
        """Create the deterministic, UNTRAINED fixture model used only by automated interface tests.

        It is flagged ``is_fixture`` everywhere and the UI labels its output as not being a real prediction.
        """
        path = self.checkpoint_path(FIXTURE_ID)
        cfg = {"base_channels": 8, "depth": 3, "bilinear": True, "dropout": 0.0}
        if not path.is_file():
            torch.manual_seed(1234)
            model = build_model("unet", cfg)
            save_checkpoint(path, model, "unet", cfg, 128, {"is_fixture": True, "note": "untrained test fixture"})
        with self.db.session() as s:
            if s.get(Experiment, FIXTURE_ID) is None:
                s.add(Experiment(id=FIXTURE_ID, name=FIXTURE_NAME, arch="unet", status="fixture", is_fixture=True,
                                 config={"arch": "unet", "model_config": cfg, "resolution": 128},
                                 history=[], finished_at=utcnow(),
                                 message="Deterministic untrained weights for automated tests. Not a trained model."))
