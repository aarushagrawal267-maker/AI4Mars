"""Starts, monitors and controls training runs.

Each run is a separate Python process (``python -m ml.training.run``). The API
never blocks on training and stays usable if a run crashes or runs out of
memory. A watcher thread mirrors the run's ``status.json`` / ``history.jsonl``
into SQLite so experiment history survives restarts.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

from ml.training.config import TrainConfig
from ml.training.estimate import estimate_workload
from ml.training.trainer import read_history

from ..core.config import REPO_ROOT, Settings
from ..database.db import Database
from ..database.models import Experiment, utcnow
from .dataset_service import DatasetService

log = logging.getLogger(__name__)
ACTIVE_STATES = ("queued", "preparing", "running", "paused")


class TrainingError(RuntimeError):
    pass


class TrainingManager:
    def __init__(self, settings: Settings, db: Database, dataset: DatasetService):
        self.settings, self.db, self.dataset = settings, db, dataset
        self._proc: subprocess.Popen | None = None
        self._active_id: str | None = None
        self._lock = threading.Lock()
        self._watcher: threading.Thread | None = None
        self._recover_orphans()

    # ------------------------------------------------------------- lifecycle
    def _recover_orphans(self) -> None:
        """Runs still marked active from a previous server session cannot be running any more."""
        with self.db.session() as s:
            for exp in s.query(Experiment).filter(Experiment.status.in_(ACTIVE_STATES)).all():
                self._sync(exp)
                if exp.status in ACTIVE_STATES:
                    exp.status = "interrupted"
                    exp.message = ("The server stopped while this run was active. Checkpoints from completed "
                                   "epochs are still available." if self.has_checkpoint(exp.id)
                                   else "The server stopped before the first epoch finished.")
                    exp.finished_at = utcnow()

    def run_dir(self, exp_id: str) -> Path:
        return self.settings.experiments_dir / exp_id

    def checkpoint_dir(self, exp_id: str) -> Path:
        return self.settings.checkpoint_dir / exp_id

    def has_checkpoint(self, exp_id: str) -> bool:
        return (self.checkpoint_dir(exp_id) / "best.pt").is_file()

    def _split_sizes(self) -> tuple[int, int]:
        counts = self.dataset.status()["split_counts"]
        return counts["train"], counts["val"]

    # -------------------------------------------------------------- planning
    def defaults(self) -> dict:
        """Hardware-aware starting values. They are suggestions; the user can change every one."""
        cuda = torch.cuda.is_available()
        n_train, n_val = self._split_sizes()
        if cuda:
            cfg = {"epochs": 30, "batch_size": 16, "resolution": 384, "max_train_samples": None,
                   "max_val_samples": None, "learning_rate": 1e-3}
            reason = "CUDA GPU detected: full training split at 384 px."
        else:
            cfg = {"epochs": 8, "batch_size": 8, "resolution": 192,
                   "max_train_samples": min(n_train, 1000) if n_train else 1000,
                   "max_val_samples": min(n_val, 200) if n_val else 200, "learning_rate": 1e-3}
            reason = ("No GPU detected: CPU mode with a 1000-image subset at 192 px so a run finishes in "
                      "minutes rather than hours. Increase gradually.")
        return {"config": {**TrainConfig().to_dict(), **cfg}, "reason": reason, "cuda_available": cuda,
                "train_available": n_train, "val_available": n_val}

    def estimate(self, cfg: TrainConfig) -> dict:
        n_train, n_val = self._split_sizes()
        if n_train < 2 or n_val < 1:
            raise TrainingError("No dataset is loaded. Load AI4Mars in the Dataset Explorer first.")
        return estimate_workload(cfg, n_train, n_val)

    # ----------------------------------------------------------------- start
    def start(self, cfg: TrainConfig, name: str | None, confirm_large: bool) -> dict:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise TrainingError("A training run is already in progress. Stop it before starting another.")
            index = self.dataset.index()
            if index is None:
                raise TrainingError("No dataset is loaded. Load AI4Mars in the Dataset Explorer first.")
            n_train, n_val = self._split_sizes()
            if n_train < 2 or n_val < 1:
                raise TrainingError("The loaded dataset has no usable train/validation pairs.")
            cfg = cfg.validate()
            if cfg.device == "cuda" and not torch.cuda.is_available():
                raise TrainingError("CUDA was selected but no CUDA GPU is available. Choose CPU or auto.")
            estimate = estimate_workload(cfg, n_train, n_val)
            if not estimate["fits_in_memory"]:
                raise TrainingError(estimate["error"])
            if estimate["large_job"] and not confirm_large:
                hours = estimate["estimated_total_seconds"] / 3600
                raise TrainingError(f"LARGE_JOB: this run is estimated at about {hours:.1f} hours on "
                                    f"{estimate['device']}. Confirm explicitly to start it.")
            stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            exp_id = f"{cfg.arch}-{stamp}-{secrets.token_hex(2)}"
            run_dir, ckpt_dir = self.run_dir(exp_id), self.checkpoint_dir(exp_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            job = {"config": cfg.to_dict(), "raw_root": str(self.settings.raw_dir),
                   "processed_root": str(self.settings.processed_dir), "index_path": str(self.settings.index_path),
                   "checkpoint_dir": str(ckpt_dir), "estimate": estimate}
            (run_dir / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
            (run_dir / "control.json").write_text(json.dumps({"command": "run"}), encoding="utf-8")
            with self.db.session() as s:
                s.add(Experiment(id=exp_id, name=name or f"{cfg.arch} {stamp}", arch=cfg.arch, status="queued",
                                 config=cfg.to_dict(), history=[], started_at=utcnow(),
                                 dataset_fingerprint=index.get("fingerprint"), message="Starting"))
            env = {**os.environ, "PYTHONPATH": str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   "PYTHONUNBUFFERED": "1"}
            stdout = open(run_dir / "process.log", "ab")
            self._proc = subprocess.Popen([sys.executable, "-m", "ml.training.run", "--run-dir", str(run_dir)],
                                          cwd=str(REPO_ROOT), env=env, stdout=stdout, stderr=subprocess.STDOUT)
            stdout.close()
            self._active_id = exp_id
            self._watcher = threading.Thread(target=self._watch, args=(exp_id, self._proc), daemon=True,
                                             name=f"train-watch-{exp_id}")
            self._watcher.start()
            log.info("Started training run %s (pid %s)", exp_id, self._proc.pid)
            return {"experiment_id": exp_id, "estimate": estimate}

    # ---------------------------------------------------------------- watcher
    def _read_status(self, exp_id: str) -> dict:
        try:
            return json.loads((self.run_dir(exp_id) / "status.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _sync(self, exp: Experiment) -> dict:
        """Copy the run's on-disk status/history into the database row."""
        st = self._read_status(exp.id)
        if not st:
            return st
        history = read_history(self.run_dir(exp.id))
        exp.status = st.get("state", exp.status)
        exp.message = st.get("message")
        exp.error = st.get("error")
        exp.history = history
        exp.epochs_completed = len(history)
        exp.best_val_miou = st.get("best_val_miou") if st.get("best_val_miou", -1) not in (None, -1.0) else None
        exp.best_epoch = st.get("best_epoch") or None
        exp.device = st.get("device", exp.device)
        exp.parameters = st.get("parameters", exp.parameters)
        exp.n_train = st.get("n_train", exp.n_train)
        exp.n_val = st.get("n_val", exp.n_val)
        exp.elapsed_seconds = st.get("elapsed_seconds", exp.elapsed_seconds)
        if st.get("pretrained_loaded") is not None:
            exp.pretrained_loaded = bool(st["pretrained_loaded"])
        if st.get("finished_at") and exp.finished_at is None:
            exp.finished_at = utcnow()
        return st

    def _watch(self, exp_id: str, proc: subprocess.Popen) -> None:
        while True:
            code = proc.poll()
            try:
                with self.db.session() as s:
                    exp = s.get(Experiment, exp_id)
                    if exp is None:
                        return
                    self._sync(exp)
                    if code is not None and exp.status in ACTIVE_STATES:
                        # The process died without writing a final state (killed, hard crash, OS OOM-killer).
                        exp.status = "failed"
                        exp.error = exp.message = (
                            f"The training process ended unexpectedly (exit code {code}). On laptops this is "
                            "usually the operating system killing it for using too much memory: lower the batch "
                            "size or resolution. See the training log for details.")
                        exp.finished_at = utcnow()
            except Exception:  # noqa: BLE001 - never let the watcher die on a transient DB/file error
                log.exception("Training watcher error")
            if code is not None:
                return
            time.sleep(1.0)

    # ---------------------------------------------------------------- control
    def _command(self, command: str) -> None:
        if self._active_id is None or self._proc is None or self._proc.poll() is not None:
            raise TrainingError("No training run is in progress.")
        (self.run_dir(self._active_id) / "control.json").write_text(json.dumps({"command": command}), encoding="utf-8")

    def pause(self) -> None:
        self._command("pause")

    def resume(self) -> None:
        self._command("run")

    def stop(self, wait_seconds: float = 60.0) -> None:
        self._command("stop")
        proc, exp_id = self._proc, self._active_id

        def enforce() -> None:  # the run stops itself between batches; kill only if it never does
            try:
                proc.wait(timeout=wait_seconds)
            except subprocess.TimeoutExpired:
                log.warning("Training run %s did not stop in time; terminating", exp_id)
                proc.terminate()

        threading.Thread(target=enforce, daemon=True).start()

    def shutdown(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._command("stop")
                self._proc.wait(timeout=20)
            except Exception:  # noqa: BLE001
                self._proc.terminate()

    # ----------------------------------------------------------------- status
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict:
        exp_id = self._active_id
        if exp_id is None:
            with self.db.session() as s:
                last = s.query(Experiment).filter(Experiment.is_fixture.is_(False)) \
                    .order_by(Experiment.created_at.desc()).first()
                exp_id = last.id if last else None
        if exp_id is None:
            return {"active": False, "experiment_id": None, "state": "idle"}
        st = self._read_status(exp_id)
        with self.db.session() as s:
            exp = s.get(Experiment, exp_id)
            if exp is None:
                return {"active": False, "experiment_id": None, "state": "idle"}
            return {
                "active": self.is_running() and exp_id == self._active_id,
                "experiment_id": exp.id, "name": exp.name, "arch": exp.arch, "config": exp.config,
                "state": exp.status if not st else st.get("state", exp.status),
                "message": st.get("message", exp.message),
                "epoch": st.get("epoch", exp.epochs_completed), "epochs_total": exp.config.get("epochs"),
                "batch": st.get("batch"), "batches_per_epoch": st.get("batches_per_epoch"),
                "progress": st.get("progress", 0.0),
                "train_loss_running": st.get("train_loss_running"),
                "best_val_miou": exp.best_val_miou, "best_epoch": exp.best_epoch,
                "elapsed_seconds": st.get("elapsed_seconds", exp.elapsed_seconds),
                "eta_seconds": st.get("eta_seconds"),
                "device": st.get("device", exp.device), "parameters": st.get("parameters", exp.parameters),
                "n_train": st.get("n_train", exp.n_train), "n_val": st.get("n_val", exp.n_val),
                "pretrained_loaded": st.get("pretrained_loaded", exp.pretrained_loaded),
                "has_checkpoint": self.has_checkpoint(exp.id),
                "error": st.get("error", exp.error),
                "history": read_history(self.run_dir(exp.id)) or exp.history,
            }

    def logs(self, exp_id: str | None, tail: int) -> dict:
        exp_id = exp_id or self._active_id
        if exp_id is None:
            with self.db.session() as s:
                last = s.query(Experiment).order_by(Experiment.created_at.desc()).first()
                exp_id = last.id if last else None
        if exp_id is None:
            return {"experiment_id": None, "lines": []}
        with self.db.session() as s:
            if s.get(Experiment, exp_id) is None:  # ids come from the database, never from a raw path
                raise KeyError(exp_id)
        lines: list[str] = []
        for name in ("train.log", "process.log"):
            path = self.run_dir(exp_id) / name
            if path.is_file() and path.stat().st_size:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                break
        return {"experiment_id": exp_id, "lines": lines[-tail:]}
