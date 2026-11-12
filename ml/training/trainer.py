"""Training loop for the terrain segmentation models.

The trainer is deliberately independent of the web server. It communicates only
through files in its run directory, so the API can start it as a separate process
(keeping the web UI responsive and surviving out-of-memory crashes):

* ``status.json``  - current state/progress, rewritten atomically
* ``history.jsonl``- one JSON line per completed epoch (losses and validation metrics)
* ``train.log``    - human-readable log
* ``control.json`` - written by the API: ``{"command": "run" | "pause" | "stop"}``

Checkpoints ``last.pt`` (every completed epoch) and ``best.pt`` (best validation
mIoU) are written atomically, so stopping or crashing never corrupts or removes a
checkpoint from an epoch that already finished.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..architectures.factory import build_model, count_parameters, resolve_device, save_checkpoint
from ..classes import CLASS_KEYS, NUM_CLASSES
from ..datasets.cache import ensure_cache
from ..datasets.torch_dataset import TerrainDataset
from ..evaluation.metrics import confusion_matrix, metrics_from_confusion
from .config import TrainConfig
from .losses import SegmentationLoss, class_weights_from_counts

log = logging.getLogger("mti.train")


class StopRequested(Exception):
    """Raised inside the loop when the user asks the run to stop."""


class RunIO:
    """File-based status, history and control channel for one training run."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.run_dir / "status.json"
        self.history_path = self.run_dir / "history.jsonl"
        self.control_path = self.run_dir / "control.json"
        self.status: dict = {}
        self._last_poll = 0.0
        self._command = "run"
        self._last_write = 0.0

    def write_status(self, force: bool = True, **fields) -> None:
        self.status.update(fields)
        self.status["updated_at"] = time.time()
        if not force and time.time() - self._last_write < 0.75:
            return
        self._last_write = time.time()
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.status), encoding="utf-8")
        for _ in range(5):  # Windows can briefly lock the target while the API reads it
            try:
                os.replace(tmp, self.status_path)
                return
            except PermissionError:
                time.sleep(0.05)

    def append_history(self, row: dict) -> None:
        with open(self.history_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def command(self) -> str:
        now = time.time()
        if now - self._last_poll > 0.4:
            self._last_poll = now
            try:
                self._command = json.loads(self.control_path.read_text(encoding="utf-8")).get("command", "run")
            except (OSError, json.JSONDecodeError):
                self._command = "run"
        return self._command

    def checkpoint_gate(self) -> None:
        """Block while paused; raise StopRequested when asked to stop."""
        cmd = self.command()
        if cmd == "pause":
            previous = self.status.get("state")
            self.write_status(state="paused", message="Paused by user")
            log.info("Training paused")
            while cmd == "pause":
                time.sleep(0.5)
                self._last_poll = 0.0
                cmd = self.command()
            if cmd != "stop":
                self.write_status(state=previous or "running", message="Resumed")
                log.info("Training resumed")
        if cmd == "stop":
            raise StopRequested()


def read_history(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "history.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return isinstance(exc, MemoryError) or (isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower())


def select_indices(n: int, limit: int | None, seed: int) -> list[int]:
    """Deterministic subset of ``range(n)``; the full range when ``limit`` is None or >= n."""
    if limit is None or limit >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n, size=limit, replace=False))


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, criterion: SegmentationLoss, device: torch.device,
             io: RunIO | None = None) -> tuple[float, dict]:
    """Return (mean validation loss, metrics dict) over a labelled loader."""
    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_loss, batches = 0.0, 0
    for images, labels in loader:
        if io is not None:
            io.checkpoint_gate()
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += float(criterion(logits, labels))
        batches += 1
        cm += confusion_matrix(logits.argmax(1), labels)
    return total_loss / max(batches, 1), metrics_from_confusion(cm)


def run_training(cfg: TrainConfig, run_dir: Path, checkpoint_dir: Path, index: dict, raw_root: Path,
                 processed_root: Path) -> dict:
    """Run one experiment end to end. Returns the final status dict (also written to status.json)."""
    cfg = cfg.validate()
    io = RunIO(run_dir)
    checkpoint_dir = Path(checkpoint_dir)
    started = time.time()
    io.write_status(state="preparing", message="Initialising", epoch=0, epochs_total=cfg.epochs, progress=0.0,
                    started_at=started, has_checkpoint=False, best_val_miou=None, best_epoch=None, error=None)
    try:
        device = resolve_device(cfg.device)
        _seed_everything(cfg.seed)
        log.info("Configuration: %s", json.dumps(cfg.to_dict()))
        log.info("Device: %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")

        def cache_progress(frac: float, msg: str) -> None:
            io.write_status(force=False, message=msg)
            if io.command() == "stop":
                raise StopRequested()

        tr_img, tr_lab, tr_ids = ensure_cache(index, raw_root, processed_root, cfg.resolution, "train", cache_progress)
        va_img, va_lab, va_ids = ensure_cache(index, raw_root, processed_root, cfg.resolution, "val", cache_progress)
        if len(tr_ids) < 2 or len(va_ids) < 1:
            raise RuntimeError("The dataset index has too few train/validation pairs. Load a dataset first.")
        tr_idx = select_indices(len(tr_ids), cfg.max_train_samples, cfg.seed)
        va_idx = select_indices(len(va_ids), cfg.max_val_samples, cfg.seed)
        log.info("Training pairs: %d of %d | validation pairs: %d of %d | resolution %dpx",
                 len(tr_idx), len(tr_ids), len(va_idx), len(va_ids), cfg.resolution)

        by_id = {r["id"]: r for r in index["records"]}
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for i in tr_idx:
            counts += np.asarray(by_id[tr_ids[i]]["labels"]["train"]["counts"], dtype=np.int64)
        class_weights = class_weights_from_counts(counts.tolist(), cfg.class_weighting)
        if class_weights is not None:
            log.info("Class weights (%s): %s", cfg.class_weighting,
                     {k: round(float(w), 3) for k, w in zip(CLASS_KEYS, class_weights)})

        train_ds = TerrainDataset(tr_img, tr_lab, tr_idx, augment=cfg.augment, seed=cfg.seed)
        val_ds = TerrainDataset(va_img, va_lab, va_idx, augment=False)
        gen = torch.Generator().manual_seed(cfg.seed)
        pin = device.type == "cuda"
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=gen,
                                  num_workers=cfg.num_workers, drop_last=len(train_ds) > cfg.batch_size,
                                  pin_memory=pin)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                                pin_memory=pin)

        model = build_model(cfg.arch, cfg.model_config).to(device)
        pretrained_loaded = getattr(model, "pretrained_loaded", None)
        if cfg.arch == "deeplabv3":
            log.info("DeepLabV3 pretrained weights loaded: %s", pretrained_loaded)
        n_params = count_parameters(model)
        criterion = SegmentationLoss(cfg.dice_weight, class_weights).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        if cfg.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs,
                                                                   eta_min=cfg.learning_rate * 0.01)
        elif cfg.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
        else:
            scheduler = None
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        batches = len(train_loader)
        io.write_status(state="running", message="Training", device=str(device), parameters=n_params,
                        n_train=len(train_ds), n_val=len(val_ds), batches_per_epoch=batches,
                        pretrained_loaded=pretrained_loaded,
                        class_pixel_counts={k: int(c) for k, c in zip(CLASS_KEYS, counts)})
        log.info("Model: %s, %s parameters, %d batches/epoch", cfg.arch, f"{n_params:,}", batches)

        best_miou, best_epoch, stale = -1.0, 0, 0
        extra_base = {"train_config": cfg.to_dict(), "n_train": len(train_ds), "n_val": len(val_ds),
                      "pretrained_loaded": pretrained_loaded, "dataset_fingerprint": index.get("fingerprint")}
        final_state, final_message = "completed", "Training finished"
        for epoch in range(1, cfg.epochs + 1):
            t_epoch = time.time()
            model.train()
            running, seen = 0.0, 0
            for b, (images, labels) in enumerate(train_loader, start=1):
                io.checkpoint_gate()
                images, labels = images.to(device, non_blocking=pin), labels.to(device, non_blocking=pin)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss = criterion(model(images), labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Loss became NaN/inf. Try a lower learning rate.")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running += float(loss.detach())
                seen += 1
                done_frac = ((epoch - 1) + b / batches) / cfg.epochs
                elapsed = time.time() - started
                io.write_status(force=False, epoch=epoch, batch=b, progress=done_frac,
                                train_loss_running=running / seen, elapsed_seconds=elapsed,
                                eta_seconds=elapsed / done_frac * (1 - done_frac) if done_frac > 0 else None,
                                message=f"Epoch {epoch}/{cfg.epochs}, batch {b}/{batches}")
            train_loss = running / max(seen, 1)
            io.write_status(message=f"Epoch {epoch}/{cfg.epochs}: validating")
            val_loss, vm = validate(model, val_loader, criterion, device, io)
            lr = optimizer.param_groups[0]["lr"]
            miou = vm["mean_iou"] if vm["mean_iou"] is not None else 0.0
            row = {
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "val_pixel_accuracy": vm["pixel_accuracy"], "val_mean_iou": vm["mean_iou"],
                "val_mean_dice": vm["mean_dice"],
                "val_per_class_iou": {k: vm["per_class"][k]["iou"] for k in CLASS_KEYS},
                "learning_rate": lr, "epoch_seconds": time.time() - t_epoch,
            }
            io.append_history(row)
            extra = {**extra_base, "epoch": epoch, "val_metrics": {k: row[k] for k in (
                "val_loss", "val_pixel_accuracy", "val_mean_iou", "val_mean_dice", "val_per_class_iou")}}
            save_checkpoint(checkpoint_dir / "last.pt", model, cfg.arch, cfg.model_config, cfg.resolution, extra)
            improved = miou > best_miou
            if improved:
                best_miou, best_epoch, stale = miou, epoch, 0
                save_checkpoint(checkpoint_dir / "best.pt", model, cfg.arch, cfg.model_config, cfg.resolution, extra)
            else:
                stale += 1
            log.info("Epoch %d/%d  train_loss %.4f  val_loss %.4f  val_mIoU %.4f  val_acc %.4f  lr %.2e  %.0fs%s",
                     epoch, cfg.epochs, train_loss, val_loss, miou, vm["pixel_accuracy"] or 0.0, lr,
                     row["epoch_seconds"], "  [best]" if improved else "")
            io.write_status(epoch=epoch, progress=epoch / cfg.epochs, has_checkpoint=True, best_val_miou=best_miou,
                            best_epoch=best_epoch, last_epoch_metrics=row, elapsed_seconds=time.time() - started)
            if scheduler is not None:
                if cfg.scheduler == "plateau":
                    scheduler.step(miou)
                else:
                    scheduler.step()
            if cfg.early_stopping_patience and stale >= cfg.early_stopping_patience:
                final_message = (f"Early stopping at epoch {epoch}: validation mIoU has not improved for "
                                 f"{stale} epochs (best {best_miou:.4f} at epoch {best_epoch})")
                log.info(final_message)
                break
        io.write_status(state=final_state, message=final_message, finished_at=time.time(),
                        elapsed_seconds=time.time() - started, eta_seconds=0)
    except StopRequested:
        has = (checkpoint_dir / "last.pt").exists()
        msg = ("Stopped by user. Checkpoints from completed epochs were kept." if has
               else "Stopped by user before the first epoch finished; no checkpoint was produced.")
        log.info(msg)
        io.write_status(state="stopped", message=msg, finished_at=time.time(), has_checkpoint=has,
                        elapsed_seconds=time.time() - started, eta_seconds=None)
    except BaseException as exc:  # noqa: BLE001 - every failure must be recorded, then re-raised if fatal
        has = (checkpoint_dir / "last.pt").exists()
        if _is_oom(exc):
            msg = ("Out of memory. Reduce the batch size or input resolution, or use a lighter model "
                   "(U-Net base_channels 16, DeepLabV3 MobileNetV3).")
        else:
            msg = f"{type(exc).__name__}: {exc}"
        log.exception("Training failed: %s", msg)
        io.write_status(state="failed", message=msg, error=msg, finished_at=time.time(), has_checkpoint=has,
                        elapsed_seconds=time.time() - started, eta_seconds=None)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    return dict(io.status)
