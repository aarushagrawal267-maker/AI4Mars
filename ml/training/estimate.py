"""Workload estimate shown before a training run starts.

The time figure is *measured*, not guessed: we build the requested model and time
a few real optimisation steps at the requested batch size and resolution on the
requested device, then scale by the number of steps. It is still an estimate
(data loading, validation and thermal throttling vary), and is labelled as such.
"""
from __future__ import annotations

import time

import torch

from ..architectures.factory import build_model, count_parameters, resolve_device
from ..classes import NUM_CLASSES
from .config import TrainConfig
from .losses import SegmentationLoss


def estimate_workload(cfg: TrainConfig, n_train_available: int, n_val_available: int, steps: int = 2) -> dict:
    cfg = cfg.validate()
    device = resolve_device(cfg.device)
    n_train = min(n_train_available, cfg.max_train_samples or n_train_available)
    n_val = min(n_val_available, cfg.max_val_samples or n_val_available)
    drop_last = n_train > cfg.batch_size
    batches = n_train // cfg.batch_size if drop_last else max(1, -(-n_train // cfg.batch_size))
    val_batches = max(1, -(-n_val // cfg.batch_size)) if n_val else 0

    # Timing never needs pretrained weights, so skip the download here.
    model = build_model(cfg.arch, cfg.model_config, for_inference=True).to(device)
    params = count_parameters(model)
    criterion = SegmentationLoss(cfg.dice_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(cfg.batch_size, 3, cfg.resolution, cfg.resolution, device=device)
    y = torch.randint(0, NUM_CLASSES, (cfg.batch_size, cfg.resolution, cfg.resolution), device=device)
    error = None
    train_step = val_step = None
    try:
        model.train()
        times = []
        for i in range(steps + 1):
            t = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            criterion(model(x), y).backward()
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i > 0:  # first step is warm-up
                times.append(time.perf_counter() - t)
        train_step = sum(times) / len(times)
        model.eval()
        with torch.no_grad():
            t = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            val_step = time.perf_counter() - t
    except (RuntimeError, MemoryError) as exc:
        if "out of memory" in str(exc).lower() or isinstance(exc, MemoryError):
            error = "This configuration does not fit in memory on the selected device. Lower batch size or resolution."
        else:
            raise
    finally:
        del model, opt, x, y
        if device.type == "cuda":
            torch.cuda.empty_cache()

    epoch_seconds = None if train_step is None else batches * train_step + val_batches * (val_step or 0)
    total = None if epoch_seconds is None else epoch_seconds * cfg.epochs
    return {
        "device": str(device),
        "parameters": params,
        "train_samples": n_train, "val_samples": n_val,
        "train_samples_available": n_train_available, "val_samples_available": n_val_available,
        "batches_per_epoch": batches, "total_steps": batches * cfg.epochs,
        "pixels_per_epoch": n_train * cfg.resolution * cfg.resolution,
        "seconds_per_step": train_step,
        "estimated_seconds_per_epoch": epoch_seconds,
        "estimated_total_seconds": total,
        "large_job": bool(total is not None and total > 2 * 3600),
        "fits_in_memory": error is None,
        "error": error,
        "note": "Measured by timing real optimisation steps on this machine; excludes one-off dataset "
                "caching and may run faster if early stopping triggers.",
    }
