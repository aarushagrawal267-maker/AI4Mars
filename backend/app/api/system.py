"""Health, hardware diagnostics and the aggregated overview."""
from __future__ import annotations

import os
import platform
import shutil
import sys
import time

import psutil
import torch
import torchvision
from fastapi import APIRouter, Depends

from ..core.config import APP_NAME, APP_VERSION
from ..database.models import Analysis, Evaluation, Experiment
from ..schemas.api import HealthResponse
from .deps import Context, get_ctx

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
def health() -> HealthResponse:
    return HealthResponse(status="ok", app=APP_NAME, version=APP_VERSION, time=time.time())


def hardware() -> dict:
    cuda = torch.cuda.is_available()
    gpus = []
    if cuda:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            gpus.append({"index": i, "name": props.name, "memory_total_gb": round(total / 1e9, 2),
                         "memory_free_gb": round(free / 1e9, 2)})
    vm = psutil.virtual_memory()
    return {
        "cuda_available": cuda, "gpus": gpus,
        "selected_device": "cuda" if cuda else "cpu",
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores_logical": os.cpu_count(), "cpu_cores_physical": psutil.cpu_count(logical=False),
        "torch_threads": torch.get_num_threads(),
        "ram_total_gb": round(vm.total / 1e9, 2), "ram_available_gb": round(vm.available / 1e9, 2),
        "diagnosis": ("CUDA GPU available: training and inference will use it by default." if cuda else
                      "No CUDA GPU detected by PyTorch: everything runs on the CPU. Use the small-subset "
                      "training defaults; full-dataset training on CPU takes many hours."),
    }


@router.get("/system/info", summary="Versions, hardware and storage locations")
def system_info(ctx: Context = Depends(get_ctx)) -> dict:
    s = ctx.settings
    disk = shutil.disk_usage(s.data_dir)
    with ctx.db.session() as db:
        counts = {"experiments": db.query(Experiment).filter(Experiment.is_fixture.is_(False)).count(),
                  "evaluations": db.query(Evaluation).count(), "analyses": db.query(Analysis).count()}
    return {
        "app": APP_NAME, "version": APP_VERSION,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "hardware": hardware(),
        "paths": {"data_dir": str(s.data_dir), "dataset_dir": str(s.raw_dir), "imports_dir": str(s.imports_dir),
                  "processed_dir": str(s.processed_dir), "model_dir": str(s.checkpoint_dir),
                  "experiments_dir": str(s.experiments_dir), "database": s.db_url},
        "disk_free_gb": round(disk.free / 1e9, 2),
        "limits": {"max_upload_mb": s.max_upload_mb, "max_output_side": s.max_output_side},
        "counts": counts,
        "test_fixture_enabled": s.enable_test_fixture,
    }


@router.get("/overview", summary="Everything the Overview dashboard shows, from real state only")
def overview(ctx: Context = Depends(get_ctx)) -> dict:
    models = ctx.registry.list_models()
    evaluations = ctx.evaluation.list()
    with ctx.db.session() as db:
        latest = db.query(Experiment).filter(Experiment.is_fixture.is_(False)) \
            .order_by(Experiment.created_at.desc()).first()
        latest_exp = None if latest is None else {
            "id": latest.id, "name": latest.name, "arch": latest.arch, "status": latest.status,
            "created_at": latest.created_at.isoformat(), "epochs_completed": latest.epochs_completed,
            "epochs_total": latest.config.get("epochs"), "best_val_miou": latest.best_val_miou,
            "message": latest.message}
    analyses = ctx.inference.list(8)
    training = ctx.training.status()
    training.pop("history", None)
    ds = ctx.dataset.status()
    return {
        "dataset": {k: ds[k] for k in ("dataset_found", "indexed", "total_images", "split_counts", "job")},
        "models": models,
        "trained_model_count": sum(1 for m in models if not m["is_fixture"]),
        "latest_experiment": latest_exp,
        "training": training,
        "latest_analysis": analyses[0] if analyses else None,
        "recent_analyses": analyses,
        "evaluations": [{"id": e["id"], "experiment_id": e["experiment_id"], "experiment_name": e["experiment_name"],
                         "arch": e["arch"], "split": e["split"], "label_variant": e["label_variant"],
                         "images": e["images"], "created_at": e["created_at"], "is_fixture": e["is_fixture"],
                         "mean_iou": e["metrics"].get("mean_iou"),
                         "pixel_accuracy": e["metrics"].get("pixel_accuracy"),
                         "mean_dice": e["metrics"].get("mean_dice")} for e in evaluations[:6]],
        "hardware": hardware(),
    }
