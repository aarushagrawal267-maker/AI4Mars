"""Training control endpoints. Runs execute in a separate process; progress is polled."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ml.training.config import ALLOWED_RESOLUTIONS, TrainConfig

from ..schemas.api import TrainRequest, TrainStartRequest
from ..services.training_manager import TrainingError
from .deps import Context, get_ctx

router = APIRouter(prefix="/training", tags=["training"])


def _config(req: TrainRequest) -> TrainConfig:
    try:
        return TrainConfig.from_dict(req.to_config_dict())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/defaults", summary="Hardware-aware suggested configuration and allowed option values")
def defaults(ctx: Context = Depends(get_ctx)) -> dict:
    d = ctx.training.defaults()
    d["options"] = {
        "architectures": [
            {"id": "unet", "label": "U-Net", "model_options": {"base_channels": [16, 32, 64], "depth": [3, 4, 5]}},
            {"id": "deeplabv3", "label": "DeepLabV3",
             "model_options": {"backbone": ["mobilenet_v3_large", "resnet50"], "pretrained": ["coco", "none"]}},
        ],
        "resolutions": list(ALLOWED_RESOLUTIONS),
        "devices": ["auto", "cpu"] + (["cuda"] if d["cuda_available"] else []),
    }
    return d


@router.post("/estimate", summary="Measured workload estimate for a configuration (nothing is started)")
def estimate(req: TrainRequest, ctx: Context = Depends(get_ctx)) -> dict:
    try:
        return ctx.training.estimate(_config(req))
    except (TrainingError, RuntimeError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/start", status_code=202, summary="Start a training run in the background")
def start(req: TrainStartRequest, ctx: Context = Depends(get_ctx)) -> dict:
    cfg = _config(req)
    try:
        return ctx.training.start(cfg, req.name, req.confirm_large)
    except TrainingError as exc:
        raise HTTPException(409, str(exc)) from exc


def _control(fn) -> dict:
    try:
        fn()
    except TrainingError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True}


@router.post("/stop", summary="Stop the active run after the current batch; completed checkpoints are kept")
def stop(ctx: Context = Depends(get_ctx)) -> dict:
    return _control(ctx.training.stop)


@router.post("/pause", summary="Pause the active run between batches")
def pause(ctx: Context = Depends(get_ctx)) -> dict:
    return _control(ctx.training.pause)


@router.post("/resume", summary="Resume a paused run")
def resume(ctx: Context = Depends(get_ctx)) -> dict:
    return _control(ctx.training.resume)


@router.get("/status", summary="Progress, losses and validation metrics of the active (or most recent) run")
def status(ctx: Context = Depends(get_ctx)) -> dict:
    return ctx.training.status()


@router.get("/logs", summary="Tail of the training log")
def logs(experiment_id: str | None = Query(None, max_length=64), tail: int = Query(200, ge=1, le=2000),
         ctx: Context = Depends(get_ctx)) -> dict:
    return ctx.training.logs(experiment_id, tail)
