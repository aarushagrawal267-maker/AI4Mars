"""Models and experiment history."""
from __future__ import annotations

import csv
import io
import json
import shutil
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response

from ..core.config import APP_VERSION
from ..database.models import Experiment
from ..schemas.api import ExperimentPatch
from .deps import Context, get_ctx

router = APIRouter(tags=["experiments"])


def _summary(exp: Experiment, ctx: Context) -> dict:
    last = exp.history[-1] if exp.history else None
    return {
        "id": exp.id, "name": exp.name, "arch": exp.arch, "status": exp.status, "is_fixture": exp.is_fixture,
        "created_at": exp.created_at.isoformat(),
        "finished_at": exp.finished_at.isoformat() if exp.finished_at else None,
        "config": exp.config, "epochs_completed": exp.epochs_completed, "best_val_miou": exp.best_val_miou,
        "best_epoch": exp.best_epoch, "device": exp.device, "parameters": exp.parameters,
        "n_train": exp.n_train, "n_val": exp.n_val, "elapsed_seconds": exp.elapsed_seconds,
        "pretrained_loaded": exp.pretrained_loaded, "dataset_fingerprint": exp.dataset_fingerprint,
        "message": exp.message, "error": exp.error, "notes": exp.notes,
        "has_checkpoint": ctx.training.has_checkpoint(exp.id),
        "final_train_loss": last["train_loss"] if last else None,
        "final_val_loss": last["val_loss"] if last else None,
        "evaluation_count": len(exp.evaluations),
    }


@router.get("/models", summary="Trained checkpoints available for inference and evaluation")
def models(ctx: Context = Depends(get_ctx)) -> dict:
    return {"models": ctx.registry.list_models()}


@router.get("/experiments", summary="All training experiments, newest first (including failed and stopped runs)")
def experiments(ctx: Context = Depends(get_ctx)) -> dict:
    with ctx.db.session() as s:
        rows = s.query(Experiment).order_by(Experiment.created_at.desc()).all()
        return {"experiments": [_summary(e, ctx) for e in rows]}


def _detail(exp_id: str, ctx: Context) -> dict:
    with ctx.db.session() as s:
        exp = s.get(Experiment, exp_id)
        if exp is None:
            raise HTTPException(404, "Experiment not found")
        d = _summary(exp, ctx)
        d["history"] = exp.history
    d["evaluations"] = ctx.evaluation.list(exp_id)
    return d


@router.get("/experiments/{exp_id}", summary="Configuration, per-epoch history and evaluations of one experiment")
def experiment(exp_id: str, ctx: Context = Depends(get_ctx)) -> dict:
    return _detail(exp_id, ctx)


@router.patch("/experiments/{exp_id}", summary="Rename an experiment or edit its notes")
def patch_experiment(exp_id: str, body: ExperimentPatch, ctx: Context = Depends(get_ctx)) -> dict:
    with ctx.db.session() as s:
        exp = s.get(Experiment, exp_id)
        if exp is None:
            raise HTTPException(404, "Experiment not found")
        if body.name is not None:
            exp.name = body.name
        if body.notes is not None:
            exp.notes = body.notes
    return _detail(exp_id, ctx)


@router.delete("/experiments/{exp_id}", summary="Delete an experiment with its checkpoints and evaluations")
def delete_experiment(exp_id: str, ctx: Context = Depends(get_ctx)) -> dict:
    with ctx.db.session() as s:
        exp = s.get(Experiment, exp_id)
        if exp is None:
            raise HTTPException(404, "Experiment not found")
        if ctx.training.is_running() and ctx.training.status().get("experiment_id") == exp_id:
            raise HTTPException(409, "This experiment is currently training. Stop it first.")
        s.delete(exp)
    ctx.registry.forget(exp_id)
    # exp_id was resolved through the database above, so these paths are application-generated.
    shutil.rmtree(ctx.training.run_dir(exp_id), ignore_errors=True)
    shutil.rmtree(ctx.training.checkpoint_dir(exp_id), ignore_errors=True)
    return {"deleted": exp_id}


@router.get("/experiments/{exp_id}/report", summary="Download an experiment report (JSON, or CSV of the epoch history)")
def experiment_report(exp_id: str, format: Literal["json", "csv"] = "json",
                      ctx: Context = Depends(get_ctx)) -> Response:
    d = _detail(exp_id, ctx)
    if format == "json":
        body = json.dumps({"application": "Mars Terrain Intelligence", "version": APP_VERSION, "experiment": d},
                          indent=2)
        return Response(body, media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="experiment_{exp_id}.json"'})
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["experiment_id", "arch", "epoch", "train_loss", "val_loss", "val_pixel_accuracy", "val_mean_iou",
                "val_mean_dice", "val_iou_soil", "val_iou_bedrock", "val_iou_sand", "val_iou_big_rock",
                "learning_rate", "epoch_seconds"])
    for h in d["history"]:
        pc = h.get("val_per_class_iou", {})
        w.writerow([exp_id, d["arch"], h["epoch"], h["train_loss"], h["val_loss"], h["val_pixel_accuracy"],
                    h["val_mean_iou"], h["val_mean_dice"], pc.get("soil"), pc.get("bedrock"), pc.get("sand"),
                    pc.get("big_rock"), h["learning_rate"], h["epoch_seconds"]])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="experiment_{exp_id}_history.csv"'})
