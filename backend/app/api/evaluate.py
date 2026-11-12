"""Held-out evaluation endpoints and downloadable reports."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from ..schemas.api import EvaluateRequest, JobResponse
from ..services.evaluation_service import JOB_KIND
from ..services.model_registry import ModelNotAvailable
from .deps import Context, get_ctx

router = APIRouter(tags=["evaluation"])


@router.post("/evaluate", response_model=JobResponse, status_code=202,
             summary="Evaluate a trained model on held-out labelled data (test or validation split)")
def evaluate(req: EvaluateRequest, ctx: Context = Depends(get_ctx)) -> dict:
    try:
        return ctx.evaluation.start(req.model_id, req.split, req.label_variant, req.max_images).to_dict()
    except ModelNotAvailable as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/evaluate/status", summary="Progress of the running (or most recent) evaluation job")
def evaluate_status(ctx: Context = Depends(get_ctx)) -> dict:
    job = ctx.jobs.latest(JOB_KIND)
    return {"job": job.to_dict() if job else None}


@router.post("/evaluate/cancel", summary="Cancel the running evaluation")
def evaluate_cancel(ctx: Context = Depends(get_ctx)) -> dict:
    job = ctx.jobs.active(JOB_KIND)
    if job:
        job.cancel_event.set()
    return {"cancelled": job is not None}


@router.get("/evaluations", summary="Stored evaluation results, newest first")
def evaluations(experiment_id: str | None = Query(None, max_length=64), ctx: Context = Depends(get_ctx)) -> dict:
    return {"evaluations": ctx.evaluation.list(experiment_id)}


@router.get("/evaluations/report", summary="Download one or more evaluations as CSV or JSON")
def report(ids: str = Query(..., pattern=r"^\d+(,\d+)*$", max_length=200),
           format: Literal["csv", "json"] = "json", ctx: Context = Depends(get_ctx)) -> Response:
    id_list = [int(x) for x in ids.split(",")]
    name = f"mti_evaluation_{'_'.join(map(str, id_list))}.{format}"
    headers = {"Content-Disposition": f'attachment; filename="{name}"'}
    if format == "csv":
        return Response(ctx.evaluation.report_csv(id_list), media_type="text/csv", headers=headers)
    return Response(ctx.evaluation.report_json(id_list), media_type="application/json", headers=headers)


@router.get("/evaluations/{evaluation_id}", summary="One stored evaluation")
def evaluation(evaluation_id: int, ctx: Context = Depends(get_ctx)) -> dict:
    return ctx.evaluation.get(evaluation_id)


@router.delete("/evaluations/{evaluation_id}", summary="Delete a stored evaluation")
def delete_evaluation(evaluation_id: int, ctx: Context = Depends(get_ctx)) -> dict:
    ctx.evaluation.delete(evaluation_id)
    return {"deleted": evaluation_id}
