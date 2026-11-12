"""Image upload, local model inference, and stored analyses."""
from __future__ import annotations

import os
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile

from ml.preprocessing.transforms import ImageValidationError, decode_upload

from ..services.inference_service import RENDER_KINDS
from ..services.model_registry import ModelNotAvailable
from .deps import Context, get_ctx

router = APIRouter(tags=["inference"])
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/pjpeg", "application/octet-stream", ""}


@router.post("/predict", summary="Segment an uploaded image (or a dataset sample) with a trained local model")
async def predict(model_id: str = Form(..., max_length=64), file: UploadFile | None = File(None),
                  sample_id: str | None = Form(None, max_length=120), ctx: Context = Depends(get_ctx)) -> dict:
    if (file is None) == (sample_id is None):
        raise HTTPException(400, "Provide exactly one of: an image file, or a dataset sample_id.")
    if file is not None:
        name = os.path.basename(file.filename or "upload")
        if os.path.splitext(name)[1].lower() not in ALLOWED_EXT:
            raise HTTPException(415, "Unsupported file type. Upload a JPG, JPEG or PNG image.")
        if (file.content_type or "") not in ALLOWED_TYPES:
            raise HTTPException(415, f"Unsupported content type {file.content_type!r}.")
        limit = ctx.settings.max_upload_mb * 1024 * 1024
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"File is larger than the {ctx.settings.max_upload_mb} MB upload limit.")
        try:
            rgb = np.asarray(decode_upload(data))
        except ImageValidationError as exc:
            raise HTTPException(400, str(exc)) from exc
        source = "upload"
    else:
        try:
            gray = ctx.dataset.image_array(sample_id)
        except (KeyError, LookupError) as exc:
            raise HTTPException(404, "Unknown dataset sample.") from exc
        rgb, name, source = np.repeat(gray[:, :, None], 3, axis=2), f"{sample_id}.JPG", "dataset"
    try:
        return ctx.inference.analyze(rgb, model_id, name, source, sample_id)
    except ModelNotAvailable as exc:
        raise HTTPException(404, str(exc)) from exc
    except MemoryError as exc:
        raise HTTPException(507, str(exc)) from exc


@router.get("/analyses", summary="Recent image analyses")
def analyses(limit: int = Query(50, ge=1, le=200), ctx: Context = Depends(get_ctx)) -> dict:
    return {"analyses": ctx.inference.list(limit)}


@router.get("/analyses/{analysis_id}", summary="Full statistics for one analysis")
def analysis(analysis_id: str, ctx: Context = Depends(get_ctx)) -> dict:
    return ctx.inference.get(analysis_id)


@router.delete("/analyses/{analysis_id}", summary="Delete an analysis and its stored images")
def delete_analysis(analysis_id: str, ctx: Context = Depends(get_ctx)) -> dict:
    ctx.inference.delete(analysis_id)
    return {"deleted": analysis_id}


@router.get("/analyses/{analysis_id}/image/{kind}", responses={200: {"content": {"image/png": {}}}},
            summary="Render an analysis output: original, mask, overlay, side_by_side, confidence, uncertainty, class_ids")
def analysis_image(analysis_id: str, kind: Literal[RENDER_KINDS], alpha: float = Query(0.5, ge=0, le=1),  # type: ignore[valid-type]
                   download: bool = False, ctx: Context = Depends(get_ctx)) -> Response:
    data = ctx.inference.render(analysis_id, kind, ctx.prefs.colors(), alpha)
    headers = {"Cache-Control": "no-store"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="mti_{analysis_id}_{kind}.png"'
    return Response(data, media_type="image/png", headers=headers)


@router.get("/analyses/{analysis_id}/grid", summary="Prediction downsampled to a terrain grid (majority vote per cell)")
def analysis_grid(analysis_id: str, size: int = Query(64, ge=8, le=160), ctx: Context = Depends(get_ctx)) -> dict:
    return ctx.inference.grid(analysis_id, size)
