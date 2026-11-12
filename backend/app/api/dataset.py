"""Dataset acquisition, validation and browsing endpoints."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from ..schemas.api import ClassKey, DatasetDownloadRequest, DatasetImportRequest, JobResponse
from .deps import Context, get_ctx

router = APIRouter(prefix="/dataset", tags=["dataset"])
PNG = "image/png"
CACHE = {"Cache-Control": "private, max-age=300"}


@router.get("/status", summary="What is on disk, whether it is indexed, and any running dataset job")
def status(ctx: Context = Depends(get_ctx)) -> dict:
    return ctx.dataset.status()


@router.post("/load", response_model=JobResponse, status_code=202,
             summary="Scan the dataset directory, validate every image/label pair and build the index")
def load(ctx: Context = Depends(get_ctx)) -> dict:
    return ctx.dataset.start_scan().to_dict()


@router.post("/download", response_model=JobResponse, status_code=202,
             summary="Download AI4Mars (MSL NavCam) from the verified Zenodo record, then index it")
def download(req: DatasetDownloadRequest, ctx: Context = Depends(get_ctx)) -> dict:
    if req.mode == "full" and not req.confirm_full:
        raise HTTPException(400, "The full archive is about 16 GB. Set confirm_full to true to proceed.")
    return ctx.dataset.start_download(req.mode, req.train_count).to_dict()


@router.post("/import", response_model=JobResponse, status_code=202,
             summary="Extract a ZIP from data/raw/imports (e.g. a Zenodo or Kaggle download), then index it")
def import_archive(req: DatasetImportRequest, ctx: Context = Depends(get_ctx)) -> dict:
    return ctx.dataset.start_import(req.archive).to_dict()


@router.post("/cancel", summary="Cancel the running dataset job")
def cancel(ctx: Context = Depends(get_ctx)) -> dict:
    return {"cancelled": ctx.dataset.cancel()}


@router.get("/statistics", summary="Per-split image counts, class distribution and unlabelled percentage")
def statistics(ctx: Context = Depends(get_ctx)) -> dict:
    idx = ctx.dataset.require_index()
    return {"created": idx["created"], "fingerprint": idx["fingerprint"], "classes": idx["classes"],
            "ignore_index": idx["ignore_index"], "split_config": idx["split_config"],
            "statistics": idx["statistics"], "leakage_check": idx["leakage_check"], "layout": idx["layout"]}


@router.get("/validation", summary="Validation report: missing pairs, unreadable files, unexpected label values")
def validation(ctx: Context = Depends(get_ctx)) -> dict:
    idx = ctx.dataset.require_index()
    return {"created": idx["created"], "scan_seconds": idx["scan_seconds"], "validation": idx["validation"],
            "leakage_check": idx["leakage_check"]}


@router.get("/structure", summary="Folder tree of the dataset directory with file counts")
def structure(ctx: Context = Depends(get_ctx)) -> dict:
    return {"root": str(ctx.settings.raw_dir), "tree": ctx.dataset.structure()}


@router.get("/samples", summary="Page through indexed image/label pairs")
def samples(split: Literal["train", "val", "test"] | None = None, offset: int = Query(0, ge=0),
            limit: int = Query(12, ge=1, le=60), contains: ClassKey | None = None,
            ctx: Context = Depends(get_ctx)) -> dict:
    return ctx.dataset.samples(split, offset, limit, contains)


@router.get("/samples/{sample_id}", summary="Metadata for one sample")
def sample(sample_id: str, ctx: Context = Depends(get_ctx)) -> dict:
    return ctx.dataset.sample(sample_id)


@router.get("/samples/{sample_id}/{kind}", summary="Render a sample: image, colour-coded label, or overlay",
            responses={200: {"content": {PNG: {}}}})
def sample_render(sample_id: str, kind: Literal["image", "label", "overlay"], variant: str | None = None,
                  size: int | None = Query(None, ge=64, le=2048), alpha: float = Query(0.5, ge=0, le=1),
                  ctx: Context = Depends(get_ctx)) -> Response:
    data = ctx.dataset.render(sample_id, kind, variant, size, ctx.prefs.colors(), alpha)
    return Response(data, media_type=PNG, headers=CACHE)
