"""Experimental terrain-aware route planning."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..navigation import planner
from ..schemas.api import ColorsRequest, CostsRequest, NavigationPlanRequest
from ..services.settings_service import COST_DISCLAIMER, DEFAULT_COSTS
from .deps import Context, get_ctx

router = APIRouter(tags=["navigation & settings"])


@router.get("/navigation/costs", summary="Saved terrain movement costs (experimental assumptions)")
def get_costs(ctx: Context = Depends(get_ctx)) -> dict:
    return {"costs": ctx.prefs.costs(), "defaults": DEFAULT_COSTS, "disclaimer": COST_DISCLAIMER}


@router.put("/navigation/costs", summary="Save terrain movement costs")
def put_costs(req: CostsRequest, ctx: Context = Depends(get_ctx)) -> dict:
    return {"costs": ctx.prefs.set_costs(req.model_dump()), "defaults": DEFAULT_COSTS, "disclaimer": COST_DISCLAIMER}


@router.post("/navigation/costs/reset", summary="Restore the default illustrative costs")
def reset_costs(ctx: Context = Depends(get_ctx)) -> dict:
    return {"costs": ctx.prefs.reset_costs(), "defaults": DEFAULT_COSTS, "disclaimer": COST_DISCLAIMER}


@router.post("/navigation/plan", summary="Plan a route with A* over the terrain grid of an analysis")
def plan(req: NavigationPlanRequest, ctx: Context = Depends(get_ctx)) -> dict:
    grid = ctx.inference.grid(req.analysis_id, req.grid_size)
    rows, cols = grid["rows"], grid["cols"]
    for name, (r, c) in (("start", req.start), ("goal", req.goal)):
        if not (0 <= r < rows and 0 <= c < cols):
            raise HTTPException(422, f"{name} [{r}, {c}] is outside the {rows} x {cols} grid.")
    if tuple(req.start) == tuple(req.goal):
        raise HTTPException(422, "Start and destination must be different cells.")
    costs = req.costs.model_dump() if req.costs else ctx.prefs.costs()
    result = planner.plan(grid["cells"], costs, tuple(req.start), tuple(req.goal), blocked=list(req.blocked),
                          allow_diagonal=req.allow_diagonal, uncertainty=grid["uncertainty"],
                          uncertainty_weight=req.uncertainty_weight, alternatives=req.alternatives)
    result["analysis_id"] = req.analysis_id
    result["grid_size"] = req.grid_size
    return result


@router.get("/settings/colors", summary="Display colours per terrain class (class ids never change)")
def get_colors(ctx: Context = Depends(get_ctx)) -> dict:
    from ml.classes import CLASS_KEYS, CLASS_LABELS, DEFAULT_COLORS, IGNORE_INDEX, UNLABELED_COLOR
    return {"colors": ctx.prefs.colors(), "defaults": DEFAULT_COLORS, "unlabeled": UNLABELED_COLOR,
            "classes": [{"id": i, "key": k, "label": CLASS_LABELS[i]} for i, k in enumerate(CLASS_KEYS)],
            "ignore_index": IGNORE_INDEX}


@router.put("/settings/colors", summary="Change display colours")
def put_colors(req: ColorsRequest, ctx: Context = Depends(get_ctx)) -> dict:
    try:
        ctx.prefs.set_colors({k: v for k, v in req.model_dump().items() if v})
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return get_colors(ctx)


@router.post("/settings/colors/reset", summary="Restore the default palette")
def reset_colors(ctx: Context = Depends(get_ctx)) -> dict:
    ctx.prefs.reset_colors()
    return get_colors(ctx)
