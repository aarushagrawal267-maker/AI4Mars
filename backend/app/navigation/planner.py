"""High-level route planning: builds the cost grid from a terrain grid and compares routes."""
from __future__ import annotations

import numpy as np

from ml.classes import CLASS_KEYS, NUM_CLASSES

from . import astar as A

DISCLAIMER = (
    "Experimental simulation. Costs are user-set assumptions, not validated measurements of rover mobility. "
    "Terrain classification alone cannot determine actual rover safety, and this simulator is not connected "
    "to any physical rover."
)


def _describe(route: A.Route, classes: np.ndarray, kind: str, label: str, description: str) -> dict:
    if not route.found:
        return {"kind": kind, "label": label, "description": description, "found": False, "path": [],
                "cost": None, "length": None, "steps": 0, "expanded_nodes": route.expanded, "composition": None,
                "mean_cost_per_unit": None}
    comp = A.path_composition(route.path, classes, NUM_CLASSES)
    return {
        "kind": kind, "label": label, "description": description, "found": True,
        "path": [[int(r), int(c)] for r, c in route.path],
        "cost": None if route.cost is None else round(route.cost, 4),
        "length": round(route.length or 0.0, 4), "steps": len(route.path) - 1,
        "expanded_nodes": route.expanded,
        "composition": {k: comp[i] for i, k in enumerate(CLASS_KEYS)},
        "mean_cost_per_unit": round(route.cost / route.length, 4) if route.cost is not None and route.length else None,
    }


def plan(classes: list[list[int]], costs: dict[str, dict], start: tuple[int, int], goal: tuple[int, int],
         blocked: list[tuple[int, int]] | None = None, allow_diagonal: bool = True,
         uncertainty: list[list[float]] | None = None, uncertainty_weight: float = 0.0,
         unknown_passable: bool = False, unknown_cost: float = 10.0, alternatives: bool = True) -> dict:
    """Plan a terrain-aware route plus, for comparison, a distance-only route and an alternative route."""
    grid = np.asarray(classes, dtype=np.int16)
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError("The terrain grid must be a non-empty 2-D array")
    class_costs = [float(costs[k]["cost"]) if costs[k]["passable"] else None for k in CLASS_KEYS]
    unc = np.asarray(uncertainty, dtype=np.float64) if uncertainty is not None else None
    cost = A.build_cost_grid(grid, class_costs, unknown_cost if unknown_passable else None, blocked, unc,
                             uncertainty_weight)
    primary = A.astar(cost, start, goal, allow_diagonal)
    routes = [_describe(primary, grid, "terrain_aware", "Terrain-aware route",
                        "Minimum cumulative terrain cost under the current cost settings.")]
    reason = None
    if not primary.found:
        if not np.isfinite(cost[start]):
            reason = "The start cell is on impassable or restricted terrain."
        elif not np.isfinite(cost[goal]):
            reason = "The destination cell is on impassable or restricted terrain."
        else:
            reason = "No route exists: the destination is enclosed by impassable or restricted cells."
    elif alternatives:
        # Distance-only baseline: same obstacles, every traversable cell costs the same.
        uniform = np.where(np.isfinite(cost), 1.0, np.inf)
        shortest = A.astar(uniform, start, goal, allow_diagonal)
        if shortest.found:
            shortest = A.Route(True, shortest.path, A.path_cost(shortest.path, cost), shortest.length,
                               shortest.expanded)
            routes.append(_describe(shortest, grid, "shortest", "Shortest route (ignores terrain)",
                                    "Shortest geometric path around obstacles, scored with the same terrain costs."))
        alt = A.alternative_route(cost, primary.path, start, goal, allow_diagonal)
        if alt.found and A.overlap_fraction(primary.path, alt.path) < 0.85:
            routes.append(_describe(alt, grid, "alternative", "Alternative route",
                                    "A different corridor found by penalising cells near the terrain-aware route."))
    impassable = [k for k in CLASS_KEYS if not costs[k]["passable"]]
    return {
        "rows": int(grid.shape[0]), "cols": int(grid.shape[1]),
        "start": [int(start[0]), int(start[1])], "goal": [int(goal[0]), int(goal[1])],
        "found": primary.found, "reason": reason, "routes": routes,
        "blocked_cells": int((~np.isfinite(cost)).sum()),
        "impassable_classes": impassable,
        "costs_used": costs, "allow_diagonal": allow_diagonal, "uncertainty_weight": uncertainty_weight,
        "algorithm": "A* (8-connected, octile heuristic)" if allow_diagonal else "A* (4-connected, Manhattan heuristic)",
        "units": "Length is in grid cells; cost is length weighted by the assumed terrain cost.",
        "disclaimer": DISCLAIMER,
    }
