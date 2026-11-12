"""Terrain-aware A* route planning over a grid of terrain classes.

This is an *experimental simulation*. Movement costs are user-defined assumptions,
not measurements of rover mobility, and a terrain label alone says nothing
definitive about safety.

Cost model
----------
Each traversable cell has a cost ``c`` (the cost assigned to its terrain class,
optionally inflated by prediction uncertainty). Moving between neighbouring
cells ``a`` and ``b`` costs ``distance * (c_a + c_b) / 2`` where ``distance`` is 1
for straight moves and sqrt(2) for diagonal moves. The heuristic is the octile
distance multiplied by the smallest traversable cell cost, which never
overestimates, so A* returns a minimum-cost path.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

SQRT2 = math.sqrt(2.0)
_STRAIGHT = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIAGONAL = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


@dataclass
class Route:
    found: bool
    path: list[tuple[int, int]]
    cost: float | None
    length: float | None
    expanded: int


def build_cost_grid(classes: np.ndarray, class_costs: list[float | None], unknown_cost: float | None = None,
                    blocked: list[tuple[int, int]] | None = None, uncertainty: np.ndarray | None = None,
                    uncertainty_weight: float = 0.0) -> np.ndarray:
    """Per-cell cost array; ``inf`` marks cells that cannot be entered.

    ``classes`` holds class ids (0..n-1) or -1 for unknown terrain. ``class_costs[i]`` is the
    cost for class ``i`` or None when that class is marked impassable by the user.
    """
    classes = np.asarray(classes)
    cost = np.full(classes.shape, np.inf, dtype=np.float64)
    for cid, value in enumerate(class_costs):
        if value is not None:
            if value <= 0:
                raise ValueError("Terrain costs must be positive")
            cost[classes == cid] = float(value)
    if unknown_cost is not None:
        cost[classes < 0] = float(unknown_cost)
    if uncertainty is not None and uncertainty_weight > 0:
        cost = cost * (1.0 + uncertainty_weight * np.clip(np.asarray(uncertainty, dtype=np.float64), 0, 1))
    for r, c in blocked or []:
        if 0 <= r < cost.shape[0] and 0 <= c < cost.shape[1]:
            cost[r, c] = np.inf
    return cost


def astar(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int], allow_diagonal: bool = True) -> Route:
    """Minimum-cost path from ``start`` to ``goal`` on a cell-cost grid (``inf`` = blocked)."""
    rows, cols = cost.shape
    for name, (r, c) in (("start", start), ("goal", goal)):
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(f"{name} is outside the grid")
    if not np.isfinite(cost[start]) or not np.isfinite(cost[goal]):
        return Route(False, [], None, None, 0)
    finite = cost[np.isfinite(cost)]
    min_cost = float(finite.min()) if finite.size else 1.0
    moves = [(dr, dc, 1.0) for dr, dc in _STRAIGHT]
    if allow_diagonal:
        moves += [(dr, dc, SQRT2) for dr, dc in _DIAGONAL]

    def h(r: int, c: int) -> float:
        dr, dc = abs(r - goal[0]), abs(c - goal[1])
        dist = (max(dr, dc) + (SQRT2 - 1) * min(dr, dc)) if allow_diagonal else (dr + dc)
        return dist * min_cost

    g = np.full(cost.shape, np.inf)
    parent = np.full((rows, cols, 2), -1, dtype=np.int32)
    closed = np.zeros(cost.shape, dtype=bool)
    g[start] = 0.0
    heap: list[tuple[float, float, int, int]] = [(h(*start), 0.0, start[0], start[1])]
    expanded = 0
    while heap:
        _f, gc, r, c = heapq.heappop(heap)
        if closed[r, c]:
            continue
        closed[r, c] = True
        expanded += 1
        if (r, c) == goal:
            break
        for dr, dc, dist in moves:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or closed[nr, nc] or not np.isfinite(cost[nr, nc]):
                continue
            # No corner cutting: a diagonal move needs both adjacent straight cells to be enterable.
            if dr and dc and (not np.isfinite(cost[r + dr, c]) or not np.isfinite(cost[r, c + dc])):
                continue
            ng = gc + dist * (cost[r, c] + cost[nr, nc]) / 2.0
            if ng < g[nr, nc]:
                g[nr, nc] = ng
                parent[nr, nc] = (r, c)
                heapq.heappush(heap, (ng + h(nr, nc), ng, nr, nc))
    if not closed[goal]:
        return Route(False, [], None, None, expanded)
    path = [goal]
    while path[-1] != start:
        pr, pc = parent[path[-1]]
        path.append((int(pr), int(pc)))
    path.reverse()
    return Route(True, path, float(g[goal]), path_length(path), expanded)


def path_length(path: list[tuple[int, int]]) -> float:
    """Geometric length in cell units (diagonal steps count sqrt(2))."""
    return float(sum(SQRT2 if (a[0] != b[0] and a[1] != b[1]) else 1.0 for a, b in zip(path, path[1:])))


def path_cost(path: list[tuple[int, int]], cost: np.ndarray) -> float | None:
    """Cost of a given path under a cost grid; None when it crosses a blocked cell."""
    total = 0.0
    for a, b in zip(path, path[1:]):
        if not np.isfinite(cost[a]) or not np.isfinite(cost[b]):
            return None
        dist = SQRT2 if (a[0] != b[0] and a[1] != b[1]) else 1.0
        total += dist * (cost[a] + cost[b]) / 2.0
    return float(total)


def path_composition(path: list[tuple[int, int]], classes: np.ndarray, n_classes: int) -> list[float]:
    """Percentage of path cells per class (unknown cells excluded from the denominator)."""
    counts = np.zeros(n_classes, dtype=np.int64)
    for r, c in path:
        cid = int(classes[r, c])
        if 0 <= cid < n_classes:
            counts[cid] += 1
    total = counts.sum()
    return [round(100.0 * float(x) / total, 2) if total else 0.0 for x in counts]


def alternative_route(cost: np.ndarray, primary: list[tuple[int, int]], start: tuple[int, int],
                      goal: tuple[int, int], allow_diagonal: bool = True, radius: int = 2,
                      penalty: float = 4.0) -> Route:
    """A distinct second route: penalise cells near the primary path, re-plan, then re-score honestly.

    The returned cost is evaluated on the ORIGINAL cost grid, so the two routes are comparable.
    """
    penalised = cost.copy()
    rows, cols = cost.shape
    near = np.zeros(cost.shape, dtype=bool)
    for r, c in primary[1:-1]:
        near[max(0, r - radius):min(rows, r + radius + 1), max(0, c - radius):min(cols, c + radius + 1)] = True
    # Keep the neighbourhood of start and goal unpenalised so every route can leave/arrive.
    for r, c in (start, goal):
        near[max(0, r - radius):min(rows, r + radius + 1), max(0, c - radius):min(cols, c + radius + 1)] = False
    penalised[near & np.isfinite(penalised)] *= penalty
    alt = astar(penalised, start, goal, allow_diagonal)
    if not alt.found:
        return alt
    return Route(True, alt.path, path_cost(alt.path, cost), alt.length, alt.expanded)


def overlap_fraction(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> float:
    """Share of cells of ``b`` that also lie on ``a``."""
    if not b:
        return 0.0
    sa = set(a)
    return sum(1 for cell in b if cell in sa) / len(b)
