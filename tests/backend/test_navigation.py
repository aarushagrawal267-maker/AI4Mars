"""A* route planning tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

from backend.app.navigation import astar as A
from backend.app.navigation.planner import plan

COSTS = {"soil": {"cost": 1.0, "passable": True}, "bedrock": {"cost": 2.0, "passable": True},
         "sand": {"cost": 5.0, "passable": True}, "big_rock": {"cost": 20.0, "passable": True}}


def test_straight_path_on_uniform_grid():
    cost = np.ones((5, 9))
    r = A.astar(cost, (2, 0), (2, 8))
    assert r.found and r.path[0] == (2, 0) and r.path[-1] == (2, 8)
    assert r.length == pytest.approx(8.0) and r.cost == pytest.approx(8.0)
    d = A.astar(np.ones((6, 6)), (0, 0), (5, 5))
    assert d.length == pytest.approx(5 * math.sqrt(2))
    m = A.astar(np.ones((6, 6)), (0, 0), (5, 5), allow_diagonal=False)
    assert m.length == pytest.approx(10.0)


def test_path_is_connected_and_avoids_blocked_cells():
    cost = np.ones((9, 9))
    cost[1:8, 4] = np.inf                                  # a wall with gaps at the top and bottom rows
    r = A.astar(cost, (4, 0), (4, 8))
    assert r.found and all(np.isfinite(cost[c]) for c in r.path)
    for a, b in zip(r.path, r.path[1:]):
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1
    assert r.length > 8.0


def test_no_corner_cutting_between_diagonal_obstacles():
    cost = np.ones((3, 3))
    cost[0, 1] = cost[1, 0] = np.inf
    assert not A.astar(cost, (0, 0), (2, 2)).found          # (0,0) is sealed in: the diagonal squeeze is not allowed


def test_unreachable_goal_and_blocked_endpoints():
    cost = np.ones((5, 5))
    cost[:, 2] = np.inf
    assert not A.astar(cost, (2, 0), (2, 4)).found
    assert not A.astar(cost, (2, 2), (2, 4)).found
    with pytest.raises(ValueError):
        A.astar(cost, (0, 0), (9, 9))


def test_route_is_optimal_against_dijkstra():
    rng = np.random.default_rng(3)
    cost = rng.choice([1.0, 2.0, 5.0, 20.0], size=(14, 17))
    cost[rng.random(cost.shape) < 0.12] = np.inf
    cost[0, 0] = cost[13, 16] = 1.0
    r = A.astar(cost, (0, 0), (13, 16))
    import heapq
    dist = np.full(cost.shape, np.inf)
    dist[0, 0] = 0
    heap = [(0.0, 0, 0)]
    while heap:                                             # reference: plain Dijkstra with the same move rules
        d, y, x = heapq.heappop(heap)
        if d > dist[y, x]:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if (dy == dx == 0) or not (0 <= ny < 14 and 0 <= nx < 17) or not np.isfinite(cost[ny, nx]):
                    continue
                if dy and dx and (not np.isfinite(cost[y + dy, x]) or not np.isfinite(cost[y, x + dx])):
                    continue
                nd = d + (math.sqrt(2) if dy and dx else 1.0) * (cost[y, x] + cost[ny, nx]) / 2
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(heap, (nd, ny, nx))
    if np.isfinite(dist[13, 16]):
        assert r.found and r.cost == pytest.approx(dist[13, 16])
        assert A.path_cost(r.path, cost) == pytest.approx(r.cost)
    else:
        assert not r.found


def test_terrain_costs_change_the_route():
    grid = np.zeros((7, 11), dtype=int)
    grid[2:5, 3:8] = 2                                      # a sand patch directly between start and goal
    res = plan(grid.tolist(), COSTS, (3, 0), (3, 10))
    aware = next(r for r in res["routes"] if r["kind"] == "terrain_aware")
    shortest = next(r for r in res["routes"] if r["kind"] == "shortest")
    assert aware["found"] and aware["composition"]["sand"] == 0.0     # detours around the sand
    assert shortest["composition"]["sand"] > 0 and shortest["length"] < aware["length"]
    assert aware["cost"] < shortest["cost"]
    cheap_sand = {**COSTS, "sand": {"cost": 1.0, "passable": True}}
    flat = plan(grid.tolist(), cheap_sand, (3, 0), (3, 10))["routes"][0]
    assert flat["length"] == pytest.approx(10.0)             # with equal costs the straight line wins
    assert "not connected" in res["disclaimer"] and res["algorithm"].startswith("A*")


def test_impassable_class_blocked_cells_and_unknown_terrain():
    grid = np.zeros((5, 9), dtype=int)
    grid[:, 4] = 3
    walled = {**COSTS, "big_rock": {"cost": 20.0, "passable": False}}
    res = plan(grid.tolist(), walled, (2, 0), (2, 8))
    assert res["found"] is False and "enclosed" in res["reason"] and res["impassable_classes"] == ["big_rock"]
    assert plan(grid.tolist(), COSTS, (2, 0), (2, 8))["found"] is True
    blocked = plan(np.zeros((5, 9), dtype=int).tolist(), COSTS, (2, 0), (2, 8), blocked=[(r, 4) for r in range(5)])
    assert blocked["found"] is False and blocked["blocked_cells"] == 5
    unknown = np.zeros((3, 5), dtype=int)
    unknown[:, 2] = -1
    assert plan(unknown.tolist(), COSTS, (1, 0), (1, 4))["found"] is False     # unknown terrain is not assumed drivable
    assert plan(unknown.tolist(), COSTS, (1, 0), (1, 4), unknown_passable=True)["found"] is True
    on_block = plan(grid.tolist(), walled, (2, 4), (2, 8))
    assert on_block["found"] is False and "start" in on_block["reason"].lower()


def test_alternative_route_and_uncertainty_penalty():
    grid = np.zeros((15, 15), dtype=int)
    res = plan(grid.tolist(), COSTS, (7, 0), (7, 14))
    kinds = [r["kind"] for r in res["routes"]]
    assert "alternative" in kinds
    alt = next(r for r in res["routes"] if r["kind"] == "alternative")
    assert alt["cost"] >= res["routes"][0]["cost"] - 1e-9    # never better than the optimum on the true costs
    unc = np.zeros((5, 9))
    unc[2, 3:6] = 1.0
    plain = plan(np.zeros((5, 9), dtype=int).tolist(), COSTS, (2, 0), (2, 8), alternatives=False)["routes"][0]
    careful = plan(np.zeros((5, 9), dtype=int).tolist(), COSTS, (2, 0), (2, 8), uncertainty=unc.tolist(),
                   uncertainty_weight=5.0, alternatives=False)["routes"][0]
    assert [2, 4] in plain["path"] and [2, 4] not in careful["path"]
    with pytest.raises(ValueError):
        A.build_cost_grid(np.zeros((2, 2), dtype=int), [0.0, 1, 1, 1])
