# Route-planning simulator

> **Experimental simulation only.** The terrain costs are illustrative assumptions chosen for
> demonstration, not validated measurements of rover mobility. Terrain class alone cannot determine
> whether a rover can drive somewhere safely: slope, rock height, wheel sinkage, rover state and
> mission rules all matter, and none of them is modelled. The simulator is not connected to any rover
> and its output must not be used for real navigation decisions.

Code: `backend/app/navigation/astar.py` (algorithm), `backend/app/navigation/planner.py` (route
comparison), `ml/inference/terrain.py` (`class_grid`), endpoint `POST /api/navigation/plan`.

## From a prediction to a grid

The planner works on a stored image analysis (a model prediction). `class_grid` divides the predicted
class map into `rows x cols` cells, where the image's long side has `grid_size` cells (8-160, default
64) and the short side is scaled proportionally. Each cell takes the **majority vote** of the
predicted classes of the pixels inside it. A cell with no classified pixel would be marked unknown
(-1); with model predictions every pixel has a class, so this occurs only in degenerate cases. The
mean per-pixel uncertainty (normalised entropy) of each cell is kept as well.

## Cost model

Each cell gets a cost from its class:

| Class | Default cost | Default passable |
|---|---|---|
| soil | 1 | yes |
| bedrock | 2 | yes |
| sand | 5 | yes |
| big rock | 20 | yes |

These defaults are **illustrative**. They can be changed and saved (`PUT /api/navigation/costs`,
cost > 0 and at most 1000), passed per request, or reset (`POST /api/navigation/costs/reset`).

Cells that cannot be entered get infinite cost:

- **Impassable classes:** any class the user marks as not passable;
- **User-blocked cells:** cells listed in the request's `blocked` list (drawn in the UI);
- **Unknown cells:** blocked by default (the API does not offer a way to make them passable).

**Uncertainty-weighted cost inflation:** with `uncertainty_weight` w (0-10, default 0), each cell's
cost is multiplied by `1 + w * u`, where u is the cell's mean normalised entropy in [0, 1]. Routes then
prefer cells where the model was more certain. This is a heuristic, not a risk model.

**Moves:** 8-connected by default (straight and diagonal), or 4-connected with
`allow_diagonal: false`. Moving from cell a to neighbouring cell b costs

```
step cost = d x (cost(a) + cost(b)) / 2,    d = 1 (straight) or sqrt(2) (diagonal)
```

**No corner cutting:** a diagonal move is allowed only if both cells it passes between (the two
straight neighbours) can be entered, so a route never slips diagonally between two blocked cells.

## A* search

`astar` is a standard A* search (Hart, Nilsson and Raphael, 1968) with a binary heap and a closed set.

- **Heuristic:** the octile distance to the goal (Manhattan distance when diagonal moves are off)
  multiplied by the smallest finite cell cost on the grid.
- **Admissibility:** every move of geometric length d costs at least d x (minimum cell cost), and the
  octile distance is the shortest possible geometric length on an 8-connected grid, so the heuristic
  never overestimates the remaining cost. It is also consistent (it satisfies the triangle inequality
  move by move), so A* with a closed set returns a **minimum-cost path** under the cost model. The
  test suite checks optimality against an independent Dijkstra implementation on a seeded random
  grid with mixed costs and blocked cells.
- If the start or goal cell is not enterable, or the goal is enclosed by blocked cells, no route is
  returned and the response gives the reason. Start and goal must be inside the grid and different.

The number of expanded nodes is reported for each route.

## Routes returned

| Route | How it is computed |
|---|---|
| **Terrain-aware** | A* on the cost grid: minimum cumulative terrain cost under the current settings |
| **Shortest (ignores terrain)** | Baseline: A* where every enterable cell costs 1 (same obstacles), then re-scored with the terrain costs so its cost is comparable |
| **Alternative** | Cells within 2 cells of the terrain-aware route (except near start and goal) get their cost multiplied by 4; A* is run again, and the resulting path is re-scored on the original cost grid. It is shown only if less than 85 % of its cells lie on the terrain-aware route |

For each route the response lists the path (grid cells), total cost, geometric length in cells,
number of steps, expanded nodes, the percentage of path cells per terrain class and the mean cost per
unit length, plus the costs used, the blocked-cell count, the algorithm and the disclaimer.

## Limitations of this simulation

- **Costs are assumptions.** No value in the cost table was derived from rover data. Different
  plausible choices change the route.
- **Terrain class is not safety.** A soil cell can be a steep slope; a bedrock cell can hide a
  step; sand may or may not be hazardous depending on depth and rover state. None of this is visible
  to a 4-class segmentation.
- **Predictions can be wrong.** The grid is built from model output, not ground truth. Errors,
  especially on the rare big-rock class, propagate directly into the plan.
- **Grid cells are image space, not metres.** NavCam images are perspective views: a cell near the
  bottom of the image covers a small patch of nearby ground, while a cell near the top can cover many
  metres of distant terrain (and labels beyond about 30 m were masked in training). Path length in
  cells is therefore not a distance on the ground, and a straight line in the image is generally not
  a straight line on the ground. No camera model, range data or terrain elevation is used.
- **Not everything in the image is terrain.** The model assigns a terrain class to every pixel,
  including sky and parts of the rover, so cells there are "passable" to the planner. Block them
  manually if a route crosses them.
- **Not connected to a rover.** The simulator only draws paths on an image.
