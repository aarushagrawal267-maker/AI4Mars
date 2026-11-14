/** Terrain-aware A* planner, ported from backend/app/navigation/{astar,planner}.py with identical rules. */
import type { ClassKey, PerClass, PlanResult, Route, TerrainCost } from '../types/api';

const KEYS: ClassKey[] = ['soil', 'bedrock', 'sand', 'big_rock'];
const SQRT2 = Math.SQRT2;
type Cell = [number, number];
interface RawRoute { found: boolean; path: Cell[]; cost: number | null; length: number | null; expanded: number }

export const DISCLAIMER = 'Experimental simulation. Costs are user-set assumptions, not validated measurements of rover mobility. '
  + 'Terrain classification alone cannot determine actual rover safety, and this simulator is not connected to any physical rover.';

type Item = [number, number, number, number];
/** Python heapq ordering: tuples compared element by element. */
const lt = (x: Item, y: Item) => x[0] !== y[0] ? x[0] < y[0] : x[1] !== y[1] ? x[1] < y[1] : x[2] !== y[2] ? x[2] < y[2] : x[3] < y[3];

class Heap {
  private a: Item[] = [];
  get size() { return this.a.length; }
  push(v: Item) {
    const a = this.a;
    a.push(v);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!lt(a[i], a[p])) break;
      [a[i], a[p]] = [a[p], a[i]];
      i = p;
    }
  }
  pop(): Item {
    const a = this.a, top = a[0], last = a.pop()!;
    if (a.length) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < a.length && lt(a[l], a[m])) m = l;
        if (r < a.length && lt(a[r], a[m])) m = r;
        if (m === i) break;
        [a[i], a[m]] = [a[m], a[i]];
        i = m;
      }
    }
    return top;
  }
}

function buildCost(grid: number[][], classCosts: (number | null)[], blocked: Cell[], unc: number[][] | null, uw: number): Float64Array[] {
  const cost = grid.map((row) => Float64Array.from(row, (cls) => (cls >= 0 && classCosts[cls] !== null ? (classCosts[cls] as number) : Infinity)));
  if (unc && uw > 0) cost.forEach((row, r) => row.forEach((v, c) => { row[c] = v * (1 + uw * Math.min(1, Math.max(0, unc[r][c]))); }));
  for (const [r, c] of blocked) if (r >= 0 && r < cost.length && c >= 0 && c < cost[0].length) cost[r][c] = Infinity;
  return cost;
}

function astar(cost: Float64Array[], start: Cell, goal: Cell, diagonal: boolean): RawRoute {
  const rows = cost.length, cols = cost[0].length;
  if (!Number.isFinite(cost[start[0]][start[1]]) || !Number.isFinite(cost[goal[0]][goal[1]])) return { found: false, path: [], cost: null, length: null, expanded: 0 };
  let minCost = Infinity;
  cost.forEach((row) => row.forEach((v) => { if (Number.isFinite(v) && v < minCost) minCost = v; }));
  if (!Number.isFinite(minCost)) minCost = 1;
  const moves: [number, number, number][] = [[-1, 0, 1], [1, 0, 1], [0, -1, 1], [0, 1, 1]];
  if (diagonal) moves.push([-1, -1, SQRT2], [-1, 1, SQRT2], [1, -1, SQRT2], [1, 1, SQRT2]);
  const h = (r: number, c: number) => {
    const dr = Math.abs(r - goal[0]), dc = Math.abs(c - goal[1]);
    return (diagonal ? Math.max(dr, dc) + (SQRT2 - 1) * Math.min(dr, dc) : dr + dc) * minCost;
  };
  const g = new Float64Array(rows * cols).fill(Infinity);
  const parent = new Int32Array(rows * cols).fill(-1);
  const closed = new Uint8Array(rows * cols);
  const id = (r: number, c: number) => r * cols + c;
  g[id(...start)] = 0;
  const heap = new Heap();
  heap.push([h(...start), 0, start[0], start[1]]);
  let expanded = 0;
  while (heap.size) {
    const [, gc, r, c] = heap.pop();
    if (closed[id(r, c)]) continue;
    closed[id(r, c)] = 1;
    expanded++;
    if (r === goal[0] && c === goal[1]) break;
    for (const [dr, dc, dist] of moves) {
      const nr = r + dr, nc = c + dc;
      if (nr < 0 || nr >= rows || nc < 0 || nc >= cols || closed[id(nr, nc)] || !Number.isFinite(cost[nr][nc])) continue;
      if (dr && dc && (!Number.isFinite(cost[r + dr][c]) || !Number.isFinite(cost[r][c + dc]))) continue;
      const ng = gc + (dist * (cost[r][c] + cost[nr][nc])) / 2;
      if (ng < g[id(nr, nc)]) {
        g[id(nr, nc)] = ng;
        parent[id(nr, nc)] = id(r, c);
        heap.push([ng + h(nr, nc), ng, nr, nc]);
      }
    }
  }
  if (!closed[id(...goal)]) return { found: false, path: [], cost: null, length: null, expanded };
  const path: Cell[] = [goal];
  while (path[path.length - 1][0] !== start[0] || path[path.length - 1][1] !== start[1]) {
    const p = parent[id(...path[path.length - 1])];
    path.push([Math.floor(p / cols), p % cols]);
  }
  path.reverse();
  return { found: true, path, cost: g[id(...goal)], length: pathLength(path), expanded };
}

const step = (a: Cell, b: Cell) => (a[0] !== b[0] && a[1] !== b[1] ? SQRT2 : 1);
const pathLength = (p: Cell[]) => p.slice(1).reduce((s, b, i) => s + step(p[i], b), 0);
function pathCost(p: Cell[], cost: Float64Array[]): number | null {
  let total = 0;
  for (let i = 1; i < p.length; i++) {
    const a = cost[p[i - 1][0]][p[i - 1][1]], b = cost[p[i][0]][p[i][1]];
    if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
    total += (step(p[i - 1], p[i]) * (a + b)) / 2;
  }
  return total;
}

function describeRoute(rt: RawRoute, grid: number[][], kind: Route['kind'], label: string, description: string): Route {
  if (!rt.found) return { kind, label, description, found: false, path: [], cost: null, length: null, steps: 0, expanded_nodes: rt.expanded, composition: null, mean_cost_per_unit: null };
  const counts = [0, 0, 0, 0];
  rt.path.forEach(([r, c]) => { const v = grid[r][c]; if (v >= 0 && v < 4) counts[v]++; });
  const total = counts.reduce((a, b) => a + b, 0);
  const composition = Object.fromEntries(KEYS.map((k, i) => [k, total ? Math.round((10000 * counts[i]) / total) / 100 : 0])) as PerClass<number>;
  const round = (v: number, d: number) => Math.round(v * 10 ** d) / 10 ** d;
  return {
    kind, label, description, found: true, path: rt.path.map(([r, c]) => [r, c]),
    cost: rt.cost === null ? null : round(rt.cost, 4), length: round(rt.length ?? 0, 4), steps: rt.path.length - 1,
    expanded_nodes: rt.expanded, composition,
    mean_cost_per_unit: rt.cost !== null && rt.length ? round(rt.cost / rt.length, 4) : null,
  };
}

export function plan(grid: number[][], costs: PerClass<TerrainCost>, start: Cell, goal: Cell, blocked: Cell[], diagonal: boolean,
  unc: number[][] | null, uw: number, alternatives: boolean): PlanResult {
  const classCosts = KEYS.map((k) => (costs[k].passable ? costs[k].cost : null));
  const cost = buildCost(grid, classCosts, blocked, unc, uw);
  const primary = astar(cost, start, goal, diagonal);
  const routes = [describeRoute(primary, grid, 'terrain_aware', 'Terrain-aware route', 'Minimum cumulative terrain cost under the current cost settings.')];
  let reason: string | null = null;
  if (!primary.found) {
    if (!Number.isFinite(cost[start[0]][start[1]])) reason = 'The start cell is on impassable or restricted terrain.';
    else if (!Number.isFinite(cost[goal[0]][goal[1]])) reason = 'The destination cell is on impassable or restricted terrain.';
    else reason = 'No route exists: the destination is enclosed by impassable or restricted cells.';
  } else if (alternatives) {
    const uniform = cost.map((row) => Float64Array.from(row, (v) => (Number.isFinite(v) ? 1 : Infinity)));
    const sh = astar(uniform, start, goal, diagonal);
    if (sh.found) routes.push(describeRoute({ ...sh, cost: pathCost(sh.path, cost) }, grid, 'shortest', 'Shortest route (ignores terrain)', 'Shortest geometric path around obstacles, scored with the same terrain costs.'));
    const rows = cost.length, cols = cost[0].length, radius = 2;
    const near = cost.map(() => new Uint8Array(cols));
    const mark = (r: number, c: number, v: number) => {
      for (let y = Math.max(0, r - radius); y < Math.min(rows, r + radius + 1); y++) for (let x = Math.max(0, c - radius); x < Math.min(cols, c + radius + 1); x++) near[y][x] = v;
    };
    primary.path.slice(1, -1).forEach(([r, c]) => mark(r, c, 1));
    [start, goal].forEach(([r, c]) => mark(r, c, 0));
    const pen = cost.map((row, r) => Float64Array.from(row, (v, c) => (near[r][c] && Number.isFinite(v) ? v * 4 : v)));
    const alt = astar(pen, start, goal, diagonal);
    if (alt.found) {
      const onPrimary = new Set(primary.path.map(([r, c]) => `${r},${c}`));
      const overlap = alt.path.filter(([r, c]) => onPrimary.has(`${r},${c}`)).length / alt.path.length;
      if (overlap < 0.85) routes.push(describeRoute({ ...alt, cost: pathCost(alt.path, cost) }, grid, 'alternative', 'Alternative route', 'A different corridor found by penalising cells near the terrain-aware route.'));
    }
  }
  let blockedCells = 0;
  cost.forEach((row) => row.forEach((v) => { if (!Number.isFinite(v)) blockedCells++; }));
  return {
    rows: grid.length, cols: grid[0].length, start, goal, found: primary.found, reason, routes, blocked_cells: blockedCells,
    impassable_classes: KEYS.filter((k) => !costs[k].passable), algorithm: diagonal ? 'A* (8-connected, octile heuristic)' : 'A* (4-connected, Manhattan heuristic)',
    units: 'Length is in grid cells; cost is length weighted by the assumed terrain cost.', disclaimer: DISCLAIMER,
  };
}
