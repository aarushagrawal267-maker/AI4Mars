import { Eraser, Flag, Map as MapIcon, MapPin, PaintBucket, RotateCcw, Route as RouteIcon, Save, ScanSearch, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { Callout, ClassLegend, ClassSwatch, CompositionBar, EmptyState, ErrorBanner, Field, Loading, PageHeader, Panel, Segmented, Toggle } from '../components/ui';
import { useApp } from '../context/AppContext';
import { useRemote } from '../hooks/usePolling';
import { api } from '../services/api';
import type { ClassKey, PerClass, PlanResult, TerrainCost, TerrainGrid } from '../types/api';
import { CLASS_KEYS, CLASS_LABEL, archLabel, cx, fixed, int } from '../utils/format';

type Tool = 'start' | 'goal' | 'block' | 'erase';
type Cell = [number, number];
const ROUTE_STYLE: Record<string, { color: string; width: number; dash: number[] }> = {
  terrain_aware: { color: '#E8743B', width: 3.5, dash: [] },
  shortest: { color: '#F2F0EC', width: 2, dash: [6, 5] },
  alternative: { color: '#62A0DE', width: 2.5, dash: [2, 4] },
};

export default function NavigationSimulator() {
  const { connection } = useApp();
  const [params, setParams] = useSearchParams();
  const history = useRemote(() => api.analyses(60), [connection]);
  const costsRemote = useRemote(() => api.costs(), [connection]);
  const id = params.get('id') ?? history.data?.analyses[0]?.id ?? '';
  const [gridSize, setGridSize] = useState(64);
  const grid = useRemote(() => (id ? api.analysisGrid(id, gridSize) : Promise.resolve(null)), [id, gridSize]);
  const [costs, setCosts] = useState<PerClass<TerrainCost> | null>(null);
  const [start, setStart] = useState<Cell | null>(null);
  const [goal, setGoal] = useState<Cell | null>(null);
  const [blocked, setBlocked] = useState<Set<string>>(new Set());
  const [tool, setTool] = useState<Tool>('start');
  const [diagonal, setDiagonal] = useState(true);
  const [alternatives, setAlternatives] = useState(true);
  const [uncWeight, setUncWeight] = useState(0);
  const [showTerrain, setShowTerrain] = useState(true);
  const [result, setResult] = useState<PlanResult | null>(null);
  const [planning, setPlanning] = useState(false);
  const [error, setError] = useState<Error>();
  const [saved, setSaved] = useState(false);

  useEffect(() => { if (costsRemote.data && !costs) setCosts(costsRemote.data.costs); }, [costsRemote.data, costs]);
  // new terrain or resolution: reset the scenario with sensible default endpoints (bottom-centre to top-centre)
  useEffect(() => {
    const g = grid.data;
    setBlocked(new Set());
    setResult(null);
    if (!g) return;
    setStart([g.rows - 2, Math.floor(g.cols / 2)]);
    setGoal([Math.max(1, Math.floor(g.rows * 0.35)), Math.floor(g.cols * 0.8)]);
  }, [grid.data]);

  const plan = useCallback(async () => {
    if (!id || !start || !goal || !costs || (start[0] === goal[0] && start[1] === goal[1])) return;
    setPlanning(true);
    setError(undefined);
    try {
      setResult(await api.plan({ analysis_id: id, grid_size: gridSize, start, goal, costs, allow_diagonal: diagonal, alternatives,
        uncertainty_weight: uncWeight, blocked: [...blocked].map((k) => k.split(',').map(Number) as Cell) }));
    } catch (e) {
      setError(e as Error);
    } finally {
      setPlanning(false);
    }
  }, [id, gridSize, start, goal, costs, diagonal, alternatives, uncWeight, blocked]);

  // re-plan automatically (debounced) whenever the scenario changes
  useEffect(() => {
    const t = window.setTimeout(() => void plan(), 250);
    return () => window.clearTimeout(t);
  }, [plan]);

  const setCost = (k: ClassKey, patch: Partial<TerrainCost>) => {
    setSaved(false);
    setCosts((c) => (c ? { ...c, [k]: { ...c[k], ...patch } } : c));
  };

  if (history.data && history.data.analyses.length === 0) {
    return (
      <div>
        <Header />
        <EmptyState icon={MapIcon} title="No terrain to plan on yet" action={<Link to="/analysis" className="btn-primary"><ScanSearch className="h-4 w-4" /> Analyse an image</Link>}>
          The simulator plans over the segmentation of an analysed image. Run an analysis first.
        </EmptyState>
      </div>
    );
  }

  const current = history.data?.analyses.find((a) => a.id === id);
  return (
    <div>
      <Header />
      <Callout tone="warn" className="mb-6" title="Experimental simulation — not a safety assessment">
        {costsRemote.data?.disclaimer ?? 'Costs are assumptions, not validated measurements. Terrain classification alone cannot determine rover safety; this is not connected to any rover.'}
        {' '}Grid cells are image-space, not metres: perspective makes distant cells cover far more ground than near ones.
      </Callout>
      {(history.error || grid.error) && <ErrorBanner error={history.error ?? grid.error} className="mb-4" />}
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="min-w-0 space-y-6">
          <Panel ticks title="Terrain grid" icon={MapIcon}
            actions={
              <select className="field w-auto max-w-[260px] py-1 text-xs" value={id} onChange={(e) => setParams({ id: e.target.value })} aria-label="Analysis">
                {history.data?.analyses.map((a) => <option key={a.id} value={a.id}>{a.is_fixture ? '[fixture] ' : ''}{a.filename} · {archLabel(a.arch)}</option>)}
              </select>
            }>
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <Segmented ariaLabel="Tool" value={tool} onChange={setTool} options={[
                { value: 'start', label: <span className="flex items-center gap-1"><MapPin className="h-3.5 w-3.5" /> Start</span> },
                { value: 'goal', label: <span className="flex items-center gap-1"><Flag className="h-3.5 w-3.5" /> Destination</span> },
                { value: 'block', label: <span className="flex items-center gap-1"><PaintBucket className="h-3.5 w-3.5" /> Restrict</span> },
                { value: 'erase', label: <span className="flex items-center gap-1"><Eraser className="h-3.5 w-3.5" /> Erase</span> },
              ]} />
              <button className="btn-ghost text-xs" onClick={() => setBlocked(new Set())} disabled={!blocked.size}><Trash2 className="h-3.5 w-3.5" /> Clear restricted ({blocked.size})</button>
              <label className="ml-auto flex items-center gap-2 text-2xs text-ink-3"><input type="checkbox" checked={showTerrain} onChange={(e) => setShowTerrain(e.target.checked)} /> terrain colours</label>
            </div>
            {!grid.data || !current ? <Loading label="Building terrain grid…" /> : (
              <GridCanvas grid={grid.data} analysisId={id} start={start} goal={goal} blocked={blocked} result={result} showTerrain={showTerrain}
                impassable={costs ? CLASS_KEYS.filter((k) => !costs[k].passable) : []}
                onCell={(c, drag) => {
                  if (tool === 'start' && !drag) setStart(c);
                  else if (tool === 'goal' && !drag) setGoal(c);
                  else if (tool === 'block' || tool === 'erase') {
                    setBlocked((b) => {
                      const n = new Set(b);
                      const k = `${c[0]},${c[1]}`;
                      if (tool === 'block') n.add(k); else n.delete(k);
                      return n;
                    });
                  }
                }} />
            )}
            <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
              <ClassLegend />
              <div className="flex flex-wrap gap-3 text-2xs text-ink-2">
                {Object.entries(ROUTE_STYLE).map(([k, s]) => (
                  <span key={k} className="flex items-center gap-1.5">
                    <svg width="22" height="6"><line x1="0" y1="3" x2="22" y2="3" stroke={s.color} strokeWidth={s.width} strokeDasharray={s.dash.join(' ')} /></svg>
                    {k === 'terrain_aware' ? 'terrain-aware' : k}
                  </span>
                ))}
                <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 bg-[repeating-linear-gradient(45deg,#e85c54,#e85c54_2px,#1a1a1a_2px,#1a1a1a_4px)]" /> restricted / impassable</span>
              </div>
            </div>
          </Panel>

          <Panel title="Route comparison" icon={RouteIcon} padded={false} actions={planning && <span className="text-2xs text-ink-3">planning…</span>}>
            {error && <div className="p-4"><ErrorBanner error={error} /></div>}
            {!result ? <p className="p-6 text-center text-sm text-ink-3">Place a start and a destination on the grid.</p> : (
              <>
                {!result.found && <div className="p-4"><Callout tone="bad" title="No route found">{result.reason}</Callout></div>}
                <div className="overflow-x-auto">
                  <table className="table-clean">
                    <thead><tr><th>Route</th><th className="text-right">Length (cells)</th><th className="text-right">Cumulative cost</th><th className="text-right">Cost / cell</th><th className="text-right">Steps</th><th className="w-48">Terrain along route</th><th className="text-right">Nodes expanded</th></tr></thead>
                    <tbody>
                      {result.routes.filter((r) => r.found).map((r) => (
                        <tr key={r.kind} title={r.description}>
                          <td><span className="flex items-center gap-2"><svg width="18" height="6"><line x1="0" y1="3" x2="18" y2="3" stroke={ROUTE_STYLE[r.kind].color} strokeWidth={3} strokeDasharray={ROUTE_STYLE[r.kind].dash.join(' ')} /></svg>{r.label}</span></td>
                          <td className="num text-right">{fixed(r.length, 1)}</td>
                          <td className={cx('num text-right', r.kind === 'terrain_aware' && 'font-medium text-accent')}>{fixed(r.cost, 1)}</td>
                          <td className="num text-right text-ink-2">{fixed(r.mean_cost_per_unit, 2)}</td>
                          <td className="num text-right text-ink-2">{r.steps}</td>
                          <td>{r.composition && <CompositionBar percent={r.composition} height={8} />}
                            <div className="num mt-1 text-[10px] text-ink-3">{r.composition && CLASS_KEYS.filter((k) => r.composition![k] > 0).map((k) => `${CLASS_LABEL[k]} ${r.composition![k].toFixed(0)}%`).join(' · ')}</div></td>
                          <td className="num text-right text-ink-3">{int(r.expanded_nodes)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="px-4 py-3 text-2xs text-ink-3">{result.algorithm}. {result.units} The shortest route ignores terrain but is scored with the same costs; the alternative route explores a different corridor and is scored on the original costs. {result.blocked_cells} cells blocked.</p>
              </>
            )}
          </Panel>
        </div>

        <div className="space-y-6">
          <Panel title="Terrain movement costs" icon={PaintBucket} eyebrow="experimental assumptions"
            actions={<>
              <button className="btn-ghost p-1.5" title="Reset to defaults" onClick={async () => { const r = await api.resetCosts(); setCosts(r.costs); setSaved(false); }}><RotateCcw className="h-3.5 w-3.5" /></button>
              <button className="btn-ghost text-xs" disabled={!costs} onClick={async () => { if (costs) { await api.saveCosts(costs); setSaved(true); } }}><Save className="h-3.5 w-3.5" /> {saved ? 'Saved' : 'Save'}</button>
            </>}>
            {!costs ? <Loading /> : (
              <div className="space-y-4">
                {CLASS_KEYS.map((k) => (
                  <div key={k} className={cx('rounded-lg border p-3', costs[k].passable ? 'border-line bg-sunken/40' : 'border-bad/40 bg-bad/[0.05]')}>
                    <div className="flex items-center justify-between gap-2">
                      <span className="flex items-center gap-2 text-sm font-medium"><ClassSwatch k={k} size={12} /> {CLASS_LABEL[k]}</span>
                      <Toggle checked={costs[k].passable} onChange={(v) => setCost(k, { passable: v })} label={<span className="text-xs">{costs[k].passable ? 'passable' : 'impassable'}</span>} />
                    </div>
                    <div className="mt-2 flex items-center gap-3">
                      <input type="range" min={0} max={100} step={1} disabled={!costs[k].passable} aria-label={`${CLASS_LABEL[k]} cost`}
                        value={Math.round((Math.log10(costs[k].cost) + 1) / 3 * 100)}
                        onChange={(e) => setCost(k, { cost: Number((10 ** (Number(e.target.value) / 100 * 3 - 1)).toPrecision(2)) })} className="flex-1" />
                      <input type="number" className="field num w-20 py-1 text-right text-xs" min={0.1} max={1000} step={0.1} disabled={!costs[k].passable}
                        value={costs[k].cost} onChange={(e) => Number(e.target.value) > 0 && setCost(k, { cost: Number(e.target.value) })} aria-label={`${CLASS_LABEL[k]} cost value`} />
                    </div>
                  </div>
                ))}
                <p className="text-2xs leading-relaxed text-ink-3">Relative cost of crossing one cell of each predicted class (log scale 0.1–100). No class is assumed safe or unsafe by default — every class starts passable. Cells with no prediction are treated as impassable.</p>
              </div>
            )}
          </Panel>
          <Panel title="Planner settings" icon={RouteIcon}>
            <div className="space-y-4">
              <Field label={<span className="flex justify-between"><span>Grid resolution</span><span className="num text-ink">{gridSize} cells on the long side</span></span>}
                hint="Each cell takes the majority class of its pixels. Changing it resets start, destination and restrictions.">
                <input type="range" min={16} max={128} step={8} value={gridSize} onChange={(e) => setGridSize(Number(e.target.value))} className="w-full" />
              </Field>
              <Field label={<span className="flex justify-between"><span>Uncertainty penalty</span><span className="num text-ink">×(1 + {uncWeight.toFixed(1)} · entropy)</span></span>}
                hint="Inflates the cost of cells where the model was unsure of the class.">
                <input type="range" min={0} max={5} step={0.5} value={uncWeight} onChange={(e) => setUncWeight(Number(e.target.value))} className="w-full" />
              </Field>
              <Toggle checked={diagonal} onChange={setDiagonal} label="Allow diagonal moves" description="8-connected grid without corner cutting" />
              <Toggle checked={alternatives} onChange={setAlternatives} label="Compare alternative routes" description="shortest (terrain-blind) and a different corridor" />
              <button className="btn-primary w-full" onClick={() => void plan()} disabled={planning || !start || !goal}><RouteIcon className="h-4 w-4" /> Re-plan now</button>
              <div className="num grid grid-cols-2 gap-2 text-2xs text-ink-3">
                <span>start {start ? `[${start[0]}, ${start[1]}]` : '—'}</span><span>destination {goal ? `[${goal[0]}, ${goal[1]}]` : '—'}</span>
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}

function Header() {
  return (
    <PageHeader eyebrow="07 · simulation" title="Navigation Simulator"
      description="Terrain-aware A* route planning over a segmentation result. Set a start and a destination, paint restricted areas, change the assumed movement costs, and compare the resulting routes." />
  );
}

/* ------------------------------------------------------------------ canvas */

function GridCanvas({ grid, analysisId, start, goal, blocked, result, showTerrain, impassable, onCell }: {
  grid: TerrainGrid;
  analysisId: string;
  start: Cell | null;
  goal: Cell | null;
  blocked: Set<string>;
  result: PlanResult | null;
  showTerrain: boolean;
  impassable: ClassKey[];
  onCell: (c: Cell, drag: boolean) => void;
}) {
  const { colors } = useApp();
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [width, setWidth] = useState(800);
  const [hover, setHover] = useState<Cell | null>(null);
  const down = useRef(false);

  useEffect(() => {
    const i = new Image();
    i.onload = () => setImg(i);
    i.src = api.analysisImageUrl(analysisId, 'original');
  }, [analysisId]);
  useEffect(() => {
    if (!wrap.current) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(240, Math.floor(e.contentRect.width))));
    ro.observe(wrap.current);
    return () => ro.disconnect();
  }, []);

  // keep the whole grid on screen: cap the height, shrink the width to match
  const fit = Math.min(width, Math.floor((620 * grid.cols) / grid.rows));
  const height = Math.round((fit * grid.rows) / grid.cols);
  const cw = fit / grid.cols;
  const ch = height / grid.rows;
  const impassableIds = useMemo(() => new Set(impassable.map((k) => CLASS_KEYS.indexOf(k))), [impassable]);

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = fit * dpr;
    c.height = height * dpr;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, fit, height);
    ctx.fillStyle = '#0c0d10';
    ctx.fillRect(0, 0, fit, height);
    if (img) {
      ctx.globalAlpha = showTerrain ? 0.55 : 1;
      ctx.drawImage(img, 0, 0, fit, height);
      ctx.globalAlpha = 1;
    }
    const hatch = (x: number, y: number) => {
      ctx.fillStyle = 'rgba(20,20,22,0.75)';
      ctx.fillRect(x, y, cw, ch);
      ctx.strokeStyle = 'rgba(232,92,84,0.9)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, y + ch);
      ctx.lineTo(x + cw, y);
      ctx.stroke();
    };
    for (let r = 0; r < grid.rows; r++) {
      for (let q = 0; q < grid.cols; q++) {
        const cls = grid.cells[r][q];
        const x = q * cw;
        const y = r * ch;
        if (cls < 0 || impassableIds.has(cls)) {
          hatch(x, y);
          continue;
        }
        if (showTerrain) {
          ctx.fillStyle = colors[CLASS_KEYS[cls]];
          ctx.globalAlpha = 0.5;
          ctx.fillRect(x, y, cw, ch);
          ctx.globalAlpha = 1;
        }
      }
    }
    blocked.forEach((k) => {
      const [r, q] = k.split(',').map(Number);
      hatch(q * cw, r * ch);
    });
    if (cw >= 7) {
      ctx.strokeStyle = 'rgba(0,0,0,0.22)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let q = 1; q < grid.cols; q++) { ctx.moveTo(q * cw, 0); ctx.lineTo(q * cw, height); }
      for (let r = 1; r < grid.rows; r++) { ctx.moveTo(0, r * ch); ctx.lineTo(fit, r * ch); }
      ctx.stroke();
    }
    // routes: draw secondary first, terrain-aware on top, each with a dark casing for legibility
    const order = ['shortest', 'alternative', 'terrain_aware'];
    [...(result?.routes ?? [])].filter((r) => r.found).sort((a, b) => order.indexOf(a.kind) - order.indexOf(b.kind)).forEach((route) => {
      const st = ROUTE_STYLE[route.kind];
      const pts = route.path.map(([r, q]) => [q * cw + cw / 2, r * ch + ch / 2]);
      for (const pass of [0, 1]) {
        ctx.beginPath();
        pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.setLineDash(pass ? st.dash : []);
        ctx.strokeStyle = pass ? st.color : 'rgba(0,0,0,0.65)';
        ctx.lineWidth = pass ? st.width : st.width + 3;
        ctx.stroke();
      }
      ctx.setLineDash([]);
    });
    const marker = (cell: Cell | null, fill: string, label: string) => {
      if (!cell) return;
      const x = cell[1] * cw + cw / 2;
      const y = cell[0] * ch + ch / 2;
      const rad = Math.max(7, Math.min(cw, ch) * 0.7);
      ctx.beginPath();
      ctx.arc(x, y, rad, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#0b0c0e';
      ctx.stroke();
      ctx.fillStyle = '#0b0c0e';
      ctx.font = `600 ${Math.max(9, rad)}px "JetBrains Mono", monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, x, y + 0.5);
    };
    marker(start, '#46BE8C', 'S');
    marker(goal, '#F2F0EC', 'D');
    if (hover) {
      ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(hover[1] * cw + 0.5, hover[0] * ch + 0.5, cw - 1, ch - 1);
    }
  }, [grid, img, fit, height, cw, ch, colors, blocked, result, start, goal, hover, showTerrain, impassableIds]);

  const cellAt = (e: React.PointerEvent): Cell | null => {
    const rect = canvas.current?.getBoundingClientRect();
    if (!rect) return null;
    const q = Math.floor(((e.clientX - rect.left) / rect.width) * grid.cols);
    const r = Math.floor(((e.clientY - rect.top) / rect.height) * grid.rows);
    return r >= 0 && r < grid.rows && q >= 0 && q < grid.cols ? [r, q] : null;
  };
  const hoverCls = hover ? grid.cells[hover[0]][hover[1]] : null;

  return (
    <div ref={wrap} className="relative">
      <canvas ref={canvas} style={{ width: fit, height }} className="mx-auto block cursor-crosshair touch-none rounded-lg border border-line" data-testid="nav-canvas"
        onPointerDown={(e) => { down.current = true; const c = cellAt(e); if (c) onCell(c, false); (e.target as HTMLElement).setPointerCapture?.(e.pointerId); }}
        onPointerMove={(e) => { const c = cellAt(e); setHover(c); if (down.current && c) onCell(c, true); }}
        onPointerUp={() => (down.current = false)}
        onPointerLeave={() => { setHover(null); down.current = false; }} />
      {hover && (
        <div className="pointer-events-none absolute left-1/2 top-2 -translate-x-1/2 rounded-md border border-white/10 bg-black/75 px-2 py-1 font-mono text-[11px] text-white">
          [{hover[0]}, {hover[1]}] · {hoverCls !== null && hoverCls >= 0 ? CLASS_LABEL[CLASS_KEYS[hoverCls]] : 'no prediction'}
          {grid.uncertainty && ` · entropy ${grid.uncertainty[hover[0]][hover[1]].toFixed(2)}`}
          {blocked.has(`${hover[0]},${hover[1]}`) && ' · restricted'}
        </div>
      )}
    </div>
  );
}
