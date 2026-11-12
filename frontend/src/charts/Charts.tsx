import type { ReactNode } from 'react';
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, LabelList, Legend, Line, LineChart, ResponsiveContainer, Tooltip,
  XAxis, YAxis,
} from 'recharts';
import { useApp } from '../context/AppContext';
import type { ClassKey, HistoryRow, PerClass, TerrainStats } from '../types/api';
import { CLASS_KEYS, CLASS_LABEL, cx, fixed, int, ratio } from '../utils/format';

/** Fixed series identities (never assigned by rank). */
export const SERIES = {
  train: '#62A0DE',
  val: '#E8743B',
  unet: '#E8743B',
  deeplabv3: '#62A0DE',
  neutral: '#8A9099',
};
export const archColor = (arch: string) => (arch === 'deeplabv3' ? SERIES.deeplabv3 : SERIES.unet);
/** Additional fixed hues for comparing more than two experiments. */
export const MULTI = ['#E8743B', '#62A0DE', '#46BE8C', '#D9A23A', '#C66BD6', '#8A9099'];

function TooltipBox({ title, rows }: { title?: ReactNode; rows: { color?: string; label: ReactNode; value: ReactNode }[] }) {
  return (
    <div className="min-w-[150px] rounded-lg border border-line-strong bg-surface/95 px-3 py-2 text-xs shadow-xl backdrop-blur">
      {title !== undefined && <div className="mb-1.5 font-mono text-2xs uppercase tracking-wider text-ink-3">{title}</div>}
      <div className="space-y-1">
        {rows.map((r, i) => (
          <div key={i} className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5 text-ink-2">
              {r.color && <span className="h-2 w-2 rounded-sm" style={{ background: r.color }} />}
              {r.label}
            </span>
            <span className="num text-ink">{r.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TT = { active?: boolean; payload?: any[]; label?: any };

/* ----------------------------------------------------------- composition */

/** Horizontal bars, one per class, direct-labelled with the percentage. */
export function ClassBars({ percent, height = 180, unit = '%', valueDigits = 1 }: {
  percent: Partial<PerClass<number | null>>;
  height?: number;
  unit?: string;
  valueDigits?: number;
}) {
  const { colors } = useApp();
  const data = CLASS_KEYS.map((k) => ({ key: k, name: CLASS_LABEL[k], value: percent[k] ?? 0 }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 56, bottom: 4, left: 4 }} barCategoryGap={8}>
        <CartesianGrid horizontal={false} strokeDasharray="0" />
        <XAxis type="number" domain={[0, (max: number) => Math.max(5, Math.ceil(max / 10) * 10)]} tickLine={false} axisLine={false}
          tickFormatter={(v) => `${v}${unit}`} />
        <YAxis type="category" dataKey="name" width={70} tickLine={false} axisLine={false} />
        <Tooltip cursor={{ fill: 'rgba(127,127,127,0.06)' }} content={({ active, payload }: TT) =>
          active && payload?.length ? (
            <TooltipBox rows={[{ color: colors[payload[0].payload.key as ClassKey], label: payload[0].payload.name,
              value: `${Number(payload[0].value).toFixed(2)}${unit}` }]} />
          ) : null} />
        <Bar dataKey="value" radius={[0, 4, 4, 0]} isAnimationActive={false}>
          {data.map((d) => <Cell key={d.key} fill={colors[d.key]} />)}
          <LabelList dataKey="value" position="right" className="num" fill="currentColor"
            formatter={(v: number) => `${v.toFixed(valueDigits)}${unit}`} style={{ fill: 'rgb(var(--ink-2))', fontSize: 11, fontFamily: 'JetBrains Mono' }} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Ring chart of the class composition with the dominant class in the centre. */
export function CompositionRing({ percent, size = 168, center }: {
  percent: PerClass<number>;
  size?: number;
  center?: ReactNode;
}) {
  const { colors } = useApp();
  const r = size / 2 - 10;
  const c = 2 * Math.PI * r;
  let offset = 0;
  const gap = 2; // px surface gap between segments
  const parts = CLASS_KEYS.map((k) => ({ k, v: percent[k] ?? 0 })).filter((p) => p.v > 0);
  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90" role="img" aria-label="Terrain composition">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgb(var(--sunken))" strokeWidth={14} />
        {parts.map((p) => {
          const len = (p.v / 100) * c;
          const dash = Math.max(0, len - (parts.length > 1 ? gap : 0));
          const el = (
            <circle key={p.k} cx={size / 2} cy={size / 2} r={r} fill="none" stroke={colors[p.k]} strokeWidth={14}
              strokeDasharray={`${dash} ${c - dash}`} strokeDashoffset={-offset}>
              <title>{`${CLASS_LABEL[p.k]} ${p.v.toFixed(2)}%`}</title>
            </circle>
          );
          offset += len;
          return el;
        })}
      </svg>
      <div className="absolute inset-0 grid place-items-center text-center">{center}</div>
    </div>
  );
}

/* --------------------------------------------------------------- training */

export function LossChart({ history, height = 240 }: { history: HistoryRow[]; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={history} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="epoch" tickLine={false} axisLine={false} allowDecimals={false} />
        <YAxis tickLine={false} axisLine={false} width={44} domain={['auto', 'auto']} tickFormatter={(v) => Number(v).toFixed(2)} />
        <Tooltip content={({ active, payload, label }: TT) => active && payload?.length ? (
          <TooltipBox title={`epoch ${label}`} rows={payload.map((p) => ({ color: p.color, label: p.name, value: fixed(p.value, 4) }))} />
        ) : null} />
        <Legend iconType="plainline" wrapperStyle={{ paddingTop: 6 }} />
        <Line type="monotone" dataKey="train_loss" name="Train loss" stroke={SERIES.train} strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} isAnimationActive={false} />
        <Line type="monotone" dataKey="val_loss" name="Validation loss" stroke={SERIES.val} strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function ValMetricChart({ history, height = 240 }: { history: HistoryRow[]; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={history} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="epoch" tickLine={false} axisLine={false} allowDecimals={false} />
        <YAxis tickLine={false} axisLine={false} width={44} domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
        <Tooltip content={({ active, payload, label }: TT) => active && payload?.length ? (
          <TooltipBox title={`epoch ${label}`} rows={payload.map((p) => ({ color: p.color, label: p.name, value: ratio(p.value, 2) }))} />
        ) : null} />
        <Legend iconType="plainline" wrapperStyle={{ paddingTop: 6 }} />
        <Line type="monotone" dataKey="val_mean_iou" name="Val mIoU" stroke={SERIES.val} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
        <Line type="monotone" dataKey="val_pixel_accuracy" name="Val pixel accuracy" stroke={SERIES.train} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function PerClassIouHistory({ history, height = 220 }: { history: HistoryRow[]; height?: number }) {
  const { colors } = useApp();
  const data = history.map((h) => ({ epoch: h.epoch, ...h.val_per_class_iou }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="epoch" tickLine={false} axisLine={false} allowDecimals={false} />
        <YAxis tickLine={false} axisLine={false} width={44} domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
        <Tooltip content={({ active, payload, label }: TT) => active && payload?.length ? (
          <TooltipBox title={`epoch ${label} · val IoU`} rows={payload.map((p) => ({ color: p.color, label: p.name, value: ratio(p.value, 1) }))} />
        ) : null} />
        <Legend iconType="plainline" wrapperStyle={{ paddingTop: 6 }} />
        {CLASS_KEYS.map((k) => (
          <Line key={k} type="monotone" dataKey={k} name={CLASS_LABEL[k]} stroke={colors[k]} strokeWidth={2} dot={{ r: 2.5 }} isAnimationActive={false} connectNulls />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Overlaid curves for several experiments (one metric, one axis). */
export function MultiCurve({ series, metric, height = 240, percent }: {
  series: { id: string; name: string; color: string; history: HistoryRow[] }[];
  metric: keyof HistoryRow;
  height?: number;
  percent?: boolean;
}) {
  const maxEpoch = Math.max(0, ...series.map((s) => s.history.length ? s.history[s.history.length - 1].epoch : 0));
  const data = Array.from({ length: maxEpoch }, (_, i) => {
    const row: Record<string, number | null> = { epoch: i + 1 };
    series.forEach((s) => {
      const h = s.history.find((x) => x.epoch === i + 1);
      row[s.id] = h ? (h[metric] as number | null) : null;
    });
    return row;
  });
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="epoch" tickLine={false} axisLine={false} allowDecimals={false} />
        <YAxis tickLine={false} axisLine={false} width={48} domain={percent ? [0, 1] : ['auto', 'auto']}
          tickFormatter={(v) => (percent ? `${Math.round(v * 100)}%` : Number(v).toFixed(2))} />
        <Tooltip content={({ active, payload, label }: TT) => active && payload?.length ? (
          <TooltipBox title={`epoch ${label}`} rows={payload.map((p) => ({ color: p.color, label: p.name,
            value: percent ? ratio(p.value, 2) : fixed(p.value, 4) }))} />
        ) : null} />
        <Legend iconType="plainline" wrapperStyle={{ paddingTop: 6 }} />
        {series.map((s) => (
          <Line key={s.id} type="monotone" dataKey={s.id} name={s.name} stroke={s.color} strokeWidth={2} dot={{ r: 2.5 }} connectNulls isAnimationActive={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------------------------------------- evaluation */

/** Grouped bars: per-class metric for several models. */
export function GroupedClassBars({ groups, height = 260 }: {
  groups: { id: string; name: string; color: string; values: PerClass<number | null> }[];
  height?: number;
}) {
  const data = CLASS_KEYS.map((k) => {
    const row: Record<string, string | number | null> = { cls: CLASS_LABEL[k] };
    groups.forEach((g) => (row[g.id] = g.values[k]));
    return row;
  });
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 18, right: 8, bottom: 4, left: 0 }} barGap={2} barCategoryGap="22%">
        <CartesianGrid vertical={false} />
        <XAxis dataKey="cls" tickLine={false} axisLine={false} />
        <YAxis tickLine={false} axisLine={false} width={44} domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
        <Tooltip content={({ active, payload, label }: TT) => active && payload?.length ? (
          <TooltipBox title={label} rows={payload.map((p) => ({ color: p.color, label: p.name, value: ratio(p.value, 1) }))} />
        ) : null} />
        <Legend wrapperStyle={{ paddingTop: 6 }} />
        {groups.map((g) => (
          <Bar key={g.id} dataKey={g.id} name={g.name} fill={g.color} radius={[4, 4, 0, 0]} isAnimationActive={false}>
            {groups.length <= 2 && (
              <LabelList dataKey={g.id} position="top" formatter={(v: number | null) => (v === null || v === undefined ? 'n/a' : `${Math.round(v * 100)}`)}
                style={{ fill: 'rgb(var(--ink-3))', fontSize: 10, fontFamily: 'JetBrains Mono' }} />
            )}
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Confusion matrix as an HTML grid; rows = ground truth, columns = prediction. */
export function ConfusionMatrix({ matrix, normalize = true }: { matrix: number[][]; normalize?: boolean }) {
  const rowSums = matrix.map((r) => r.reduce((a, b) => a + b, 0));
  const max = Math.max(1, ...matrix.flat());
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-separate border-spacing-[3px] text-xs" aria-label="Confusion matrix">
        <thead>
          <tr>
            <th className="w-24" />
            <th colSpan={4} className="pb-1 text-center font-mono text-2xs uppercase tracking-wider text-ink-3">Predicted</th>
          </tr>
          <tr>
            <th className="text-left font-mono text-2xs uppercase tracking-wider text-ink-3">True ↓</th>
            {CLASS_KEYS.map((k) => <th key={k} className="pb-1 text-center font-medium text-ink-2">{CLASS_LABEL[k]}</th>)}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th className="pr-2 text-left font-medium text-ink-2">{CLASS_LABEL[CLASS_KEYS[i]]}</th>
              {row.map((v, j) => {
                const share = rowSums[i] ? v / rowSums[i] : 0;
                const t = normalize ? share : v / max;
                const diag = i === j;
                return (
                  <td key={j} title={`True ${CLASS_LABEL[CLASS_KEYS[i]]} → predicted ${CLASS_LABEL[CLASS_KEYS[j]]}: ${int(v)} px (${(share * 100).toFixed(2)}% of row)`}
                    className="h-14 min-w-[72px] rounded-md text-center align-middle"
                    style={{ background: diag ? `rgba(70,190,140,${0.08 + t * 0.62})` : `rgba(232,116,59,${0.05 + t * 0.6})` }}>
                    <div className={cx('num text-sm', t > 0.55 ? 'text-white' : 'text-ink')}>
                      {normalize ? (rowSums[i] ? `${(share * 100).toFixed(1)}%` : 'n/a') : int(v)}
                    </div>
                    <div className={cx('num text-[10px]', t > 0.55 ? 'text-white/75' : 'text-ink-3')}>
                      {normalize ? `${int(v)} px` : rowSums[i] ? `${(share * 100).toFixed(1)}%` : ''}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-2 flex flex-wrap gap-4 text-2xs text-ink-3">
        <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm bg-good/70" /> correct (diagonal)</span>
        <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm bg-accent/70" /> confusion</span>
        <span>Unlabelled pixels are excluded. {normalize ? 'Cells show share of each true-class row (= recall on the diagonal).' : 'Cells show pixel counts.'}</span>
      </div>
    </div>
  );
}

/* --------------------------------------------------------------- terrain */

export function ConfidenceHistogram({ histogram, height = 180 }: {
  histogram: NonNullable<TerrainStats['confidence']>['histogram'];
  height?: number;
}) {
  const data = histogram.map((b) => ({ ...b, label: `${Math.round(b.from * 100)}–${Math.round(b.to * 100)}` }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: 0 }} barCategoryGap={2}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="label" tickLine={false} axisLine={false} interval={1} />
        <YAxis tickLine={false} axisLine={false} width={40} tickFormatter={(v) => `${v}%`} />
        <Tooltip content={({ active, payload }: TT) => active && payload?.length ? (
          <TooltipBox title={`confidence ${payload[0].payload.label}%`} rows={[
            { label: 'Share of pixels', value: `${Number(payload[0].payload.percent).toFixed(2)}%` },
            { label: 'Pixels', value: int(payload[0].payload.pixels) },
          ]} />
        ) : null} />
        <Bar dataKey="percent" radius={[4, 4, 0, 0]} isAnimationActive={false}>
          {data.map((_d, i) => <Cell key={i} fill={`rgba(98,160,222,${0.35 + 0.65 * (i / Math.max(1, data.length - 1))})`} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Class composition from the top of the frame (far) to the bottom (near), as stacked areas. */
export function BandProfile({ bands, height = 240 }: { bands: TerrainStats['band_profile']; height?: number }) {
  const { colors } = useApp();
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={bands} margin={{ top: 8, right: 12, bottom: 4, left: 0 }} stackOffset="expand">
        <CartesianGrid vertical={false} />
        <XAxis dataKey="band" tickLine={false} axisLine={false} tickFormatter={(b) => (b === 1 ? 'top' : b === bands.length ? 'bottom' : String(b))} />
        <YAxis tickLine={false} axisLine={false} width={40} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
        <Tooltip content={({ active, payload, label }: TT) => active && payload?.length ? (
          <TooltipBox title={`band ${label} of ${bands.length}`} rows={[...payload].reverse().map((p) => ({ color: p.color, label: p.name, value: `${Number(p.payload[p.dataKey]).toFixed(1)}%` }))} />
        ) : null} />
        <Legend wrapperStyle={{ paddingTop: 6 }} />
        {CLASS_KEYS.map((k) => (
          <Area key={k} type="monotone" dataKey={k} name={CLASS_LABEL[k]} stackId="1" stroke={colors[k]} fill={colors[k]} fillOpacity={0.75} strokeWidth={1} isAnimationActive={false} />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}

/** 3x3 frame zones, each showing its dominant class and composition. */
export function ZoneGrid({ zones }: { zones: TerrainStats['zone_composition'] }) {
  const { colors } = useApp();
  const rows = ['upper', 'middle', 'lower'];
  const cols = ['left', 'centre', 'right'];
  return (
    <div className="grid grid-cols-3 gap-[3px] overflow-hidden rounded-lg" role="table" aria-label="Zone composition">
      {rows.flatMap((r) => cols.map((c) => {
        const z = zones[`${r}_${c}`];
        const dom = z?.dominant;
        const share = dom ? z.percent[dom] : 0;
        return (
          <div key={`${r}_${c}`} role="cell" className="relative aspect-[4/3] min-w-0 bg-sunken p-2"
            title={CLASS_KEYS.map((k) => `${CLASS_LABEL[k]} ${z?.percent[k]?.toFixed(1) ?? 0}%`).join(' · ')}>
            {dom && <div className="absolute inset-0" style={{ background: colors[dom], opacity: 0.15 + (share / 100) * 0.55 }} />}
            <div className="relative flex h-full flex-col justify-between">
              <span className="font-mono text-[9px] uppercase tracking-wider text-ink-3">{r} {c}</span>
              <div>
                <div className="truncate text-xs font-medium text-ink">{dom ? CLASS_LABEL[dom] : '—'}</div>
                <div className="num text-[11px] text-ink-2">{dom ? `${share.toFixed(0)}%` : 'no pixels'}</div>
              </div>
              <div className="flex h-1 w-full gap-[1px] overflow-hidden rounded-full">
                {CLASS_KEYS.map((k) => (z?.percent[k] ?? 0) > 0 && (
                  <div key={k} style={{ width: `${z.percent[k]}%`, background: colors[k] }} />
                ))}
              </div>
            </div>
          </div>
        );
      }))}
    </div>
  );
}
