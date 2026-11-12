import type { ClassKey } from '../types/api';

export const CLASS_KEYS: ClassKey[] = ['soil', 'bedrock', 'sand', 'big_rock'];
export const CLASS_LABEL: Record<ClassKey, string> = {
  soil: 'Soil',
  bedrock: 'Bedrock',
  sand: 'Sand',
  big_rock: 'Big rock',
};

/** Fallback palette (the live palette comes from /api/settings/colors). */
export const DEFAULT_COLORS: Record<ClassKey, string> = {
  soil: '#B0682C',
  bedrock: '#35A3E8',
  sand: '#E6AE35',
  big_rock: '#B44BD8',
};

/** Okabe-Ito based preset, designed to stay distinguishable for colour-vision deficiencies. */
export const CVD_COLORS: Record<ClassKey, string> = {
  soil: '#D55E00',
  bedrock: '#0072B2',
  sand: '#F0E442',
  big_rock: '#CC79A7',
};

export const dash = '—';

export function pct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return dash;
  return `${v.toFixed(digits)}%`;
}

/** 0..1 ratio rendered as a percentage. */
export function ratio(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return dash;
  return `${(v * 100).toFixed(digits)}%`;
}

export function fixed(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return dash;
  return v.toFixed(digits);
}

export function int(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return dash;
  return Math.round(v).toLocaleString('en-US');
}

export function compact(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return dash;
  const a = Math.abs(v);
  if (a >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
  if (a >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  return `${Math.round(v)}`;
}

export function bytes(v: number | null | undefined): string {
  if (v === null || v === undefined) return dash;
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let n = v;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n.toFixed(n >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
}

export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return dash;
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h) return `${h}h ${String(m).padStart(2, '0')}m`;
  if (m) return `${m}m ${String(r).padStart(2, '0')}s`;
  return `${r}s`;
}

export function ms(v: number | null | undefined): string {
  if (v === null || v === undefined) return dash;
  return v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${v.toFixed(v < 10 ? 2 : 1)} ms`;
}

export function when(iso: string | null | undefined): string {
  if (!iso) return dash;
  // Backend timestamps are naive UTC.
  const d = new Date(iso.endsWith('Z') || iso.includes('+') ? iso : `${iso}Z`);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

export function ago(iso: string | null | undefined): string {
  if (!iso) return dash;
  const d = new Date(iso.endsWith('Z') || iso.includes('+') ? iso : `${iso}Z`);
  const s = (Date.now() - d.getTime()) / 1000;
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${Math.floor(s / 86400)} d ago`;
}

export const ARCH_LABEL: Record<string, string> = { unet: 'U-Net', deeplabv3: 'DeepLabV3' };
export const archLabel = (a: string | undefined | null) => (a ? ARCH_LABEL[a] ?? a : dash);

export function hexToRgb(hex: string): [number, number, number] {
  const v = hex.replace('#', '');
  return [parseInt(v.slice(0, 2), 16), parseInt(v.slice(2, 4), 16), parseInt(v.slice(4, 6), 16)];
}

/** Readable text colour (dark or light) on top of a given fill. */
export function inkOn(hex: string): string {
  const [r, g, b] = hexToRgb(hex);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 150 ? '#14100c' : '#ffffff';
}

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(' ');
}
