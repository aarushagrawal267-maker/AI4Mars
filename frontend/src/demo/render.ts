/** Canvas rendering helpers for the browser edition (mirrors ml/inference/visualize.py). */
import type { ClassKey, PerClass } from '../types/api';

export const KEYS: ClassKey[] = ['soil', 'bedrock', 'sand', 'big_rock'];
const UNLABELED: [number, number, number] = [0x1f, 0x23, 0x28];

export function hexRgb(hex: string): [number, number, number] {
  const v = hex.replace('#', '');
  return [parseInt(v.slice(0, 2), 16), parseInt(v.slice(2, 4), 16), parseInt(v.slice(4, 6), 16)];
}

function ramp(stops: string[]): Uint8Array {
  const lut = new Uint8Array(256 * 3);
  const rgb = stops.map(hexRgb);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (rgb.length - 1);
    const a = Math.floor(t), b = Math.min(rgb.length - 1, a + 1), f = t - a;
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(rgb[a][c] * (1 - f) + rgb[b][c] * f);
  }
  return lut;
}
export const VIRIDIS = ramp(['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']);
export const INFERNO = ramp(['#000004', '#420a68', '#932667', '#dd513a', '#fca50a', '#fcffa4']);

function canvas(w: number, h: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Canvas is not available in this browser.');
  return [c, ctx];
}

export function toUrl(rgba: Uint8ClampedArray, w: number, h: number): string {
  const [c, ctx] = canvas(w, h);
  ctx.putImageData(new ImageData(rgba, w, h), 0, 0);
  return c.toDataURL('image/png');
}

/** Single-channel 0-255 values as a greyscale PNG (class ids, confidence*255, entropy*255). */
export function grayUrl(values: ArrayLike<number>, w: number, h: number): string {
  const out = new Uint8ClampedArray(w * h * 4);
  for (let p = 0; p < w * h; p++) {
    const v = values[p];
    out[p * 4] = out[p * 4 + 1] = out[p * 4 + 2] = v;
    out[p * 4 + 3] = 255;
  }
  return toUrl(out, w, h);
}

export function maskRgba(cls: Uint8Array, colors: PerClass<string>): Uint8ClampedArray {
  const pal = KEYS.map((k) => hexRgb(colors[k]));
  const out = new Uint8ClampedArray(cls.length * 4);
  for (let p = 0; p < cls.length; p++) {
    const c = cls[p] < 4 ? pal[cls[p]] : UNLABELED;
    out[p * 4] = c[0]; out[p * 4 + 1] = c[1]; out[p * 4 + 2] = c[2]; out[p * 4 + 3] = 255;
  }
  return out;
}

export function overlayRgba(rgb: Uint8ClampedArray, cls: Uint8Array, colors: PerClass<string>, alpha: number): Uint8ClampedArray {
  const pal = KEYS.map((k) => hexRgb(colors[k]));
  const a = Math.min(1, Math.max(0, alpha));
  const out = new Uint8ClampedArray(rgb.length);
  for (let p = 0; p < cls.length; p++) {
    const i = p * 4;
    if (cls[p] >= 4) { out[i] = rgb[i]; out[i + 1] = rgb[i + 1]; out[i + 2] = rgb[i + 2]; }
    else {
      const c = pal[cls[p]];
      out[i] = rgb[i] * (1 - a) + c[0] * a; out[i + 1] = rgb[i + 1] * (1 - a) + c[1] * a; out[i + 2] = rgb[i + 2] * (1 - a) + c[2] * a;
    }
    out[i + 3] = 255;
  }
  return out;
}

export function heatRgba(values: Float32Array, lut: Uint8Array): Uint8ClampedArray {
  const out = new Uint8ClampedArray(values.length * 4);
  for (let p = 0; p < values.length; p++) {
    const v = Math.round(Math.min(1, Math.max(0, values[p])) * 255) * 3;
    out[p * 4] = lut[v]; out[p * 4 + 1] = lut[v + 1]; out[p * 4 + 2] = lut[v + 2]; out[p * 4 + 3] = 255;
  }
  return out;
}

/** Equally sized panels next to each other with an 8 px gap (like side_by_side in Python). */
export function sideBySideUrl(panels: Uint8ClampedArray[], w: number, h: number, gap = 8): string {
  const W = w * panels.length + gap * (panels.length - 1);
  const [c, ctx] = canvas(W, h);
  ctx.fillStyle = '#111317';
  ctx.fillRect(0, 0, W, h);
  panels.forEach((p, i) => ctx.putImageData(new ImageData(p, w, h), i * (w + gap), 0));
  return c.toDataURL('image/png');
}

/** Draw an image element (or bitmap) into an RGBA buffer of the given size. */
export function drawToRgba(src: CanvasImageSource, w: number, h: number): Uint8ClampedArray {
  const [, ctx] = canvas(w, h);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(src, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h).data;
}
