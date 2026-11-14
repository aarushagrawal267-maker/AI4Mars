/**
 * In-browser inference for the browser edition, mirroring ml/preprocessing + ml/inference/predictor.py:
 * grayscale -> resize (long side = training resolution, multiple of 32) -> 3 channels -> ImageNet normalisation
 * -> ONNX model (exported from the trained PyTorch checkpoint) -> softmax -> bilinear upsampling -> class map,
 * max-softmax confidence and normalised entropy.
 */
import { drawToRgba } from './render';

export interface BrowserModel { id: string; arch: string; resolution: number; parts: string[]; onnx_bytes: number; parameters: number }

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Ort = any;
declare global { interface Window { ort?: Ort } }

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
export const MAX_OUTPUT_SIDE = 768;

let ortPromise: Promise<Ort> | null = null;
function loadOrt(): Promise<Ort> {
  if (window.ort) return Promise.resolve(window.ort);
  ortPromise ??= new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = 'ort/ort.wasm.min.js';
    s.onload = () => {
      const ort = window.ort;
      if (!ort) return reject(new Error('The inference engine did not load.'));
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      ort.env.wasm.wasmPaths = new URL('ort/', location.href).href;
      resolve(ort);
    };
    s.onerror = () => reject(new Error('The inference engine could not be loaded.'));
    document.head.appendChild(s);
  });
  return ortPromise;
}

let current: { id: string; session: Ort } | null = null;
let loading: Promise<Ort> | null = null;
let loadingId = '';

/** Load (and keep) one model at a time; `onProgress` reports download progress 0..1. */
export async function getSession(model: BrowserModel, onProgress?: (f: number, msg: string) => void): Promise<Ort> {
  if (current?.id === model.id) return current.session;
  if (loading && loadingId === model.id) return loading;
  loadingId = model.id;
  loading = (async () => {
    const ort = await loadOrt();
    const chunks: Uint8Array[] = [];
    for (let i = 0; i < model.parts.length; i++) {
      onProgress?.(i / model.parts.length, `Downloading model part ${i + 1} of ${model.parts.length}`);
      const res = await fetch(model.parts[i]);
      if (!res.ok) throw new Error(`Model download failed (HTTP ${res.status}).`);
      const bin = atob((await res.text()).trim());
      const bytes = new Uint8Array(bin.length);
      for (let j = 0; j < bin.length; j++) bytes[j] = bin.charCodeAt(j);
      chunks.push(bytes);
    }
    onProgress?.(1, 'Preparing model');
    const all = new Uint8Array(chunks.reduce((s, c) => s + c.length, 0));
    let off = 0;
    for (const c of chunks) { all.set(c, off); off += c.length; }
    if (current) { try { await current.session.release?.(); } catch { /* ignore */ } current = null; }
    const session = await ort.InferenceSession.create(all, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    current = { id: model.id, session };
    return session;
  })();
  try {
    return await loading;
  } finally {
    loading = null;
  }
}

export function inferenceSize(h: number, w: number, resolution: number, multiple = 32): [number, number] {
  const scale = resolution / Math.max(h, w);
  return [Math.max(multiple, Math.round((h * scale) / multiple) * multiple), Math.max(multiple, Math.round((w * scale) / multiple) * multiple)];
}

export function outputSize(h: number, w: number, maxSide = MAX_OUTPUT_SIDE): [number, number] {
  const s = Math.min(1, maxSide / Math.max(h, w));
  return [Math.max(1, Math.round(h * s)), Math.max(1, Math.round(w * s))];
}

export interface Prediction {
  cls: Uint8Array; conf: Float32Array; unc: Float32Array; rgb: Uint8ClampedArray;
  width: number; height: number; inputSize: [number, number]; originalSize: [number, number];
  inferenceMs: number; meanProbs: number[]; colorInput: boolean;
}

export async function predict(model: BrowserModel, img: CanvasImageSource, origW: number, origH: number,
  onProgress?: (f: number, msg: string) => void): Promise<Prediction> {
  const session = await getSession(model, onProgress);
  const ort = window.ort;
  onProgress?.(1, 'Running the model');
  const [nh, nw] = inferenceSize(origH, origW, model.resolution);
  const small = drawToRgba(img, nw, nh);
  const n = nh * nw;
  const x = new Float32Array(3 * n);
  for (let p = 0; p < n; p++) {
    const g = Math.round(0.299 * small[p * 4] + 0.587 * small[p * 4 + 1] + 0.114 * small[p * 4 + 2]) / 255;
    for (let c = 0; c < 3; c++) x[c * n + p] = (g - MEAN[c]) / STD[c];
  }
  const t0 = performance.now();
  const out = await session.run({ input: new ort.Tensor('float32', x, [1, 3, nh, nw]) });
  const inferenceMs = performance.now() - t0;
  const logits: Float32Array = out.logits.data;
  const lh = out.logits.dims[2], lw = out.logits.dims[3], ln = lh * lw;
  // softmax at network resolution
  const probs = new Float32Array(4 * ln);
  for (let p = 0; p < ln; p++) {
    let m = -Infinity;
    for (let c = 0; c < 4; c++) m = Math.max(m, logits[c * ln + p]);
    let s = 0;
    for (let c = 0; c < 4; c++) { const e = Math.exp(logits[c * ln + p] - m); probs[c * ln + p] = e; s += e; }
    for (let c = 0; c < 4; c++) probs[c * ln + p] /= s;
  }
  // bilinear upsampling to the output size (align_corners = False), then class map / confidence / entropy
  const [oh, ow] = outputSize(origH, origW);
  const cls = new Uint8Array(oh * ow), conf = new Float32Array(oh * ow), unc = new Float32Array(oh * ow);
  const meanProbs = [0, 0, 0, 0];
  const sy = lh / oh, sx = lw / ow, LN4 = Math.log(4);
  const pv = [0, 0, 0, 0];
  for (let y = 0; y < oh; y++) {
    const fy = Math.max(0, (y + 0.5) * sy - 0.5), y0 = Math.min(lh - 1, Math.floor(fy)), y1 = Math.min(lh - 1, y0 + 1), wy = fy - y0;
    for (let xo = 0; xo < ow; xo++) {
      const fx = Math.max(0, (xo + 0.5) * sx - 0.5), x0 = Math.min(lw - 1, Math.floor(fx)), x1 = Math.min(lw - 1, x0 + 1), wx = fx - x0;
      let s = 0;
      for (let c = 0; c < 4; c++) {
        const b = c * ln;
        const v = (probs[b + y0 * lw + x0] * (1 - wx) + probs[b + y0 * lw + x1] * wx) * (1 - wy)
          + (probs[b + y1 * lw + x0] * (1 - wx) + probs[b + y1 * lw + x1] * wx) * wy;
        pv[c] = v; s += v;
      }
      let best = 0, ent = 0;
      for (let c = 0; c < 4; c++) {
        pv[c] /= s || 1;
        meanProbs[c] += pv[c];
        if (pv[c] > pv[best]) best = c;
        ent -= pv[c] * Math.log(Math.max(pv[c], 1e-8));
      }
      const p = y * ow + xo;
      cls[p] = best; conf[p] = pv[best]; unc[p] = Math.min(1, Math.max(0, ent / LN4));
    }
  }
  const rgb = drawToRgba(img, ow, oh);
  let colorInput = false;
  for (let p = 0; p < oh * ow && !colorInput; p += 7) {
    if (rgb[p * 4] !== rgb[p * 4 + 1] || rgb[p * 4 + 1] !== rgb[p * 4 + 2]) colorInput = true;
  }
  return { cls, conf, unc, rgb, width: ow, height: oh, inputSize: [nh, nw], originalSize: [origH, origW], inferenceMs,
    meanProbs: meanProbs.map((v) => v / (oh * ow)), colorInput };
}
