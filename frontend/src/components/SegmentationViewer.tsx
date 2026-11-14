import { Columns2, Crosshair, Download, Eye, Layers, Maximize2, SplitSquareHorizontal } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useApp } from '../context/AppContext';
import { api, type RenderKind } from '../services/api';
import type { Analysis, ClassKey } from '../types/api';
import { CLASS_KEYS, CLASS_LABEL, cx } from '../utils/format';
import { IS_DEMO } from '../demo/env';
import { ClassLegend, Modal, Segmented } from './ui';

type Mode = 'overlay' | 'mask' | 'original' | 'compare' | 'confidence' | 'uncertainty' | 'side_by_side';

const MODES: { value: Mode; label: string }[] = [
  { value: 'overlay', label: 'Overlay' },
  { value: 'mask', label: 'Mask' },
  { value: 'original', label: 'Original' },
  { value: 'compare', label: 'Swipe' },
  { value: 'confidence', label: 'Confidence' },
  { value: 'uncertainty', label: 'Uncertainty' },
  { value: 'side_by_side', label: 'Side by side' },
];

interface Probe {
  x: number;
  y: number;
  px: number;
  py: number;
  cls: ClassKey | null;
  conf: number | null;
  unc: number | null;
}

/** Decodes a server PNG into raw pixel data so the cursor can be "probed". */
function useImagePixels(url: string | null): ImageData | null {
  const [data, setData] = useState<ImageData | null>(null);
  useEffect(() => {
    setData(null);
    if (!url) return;
    let alive = true;
    const img = new Image();
    img.onload = () => {
      if (!alive) return;
      const c = document.createElement('canvas');
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      const ctx = c.getContext('2d', { willReadFrequently: true });
      if (!ctx) return;
      ctx.drawImage(img, 0, 0);
      try {
        setData(ctx.getImageData(0, 0, c.width, c.height));
      } catch {
        setData(null);
      }
    };
    img.src = url;
    return () => {
      alive = false;
    };
  }, [url]);
  return data;
}

export default function SegmentationViewer({ analysis, defaultMode = 'overlay', maxHeight = 560, compact }: {
  analysis: Analysis;
  defaultMode?: Mode;
  maxHeight?: number;
  compact?: boolean;
}) {
  const { colors, colorVersion } = useApp();
  const [mode, setMode] = useState<Mode>(defaultMode);
  const [alpha, setAlpha] = useState(0.5);
  const [split, setSplit] = useState(0.5);
  const [probe, setProbe] = useState<Probe | null>(null);
  const [inspect, setInspect] = useState(true);
  const [zoom, setZoom] = useState(false);
  const frame = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const v = colorVersion;
  const url = (kind: RenderKind, a = alpha) => api.analysisImageUrl(analysis.id, kind, { alpha: a, v });
  const ids = useImagePixels(inspect ? api.analysisImageUrl(analysis.id, 'class_ids') : null);
  const conf = useImagePixels(inspect ? api.analysisImageUrl(analysis.id, 'confidence_raw') : null);
  const [h, w] = analysis.meta.output_size;

  // Debounce alpha so dragging the slider does not request dozens of PNGs.
  const [alphaReq, setAlphaReq] = useState(alpha);
  useEffect(() => {
    const t = window.setTimeout(() => setAlphaReq(alpha), 180);
    return () => window.clearTimeout(t);
  }, [alpha]);

  const src: Record<Exclude<Mode, 'compare'>, string> = {
    overlay: url('overlay', alphaReq),
    mask: url('mask'),
    original: url('original'),
    confidence: url('confidence'),
    uncertainty: url('uncertainty'),
    side_by_side: url('side_by_side', alphaReq),
  };

  function onMove(e: React.PointerEvent<HTMLDivElement>) {
    const rect = frame.current?.getBoundingClientRect();
    if (!rect) return;
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;
    if (mode === 'compare' && dragging.current) setSplit(Math.min(1, Math.max(0, fx)));
    if (!inspect || mode === 'side_by_side' || fx < 0 || fx > 1 || fy < 0 || fy > 1) {
      setProbe(null);
      return;
    }
    const px = Math.min(w - 1, Math.floor(fx * w));
    const py = Math.min(h - 1, Math.floor(fy * h));
    // class_ids and confidence_raw are greyscale: R holds the class id / confidence * 255.
    const ci = ids ? ids.data[(py * ids.width + px) * 4] : null;
    const cv = conf ? conf.data[(py * conf.width + px) * 4] / 255 : null;
    setProbe({ x: e.clientX - rect.left, y: e.clientY - rect.top, px, py,
      cls: ci !== null && ci < 4 ? CLASS_KEYS[ci] : null, conf: cv, unc: null });
  }

  const aspect = mode === 'side_by_side' ? (w * 3 + 16) / h : w / h;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Segmented ariaLabel="View mode" size="sm" options={MODES.filter((m) => !compact || m.value !== 'side_by_side')} value={mode} onChange={setMode} />
        <div className="ml-auto flex items-center gap-2">
          {(mode === 'overlay' || mode === 'side_by_side') && (
            <label className="flex items-center gap-2 text-2xs text-ink-3">
              <Layers className="h-3.5 w-3.5" /> opacity
              <input type="range" min={0.1} max={0.9} step={0.05} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} className="w-24" aria-label="Overlay opacity" />
              <span className="num w-8 text-ink-2">{Math.round(alpha * 100)}%</span>
            </label>
          )}
          <button className={cx('btn-ghost p-1.5', inspect && 'text-accent')} onClick={() => setInspect(!inspect)} title="Pixel inspector" aria-pressed={inspect}>
            <Crosshair className="h-4 w-4" />
          </button>
          <button className="btn-ghost p-1.5" onClick={() => setZoom(true)} title="Enlarge"><Maximize2 className="h-4 w-4" /></button>
        </div>
      </div>

      <div className="flex justify-center rounded-xl border border-line bg-checker p-2">
        <div ref={frame} className="relative max-w-full select-none overflow-hidden rounded-md"
          style={{ aspectRatio: String(aspect), height: `min(${maxHeight}px, calc((100vw - 2rem) / ${aspect}))`, maxHeight }}
          onPointerMove={onMove} onPointerLeave={() => setProbe(null)}
          onPointerDown={(e) => { if (mode === 'compare') { dragging.current = true; onMove(e); (e.target as HTMLElement).setPointerCapture?.(e.pointerId); } }}
          onPointerUp={() => (dragging.current = false)}
          data-testid="segmentation-frame">
          {mode === 'compare' ? (
            <>
              <img src={url('original')} alt="Original" className="absolute inset-0 h-full w-full object-fill" draggable={false} />
              <div className="absolute inset-0" style={{ clipPath: `inset(0 0 0 ${split * 100}%)` }}>
                <img src={url('overlay', 0.6)} alt="Segmentation overlay" className="h-full w-full object-fill" draggable={false} />
              </div>
              <div className="absolute inset-y-0 w-0.5 bg-white/90 shadow-[0_0_12px_rgba(0,0,0,0.6)]" style={{ left: `calc(${split * 100}% - 1px)` }}>
                <div className="absolute left-1/2 top-1/2 grid h-8 w-8 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize place-items-center rounded-full border border-white/80 bg-black/60 text-white">
                  <SplitSquareHorizontal className="h-4 w-4" />
                </div>
              </div>
              <span className="absolute left-2 top-2 rounded bg-black/60 px-1.5 py-0.5 font-mono text-[10px] text-white">ORIGINAL</span>
              <span className="absolute right-2 top-2 rounded bg-black/60 px-1.5 py-0.5 font-mono text-[10px] text-white">PREDICTION</span>
            </>
          ) : (
            <img src={src[mode]} alt={`${mode} view`} className={cx('h-full w-full object-fill', mode === 'mask' && 'pixelated')} draggable={false} />
          )}
          {probe && inspect && (
            <>
              <div className="pointer-events-none absolute h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/90 shadow"
                style={{ left: probe.x, top: probe.y }} />
              <div className="pointer-events-none absolute z-10 min-w-[150px] rounded-lg border border-white/15 bg-black/80 px-2.5 py-2 text-xs text-white shadow-xl backdrop-blur"
                style={{ left: Math.min(probe.x + 14, (frame.current?.clientWidth ?? 0) - 170), top: Math.max(4, probe.y - 70) }}>
                <div className="font-mono text-[10px] text-white/60">px ({probe.px}, {probe.py})</div>
                <div className="mt-1 flex items-center gap-1.5 font-medium">
                  {probe.cls && <span className="h-2.5 w-2.5 rounded-sm" style={{ background: colors[probe.cls] }} />}
                  {probe.cls ? CLASS_LABEL[probe.cls] : ids ? 'n/a' : 'loading…'}
                </div>
                <div className="mt-0.5 font-mono text-[11px] text-white/80">
                  confidence {probe.conf !== null ? `${(probe.conf * 100).toFixed(1)}%` : '…'}
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        {mode === 'confidence' || mode === 'uncertainty' ? (
          <div className="flex items-center gap-2 text-2xs text-ink-3">
            <span>{mode === 'confidence' ? 'low confidence' : 'certain'}</span>
            <span className="h-2 w-40 rounded-full" style={{ background: mode === 'confidence'
              ? 'linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)'
              : 'linear-gradient(90deg,#000004,#420a68,#932667,#dd513a,#fca50a,#fcffa4)' }} />
            <span>{mode === 'confidence' ? 'high confidence' : 'maximally uncertain'}</span>
            <span className="ml-2 hidden text-ink-3 md:inline">
              {mode === 'confidence' ? 'max softmax probability (uncalibrated)' : 'normalised predictive entropy'}
            </span>
          </div>
        ) : (
          <ClassLegend />
        )}
        <div className={cx('flex items-center gap-1', IS_DEMO && 'hidden')}>
          <Eye className="hidden h-3.5 w-3.5 text-ink-3 sm:block" />
          {(['mask', 'overlay', 'side_by_side', 'class_ids'] as RenderKind[]).map((k) => (
            <a key={k} className="btn-ghost px-2 py-1 text-2xs" href={api.analysisImageUrl(analysis.id, k, { alpha: alphaReq, download: true, v })}
              download title={k === 'class_ids' ? 'Raw class-id PNG (0 soil, 1 bedrock, 2 sand, 3 big rock)' : undefined}>
              <Download className="h-3 w-3" /> {k === 'side_by_side' ? 'panel' : k === 'class_ids' ? 'class ids' : k}
            </a>
          ))}
        </div>
      </div>

      <Modal open={zoom} onClose={() => setZoom(false)} title={<span className="flex items-center gap-2"><Columns2 className="h-4 w-4 text-accent" /> {analysis.filename}</span>} wide>
        <img src={src.side_by_side} alt="Original, mask and overlay" className="w-full rounded-md" />
        <ClassLegend className="mt-3" />
      </Modal>
    </div>
  );
}
