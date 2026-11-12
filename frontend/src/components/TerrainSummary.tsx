import { useApp } from '../context/AppContext';
import type { Analysis } from '../types/api';
import { CLASS_KEYS, CLASS_LABEL, int, ms, ratio } from '../utils/format';
import { ClassBars, CompositionRing } from '../charts/Charts';
import { Callout, ClassSwatch, FixtureBadge } from './ui';

/** Composition + confidence summary for one analysis. Every value comes from the stored prediction. */
export default function TerrainSummary({ analysis }: { analysis: Analysis }) {
  const { colorOf } = useApp();
  const s = analysis.stats;
  const dom = s.dominant_class;
  return (
    <div className="space-y-4">
      {analysis.is_fixture && (
        <Callout tone="info" title="Test fixture output">
          This result comes from the untrained, deterministic test fixture. It exists only so automated interface tests can run.
          It is <b>not</b> a real AI prediction.
        </Callout>
      )}
      <div className="flex flex-wrap items-center gap-5">
        <CompositionRing percent={s.class_percent} center={
          <div>
            <div className="eyebrow">dominant</div>
            <div className="mt-0.5 text-base font-semibold" style={{ color: dom ? colorOf(dom) : undefined }}>{dom ? CLASS_LABEL[dom] : '—'}</div>
            <div className="num text-xs text-ink-2">{dom ? `${s.class_percent[dom].toFixed(1)}%` : ''}</div>
          </div>
        } />
        <div className="min-w-[220px] flex-1">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left font-mono text-2xs uppercase tracking-wider text-ink-3">
                <th className="pb-1.5 font-medium">Class</th>
                <th className="pb-1.5 text-right font-medium">Share</th>
                <th className="pb-1.5 text-right font-medium">Pixels</th>
                <th className="pb-1.5 text-right font-medium" title="Mean softmax confidence of pixels predicted as this class">Conf.</th>
              </tr>
            </thead>
            <tbody>
              {CLASS_KEYS.map((k) => (
                <tr key={k} className="border-t border-line/60">
                  <td className="py-1.5"><span className="flex items-center gap-2"><ClassSwatch k={k} /> {CLASS_LABEL[k]}</span></td>
                  <td className="num py-1.5 text-right">{s.class_percent[k].toFixed(2)}%</td>
                  <td className="num py-1.5 text-right text-ink-2">{int(s.class_pixels[k])}</td>
                  <td className="num py-1.5 text-right text-ink-2">{ratio(s.confidence?.per_class_mean[k] ?? null, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <ClassBars percent={s.class_percent} height={150} />
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {[
          ['Valid pixels', int(s.valid_pixels)],
          ['Mean confidence', ratio(s.confidence?.mean ?? null, 1)],
          ['Below 50% conf.', `${s.confidence?.low_confidence_percent.toFixed(1) ?? '—'}%`],
          ['Forward pass', ms(analysis.meta.inference_ms)],
        ].map(([k, v]) => (
          <div key={k} className="rounded-lg border border-line bg-sunken/60 px-3 py-2">
            <div className="eyebrow">{k}</div>
            <div className="num mt-1 text-sm text-ink">{v}</div>
          </div>
        ))}
      </div>
      <p className="text-sm leading-relaxed text-ink-2">{s.summary}</p>
      <div className="flex flex-wrap items-center gap-2 text-2xs text-ink-3">
        {analysis.is_fixture && <FixtureBadge />}
        <span>Model input {analysis.meta.input_size.join('×')} px</span>·
        <span>output {analysis.meta.output_size.join('×')} px</span>·
        <span>{analysis.meta.device}</span>
        {analysis.meta.color_input && <span className="text-warn">· colour image converted to grayscale (models were trained on grayscale NavCam)</span>}
      </div>
    </div>
  );
}
