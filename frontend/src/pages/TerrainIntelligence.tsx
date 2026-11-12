import { Activity, Compass, Grid3x3, Layers, Map, Mountain, ScanSearch, Waves } from 'lucide-react';
import { useEffect } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { BandProfile, ClassBars, ConfidenceHistogram, ZoneGrid } from '../charts/Charts';
import SegmentationViewer from '../components/SegmentationViewer';
import { AnalysisThumb } from '../components/shared';
import { Callout, ClassSwatch, EmptyState, ErrorBanner, FixtureBadge, Loading, PageHeader, Panel, Stat } from '../components/ui';
import { useApp } from '../context/AppContext';
import { useRemote } from '../hooks/usePolling';
import { api } from '../services/api';
import { archLabel, CLASS_KEYS, CLASS_LABEL, compact, fixed, int, pct, ratio, when } from '../utils/format';

export default function TerrainIntelligence() {
  const { connection, colorOf } = useApp();
  const [params, setParams] = useSearchParams();
  const history = useRemote(() => api.analyses(60), [connection]);
  const id = params.get('id') ?? history.data?.analyses[0]?.id ?? null;
  const analysis = useRemote(() => (id ? api.analysis(id) : Promise.resolve(null)), [id]);
  useEffect(() => {
    if (!params.get('id') && history.data?.analyses[0]) setParams({ id: history.data.analyses[0].id }, { replace: true });
  }, [history.data, params, setParams]);

  const a = analysis.data;
  const s = a?.stats;

  return (
    <div>
      <PageHeader eyebrow="06 · interpretation" title="Terrain Intelligence"
        description="Detailed terrain composition, spatial layout, confidence and uncertainty for an analysed image. All values are arithmetic on the model's prediction for this image; they describe what the model predicted, not verified ground truth." />
      {history.error && <ErrorBanner error={history.error} onRetry={history.reload} />}
      {history.data && history.data.analyses.length === 0 ? (
        <EmptyState icon={Mountain} title="No analysed images yet" action={<Link to="/analysis" className="btn-primary"><ScanSearch className="h-4 w-4" /> Analyse an image</Link>}>
          Run a segmentation in Image Analysis; its terrain breakdown appears here.
        </EmptyState>
      ) : (
        <div className="grid gap-6 xl:grid-cols-[260px_minmax(0,1fr)]">
          <Panel title="Analyses" icon={Layers} padded={false} className="xl:sticky xl:top-20 xl:self-start">
            <div className="max-h-[70vh] space-y-0.5 overflow-y-auto p-2">
              {history.data?.analyses.map((x) => <AnalysisThumb key={x.id} a={x} active={x.id === id} onClick={() => setParams({ id: x.id })} />)}
            </div>
          </Panel>
          <div className="min-w-0 space-y-6">
            {analysis.error && <ErrorBanner error={analysis.error} />}
            {!a || !s ? <Loading /> : (
              <>
                <div className="flex flex-wrap items-center gap-2 text-xs text-ink-3">
                  <span className="font-medium text-ink">{a.filename}</span>·<span>{archLabel(a.arch)} · {a.model_name}</span>·<span>{when(a.created_at)}</span>
                  {a.is_fixture && <FixtureBadge />}
                  <Link to={`/navigation?id=${a.id}`} className="btn-secondary ml-auto text-xs"><Map className="h-3.5 w-3.5" /> Plan a route on this terrain</Link>
                </div>
                <div className="grid grid-cols-2 gap-3 md:grid-cols-4 2xl:grid-cols-7">
                  <div className="panel col-span-2 px-4 py-3.5">
                    <div className="eyebrow">dominant terrain</div>
                    <div className="mt-1.5 flex items-center gap-2 text-2xl font-semibold" style={{ color: s.dominant_class ? colorOf(s.dominant_class) : undefined }}>
                      {s.dominant_class && <ClassSwatch k={s.dominant_class} size={14} />}
                      {s.dominant_class ? CLASS_LABEL[s.dominant_class] : '—'}
                    </div>
                    <div className="num mt-1 text-xs text-ink-3">{s.dominant_class ? pct(s.class_percent[s.dominant_class], 2) : ''} of classified pixels</div>
                  </div>
                  {CLASS_KEYS.map((k) => <Stat key={k} label={CLASS_LABEL[k]} value={pct(s.class_percent[k], 1)} sub={`${int(s.class_pixels[k])} px`} />)}
                  <Stat label="Valid pixels" value={compact(s.valid_pixels)} sub={`${int(s.valid_pixels)} · ${s.width}×${s.height}`} />
                </div>

                <Panel title="Terrain map" icon={Mountain} ticks><SegmentationViewer analysis={a} defaultMode="mask" maxHeight={520} /></Panel>

                <Panel title="Interpretation" icon={Compass}>
                  <p className="text-sm leading-relaxed text-ink-2">{s.summary}</p>
                  <Callout tone="warn" className="mt-3">Segmentation describes appearance only. It cannot tell slope, rock height, sinkage risk or traction, so it is not by itself a statement about whether a rover can drive here.</Callout>
                </Panel>

                <div className="grid gap-6 xl:grid-cols-2">
                  <Panel title="Terrain distribution" icon={Layers}><ClassBars percent={s.class_percent} height={180} valueDigits={2} /></Panel>
                  <Panel title="Frame zones" eyebrow="dominant class per 3×3 zone" icon={Grid3x3}><ZoneGrid zones={s.zone_composition} /></Panel>
                </div>

                <Panel title="Near-to-far profile" eyebrow="composition by horizontal band, top of frame (usually far) to bottom (usually near)" icon={Waves}>
                  <BandProfile bands={s.band_profile} />
                </Panel>

                <Panel title="Spatial distribution" icon={Grid3x3} padded={false}>
                  <div className="overflow-x-auto">
                    <table className="table-clean">
                      <thead><tr><th>Class</th><th className="text-right">Share</th><th className="text-right">Regions</th><th className="text-right">Largest region</th><th className="text-right">Centroid (row, col)</th><th className="text-right">Extent</th><th>Where most of it is</th></tr></thead>
                      <tbody>
                        {CLASS_KEYS.map((k) => {
                          const e = s.spatial[k];
                          const zone = Object.entries(e.zone_share_percent).sort((x, y) => y[1] - x[1])[0];
                          return (
                            <tr key={k}>
                              <td><span className="flex items-center gap-2"><ClassSwatch k={k} />{CLASS_LABEL[k]}</span></td>
                              <td className="num text-right">{pct(s.class_percent[k], 2)}</td>
                              <td className="num text-right">{int(e.regions)}</td>
                              <td className="num text-right">{e.pixels ? `${e.largest_region_percent_of_class.toFixed(1)}% of class` : '—'}</td>
                              <td className="num text-right text-ink-2">{e.centroid ? `${(e.centroid[0] * 100).toFixed(0)}%, ${(e.centroid[1] * 100).toFixed(0)}%` : '—'}</td>
                              <td className="num text-right text-ink-2">{e.bbox ? `${((e.bbox[2] - e.bbox[0]) * 100).toFixed(0)}% × ${((e.bbox[3] - e.bbox[1]) * 100).toFixed(0)}%` : '—'}</td>
                              <td className="text-ink-2">{e.pixels && zone ? `${zone[0].replace('_', ' ')} (${zone[1].toFixed(0)}%)` : '—'}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <p className="px-4 py-3 text-2xs text-ink-3">Regions are 8-connected components of the predicted class. Centroid and extent are fractions of image height and width.</p>
                </Panel>

                <div className="grid gap-6 xl:grid-cols-2">
                  <Panel title="Confidence distribution" icon={Activity} eyebrow="max softmax probability · uncalibrated">
                    {s.confidence && <ConfidenceHistogram histogram={s.confidence.histogram} />}
                    <div className="mt-3 grid grid-cols-3 gap-2 text-center text-xs">
                      <div className="rounded-lg border border-line bg-sunken/60 py-2"><div className="num text-base">{ratio(s.confidence?.mean, 1)}</div><div className="eyebrow">mean</div></div>
                      <div className="rounded-lg border border-line bg-sunken/60 py-2"><div className="num text-base">{ratio(s.confidence?.median, 1)}</div><div className="eyebrow">median</div></div>
                      <div className="rounded-lg border border-line bg-sunken/60 py-2"><div className="num text-base">{pct(s.confidence?.low_confidence_percent)}</div><div className="eyebrow">&lt; 50%</div></div>
                    </div>
                    <p className="mt-3 text-2xs leading-relaxed text-ink-3">{s.confidence?.note}</p>
                  </Panel>
                  <Panel title="Uncertainty" icon={Activity} eyebrow="normalised predictive entropy">
                    <img src={api.analysisImageUrl(a.id, 'uncertainty')} alt="Uncertainty map" className="w-full rounded-md border border-line" />
                    <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                      <div className="rounded-lg border border-line bg-sunken/60 p-2"><div className="eyebrow">mean entropy</div><div className="num mt-0.5 text-base">{fixed(s.uncertainty?.mean_entropy, 3)}</div></div>
                      <div className="rounded-lg border border-line bg-sunken/60 p-2"><div className="eyebrow">entropy &gt; 0.5</div><div className="num mt-0.5 text-base">{pct(s.uncertainty?.high_uncertainty_percent)}</div></div>
                    </div>
                    <table className="mt-3 w-full text-xs">
                      <tbody>
                        {CLASS_KEYS.map((k) => (
                          <tr key={k} className="border-t border-line/60">
                            <td className="py-1"><span className="flex items-center gap-2"><ClassSwatch k={k} />{CLASS_LABEL[k]}</span></td>
                            <td className="num py-1 text-right text-ink-2">conf {ratio(s.confidence?.per_class_mean[k] ?? null, 1)}</td>
                            <td className="num py-1 text-right text-ink-2">entropy {fixed(s.uncertainty?.per_class_mean_entropy[k] ?? null, 3)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p className="mt-2 text-2xs text-ink-3">{s.uncertainty?.definition} Bright regions are where the model hesitates between classes — typically class boundaries, shadows and distant terrain.</p>
                  </Panel>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
