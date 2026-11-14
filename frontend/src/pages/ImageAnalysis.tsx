import { ArrowRight, History, ImageIcon, Map, Mountain, Play, ScanSearch, Trash2, X } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { ConfidenceHistogram } from '../charts/Charts';
import SegmentationViewer from '../components/SegmentationViewer';
import TerrainSummary from '../components/TerrainSummary';
import { AnalysisThumb, Dropzone, ModelSelect, NoModels } from '../components/shared';
import { Callout, ErrorBanner, Field, Loading, PageHeader, Panel, Segmented } from '../components/ui';
import { useApp } from '../context/AppContext';
import { useLocalState, useRemote } from '../hooks/usePolling';
import { api } from '../services/api';
import type { Analysis } from '../types/api';
import { bytes, cx } from '../utils/format';

const MAX_MB = 20;
const OK_TYPES = ['image/jpeg', 'image/png', 'image/jpg'];

export function validateFile(f: File): string | null {
  const ext = f.name.split('.').pop()?.toLowerCase() ?? '';
  if (!['jpg', 'jpeg', 'png'].includes(ext) || (f.type && !OK_TYPES.includes(f.type))) {
    return `“${f.name}” is not a JPG, JPEG or PNG image.`;
  }
  if (f.size === 0) return 'The file is empty.';
  if (f.size > MAX_MB * 1024 * 1024) return `The file is ${bytes(f.size)}; the limit is ${MAX_MB} MB.`;
  return null;
}

export default function ImageAnalysis() {
  const { connection } = useApp();
  const [params, setParams] = useSearchParams();
  const models = useRemote(() => api.models(), [connection]);
  const history = useRemote(() => api.analyses(30), [connection]);
  const ds = useRemote(() => api.datasetStatus(), [connection]);
  const [modelId, setModelId] = useLocalState<string>('mti.analysis.model', '');
  const [source, setSource] = useState<'upload' | 'dataset'>('upload');
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [sampleId, setSampleId] = useState<string>('');
  const [fileError, setFileError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<Error | null>(null);
  const [result, setResult] = useState<Analysis | null>(null);
  const [progress, setProgress] = useState('');
  useEffect(() => {
    const on = (e: Event) => setProgress(String((e as CustomEvent).detail ?? ''));
    window.addEventListener('mti-demo-progress', on);
    return () => window.removeEventListener('mti-demo-progress', on);
  }, []);

  const list = useMemo(() => models.data?.models ?? [], [models.data]);
  const real = list.filter((m) => !m.is_fixture);
  const samples = useRemote(() => (ds.data?.indexed ? api.datasetSamples({ split: 'test', limit: 24 }) : Promise.resolve(null)), [ds.data?.indexed]);

  // pick a sensible default model: the remembered one if it still exists, else the best real model
  useEffect(() => {
    if (!list.length) return;
    if (!list.some((m) => m.id === modelId)) {
      const best = [...real].sort((a, b) => (b.best_val_miou ?? 0) - (a.best_val_miou ?? 0))[0] ?? list[0];
      setModelId(best.id);
    }
  }, [list, real, modelId, setModelId]);

  // deep link: /analysis?id=<analysis>
  const linked = params.get('id');
  useEffect(() => {
    if (linked && linked !== result?.id) api.analysis(linked).then(setResult).catch(() => undefined);
  }, [linked, result?.id]);

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  function pick(f: File) {
    setRunError(null);
    const err = validateFile(f);
    setFileError(err);
    if (err) {
      setFile(null);
      setPreview(null);
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }

  async function run() {
    if (!modelId) return;
    setRunning(true);
    setRunError(null);
    setProgress('');
    try {
      const a = source === 'upload' && file ? await api.predictUpload(modelId, file) : await api.predictSample(modelId, sampleId);
      setResult(a);
      setParams({ id: a.id }, { replace: true });
      void history.reload();
    } catch (e) {
      setRunError(e as Error);
    } finally {
      setRunning(false);
    }
  }

  async function remove(id: string) {
    await api.deleteAnalysis(id).catch(() => undefined);
    if (result?.id === id) {
      setResult(null);
      setParams({}, { replace: true });
    }
    void history.reload();
  }

  const canRun = !!modelId && !running && (source === 'upload' ? !!file : !!sampleId);

  return (
    <div>
      <PageHeader eyebrow="02 · inference" title="Image Analysis"
        description="Upload a Mars rover photograph and segment every pixel into soil, bedrock, sand or big rock with a model you trained locally. Confidence is the model's own softmax score: it is not calibrated, and a confident prediction can still be wrong." />

      {models.error && <ErrorBanner error={models.error} onRetry={models.reload} className="mb-4" />}
      {models.loading && !models.data ? <Loading /> : list.length === 0 ? <NoModels /> : (
        <div className="grid gap-6 xl:grid-cols-[380px_1fr]">
          <div className="space-y-4">
            <Panel title="Input" icon={ImageIcon} ticks>
              <div className="space-y-4">
                <Field label="Model" htmlFor="model">
                  <ModelSelect id="model" models={list} value={modelId} onChange={setModelId} />
                </Field>
                {list.find((m) => m.id === modelId)?.is_fixture && (
                  <Callout tone="info">The selected model is the untrained test fixture. Its output is not a real prediction.</Callout>
                )}
                <Segmented ariaLabel="Image source" value={source} onChange={(v) => { setSource(v); setRunError(null); }}
                  options={[{ value: 'upload', label: 'Upload image' }, { value: 'dataset', label: 'Dataset test image', disabled: !ds.data?.indexed }]} />
                {source === 'upload' ? (
                  <>
                    <Dropzone onFile={pick} disabled={running} maxMb={MAX_MB} />
                    {fileError && <ErrorBanner error={fileError} />}
                    {file && preview && (
                      <div className="flex items-center gap-3 rounded-lg border border-line bg-sunken/60 p-2">
                        <img src={preview} alt="Selected image preview" className="h-16 w-16 rounded-md object-cover" />
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-sm text-ink">{file.name}</div>
                          <div className="num text-2xs text-ink-3">{bytes(file.size)} · {file.type || 'unknown type'}</div>
                        </div>
                        <button className="btn-ghost p-1.5" onClick={() => { setFile(null); setPreview(null); }} aria-label="Remove file"><X className="h-4 w-4" /></button>
                      </div>
                    )}
                  </>
                ) : (
                  <div>
                    <p className="mb-2 text-xs text-ink-3">Expert-labelled test images (held out from training). Ground truth can be compared in the Dataset Explorer.</p>
                    <div className="grid max-h-64 grid-cols-4 gap-1.5 overflow-y-auto pr-1">
                      {samples.data?.items.map((s) => (
                        <button key={s.id} onClick={() => setSampleId(s.id)} title={s.id}
                          className={cx('overflow-hidden rounded-md border-2 transition-colors', sampleId === s.id ? 'border-accent' : 'border-transparent hover:border-line-strong')}>
                          <img src={api.sampleImageUrl(s.id, 'image', undefined, 160)} alt={s.id} loading="lazy" className="aspect-square w-full object-cover" />
                        </button>
                      ))}
                    </div>
                  </div>
                )}
                <button className="btn-primary w-full py-2.5" disabled={!canRun} onClick={run} data-testid="run-analysis">
                  {running ? <><span className="h-4 w-4 animate-spin rounded-full border-2 border-accent-ink/30 border-t-accent-ink" /> {progress || 'Running inference…'}</> : <><Play className="h-4 w-4" /> Run analysis</>}
                </button>
                {runError && <ErrorBanner error={runError} />}
              </div>
            </Panel>

            <Panel title="History" icon={History} padded={false}>
              <div className="max-h-[420px] space-y-0.5 overflow-y-auto p-2">
                {(history.data?.analyses ?? []).length === 0 && <p className="py-6 text-center text-xs text-ink-3">No analyses yet.</p>}
                {history.data?.analyses.map((a) => (
                  <div key={a.id} className="group relative">
                    <AnalysisThumb a={a} active={a.id === result?.id} onClick={() => api.analysis(a.id).then((r) => { setResult(r); setParams({ id: r.id }, { replace: true }); })} />
                    <button className="btn-ghost absolute right-1 top-1/2 hidden -translate-y-1/2 p-1.5 group-hover:inline-flex" onClick={() => remove(a.id)} aria-label="Delete analysis">
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            </Panel>
          </div>

          <div className="min-w-0 space-y-6">
            {running && !result && (
              <div className="panel relative grid aspect-video place-items-center overflow-hidden">
                {preview && <img src={preview} alt="" className="absolute inset-0 h-full w-full object-contain opacity-40" />}
                <div className="absolute inset-x-0 h-1/3 bg-gradient-to-b from-transparent via-accent/20 to-transparent animate-scan" />
                <div className="relative text-sm text-ink-2">Segmenting pixels…</div>
              </div>
            )}
            {!result && !running && (
              <div className="panel grid min-h-[420px] place-items-center bg-grid p-8 text-center">
                <div>
                  <ScanSearch className="mx-auto h-10 w-10 text-ink-3" strokeWidth={1.3} />
                  <h3 className="mt-4 text-base font-semibold">Results appear here</h3>
                  <p className="mt-1 max-w-sm text-sm text-ink-3">Choose an image and a model, then run the analysis. You will get the mask, an overlay, a swipe comparison, confidence and uncertainty maps, and terrain statistics.</p>
                </div>
              </div>
            )}
            {result && (
              <>
                <Panel title={result.filename} eyebrow={`analysis ${result.id}`} icon={ScanSearch} ticks
                  actions={<>
                    <Link to={`/terrain?id=${result.id}`} className="btn-ghost text-xs"><Mountain className="h-3.5 w-3.5" /> Terrain detail</Link>
                    <Link to={`/navigation?id=${result.id}`} className="btn-ghost text-xs"><Map className="h-3.5 w-3.5" /> Plan route</Link>
                  </>}>
                  <SegmentationViewer analysis={result} />
                </Panel>
                <div className="grid gap-6 2xl:grid-cols-[1.3fr_1fr]">
                  <Panel title="Terrain distribution" eyebrow="computed from this prediction">
                    <TerrainSummary analysis={result} />
                  </Panel>
                  <Panel title="Prediction confidence" eyebrow="softmax · uncalibrated">
                    {result.stats.confidence && <ConfidenceHistogram histogram={result.stats.confidence.histogram} />}
                    <p className="mt-3 text-xs leading-relaxed text-ink-3">
                      Share of pixels by the probability the model gave its chosen class. With four classes the minimum is 25%.
                      {result.stats.confidence && ` ${result.stats.confidence.high_confidence_percent.toFixed(1)}% of pixels are at ≥90% confidence.`}{' '}
                      Calibration has not been tested, so these values are not probabilities of being correct.
                    </p>
                    <Link to={`/terrain?id=${result.id}`} className="btn-secondary mt-4 w-full text-xs">Spatial analysis & uncertainty map <ArrowRight className="h-3.5 w-3.5" /></Link>
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
