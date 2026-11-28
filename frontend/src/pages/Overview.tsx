import {
  ArrowRight, BrainCircuit, CheckCircle2, Circle, Cpu, Database, Gauge, ListChecks, Loader2, Map, Mountain, ScanSearch,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { ClassLegend, CompositionBar, EmptyState, ErrorBanner, Loading, Panel, ProgressBar, Segmented, Stat, StatusBadge } from '../components/ui';
import { AnalysisThumb, ModelCard } from '../components/shared';
import { useApp } from '../context/AppContext';
import { useRemote } from '../hooks/usePolling';
import { api } from '../services/api';
import type { ClassKey } from '../types/api';
import { ago, archLabel, CLASS_KEYS, CLASS_LABEL, cx, duration, int, ratio } from '../utils/format';

function Step({ n, title, done, active, to, detail, icon: Icon }: {
  n: number; title: string; done: boolean; active?: boolean; to: string; detail: string; icon: typeof Database;
}) {
  return (
    <Link to={to} className={cx('group relative flex min-w-0 flex-1 flex-col rounded-lg border p-4 transition-colors',
      done ? 'border-good/30 bg-good/[0.05]' : active ? 'border-accent/50 bg-accent/[0.06]' : 'border-line bg-sunken/50 hover:border-line-strong')}>
      <div className="flex items-center justify-between">
        <span className="num text-2xs text-ink-3">Step {String(n).padStart(2, '0')}</span>
        {done ? <CheckCircle2 className="h-4 w-4 text-good" /> : active ? <Loader2 className="h-4 w-4 animate-spin text-accent" /> : <Circle className="h-4 w-4 text-ink-3" />}
      </div>
      <div className="mt-2 flex items-center gap-2 text-sm font-medium text-ink">
        <Icon className="h-4 w-4 text-ink-3 group-hover:text-ink-2" strokeWidth={1.5} /> {title}
      </div>
      <div className="mt-1 text-2xs leading-snug text-ink-3">{detail}</div>
    </Link>
  );
}

export default function Overview() {
  const { connection } = useApp();
  const ov = useRemote(() => api.overview(), [connection], 5000);
  const d = ov.data;

  if (!d) {
    return (
      <>
        <Hero />
        {ov.error ? <ErrorBanner error={ov.error} onRetry={ov.reload} /> : <Loading />}
      </>
    );
  }

  const realModels = d.models.filter((m) => !m.is_fixture);
  const realEvals = d.evaluations.filter((e) => !e.is_fixture);
  const bestTest = realEvals.filter((e) => e.split === 'test').sort((a, b) => (b.mean_iou ?? 0) - (a.mean_iou ?? 0))[0];
  const t = d.training;
  const steps = [
    { title: 'Load AI4Mars', icon: Database, to: '/dataset', done: d.dataset.indexed,
      active: d.dataset.job?.status === 'running',
      detail: d.dataset.indexed ? `${int(d.dataset.total_images)} validated image–label pairs` : 'Download or import, then validate' },
    { title: 'Train models', icon: BrainCircuit, to: '/training', done: realModels.length > 0, active: t.active,
      detail: t.active ? `${archLabel(t.arch)} · epoch ${t.epoch}/${t.epochs_total}` : realModels.length ? `${realModels.length} checkpoint(s)` : 'U-Net and DeepLabV3' },
    { title: 'Evaluate', icon: Gauge, to: '/evaluation', done: realEvals.some((e) => e.split === 'test'),
      detail: realEvals.length ? `${realEvals.length} evaluation(s) stored` : 'Held-out expert test set' },
    { title: 'Analyse images', icon: ScanSearch, to: '/analysis', done: d.recent_analyses.some((a) => !a.is_fixture),
      detail: d.recent_analyses.length ? `${d.recent_analyses.length} recent analyses` : 'Upload a rover photo' },
    { title: 'Plan routes', icon: Map, to: '/navigation', done: false, detail: 'Experimental A* simulator' },
  ];

  return (
    <div className="space-y-6">
      <Hero latestId={d.latest_analysis?.id} latestName={d.latest_analysis?.filename} latestFixture={d.latest_analysis?.is_fixture} />

      <Panel title="Research pipeline" eyebrow="Mission checklist" icon={ListChecks}>
        <div className="flex flex-col gap-2 md:flex-row">
          {steps.map((s, i) => <Step key={s.title} n={i + 1} {...s} />)}
        </div>
      </Panel>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <Stat label="Dataset" icon={Database} value={d.dataset.indexed ? int(d.dataset.total_images) : '—'}
          sub={d.dataset.indexed ? `${int(d.dataset.split_counts.train)} train · ${int(d.dataset.split_counts.val)} val · ${int(d.dataset.split_counts.test)} test` : d.dataset.dataset_found ? 'found, not indexed' : 'not loaded'} />
        <Stat label="Trained models" icon={BrainCircuit} value={realModels.length} tone={realModels.length ? 'default' : 'warn'}
          sub={realModels.length ? realModels.map((m) => archLabel(m.arch)).filter((v, i, a) => a.indexOf(v) === i).join(' · ') : 'none yet'} />
        <Stat label="Best test mIoU" icon={Gauge} tone={bestTest ? 'accent' : 'default'} value={bestTest ? ratio(bestTest.mean_iou, 1) : '—'}
          sub={bestTest ? `${archLabel(bestTest.arch)} · ${bestTest.label_variant} · ${bestTest.images} imgs` : 'no test evaluation yet'} />
        <Stat label="Training" icon={Loader2} tone={t.active ? 'accent' : 'default'} value={t.active ? `${Math.round((t.progress ?? 0) * 100)}%` : t.state === 'idle' ? 'idle' : t.state}
          sub={t.active ? `ETA ${duration(t.eta_seconds)}` : t.name ?? 'no runs yet'} />
        <Stat label="Compute" icon={Cpu} value={d.hardware.cuda_available ? 'CUDA' : 'CPU'}
          sub={d.hardware.cuda_available ? d.hardware.gpus[0]?.name : `${d.hardware.cpu_cores_logical} threads · ${d.hardware.ram_total_gb} GB RAM`} />
      </div>

      {t.active && (
        <Panel title={`Training ${archLabel(t.arch)} — ${t.name}`} icon={BrainCircuit} actions={<Link to="/training" className="btn-ghost text-xs">Open <ArrowRight className="h-3.5 w-3.5" /></Link>}>
          <div className="flex items-center gap-4">
            <ProgressBar value={t.progress ?? 0} className="flex-1" />
            <span className="num text-xs text-ink-2">epoch {t.epoch}/{t.epochs_total}</span>
          </div>
          <div className="mt-2 text-xs text-ink-3">{t.message} · best val mIoU {ratio(t.best_val_miou, 2)}</div>
        </Panel>
      )}

      <div className="grid gap-6 xl:grid-cols-3">
        <Panel title="Model evaluation summary" icon={Gauge} className="xl:col-span-2" padded={false}
          actions={<Link to="/evaluation" className="btn-ghost text-xs">All results <ArrowRight className="h-3.5 w-3.5" /></Link>}>
          {d.evaluations.length === 0 ? (
            <div className="p-4"><EmptyState icon={Gauge} title="No evaluations yet">Evaluate a trained model on the held-out expert test set to see pixel accuracy, mIoU and per-class scores here.</EmptyState></div>
          ) : (
            <div className="overflow-x-auto">
              <table className="table-clean">
                <thead><tr><th>Model</th><th>Split</th><th className="text-right">Images</th><th className="text-right">Pixel acc.</th><th className="text-right">mIoU</th><th className="text-right">Dice</th><th className="text-right">When</th></tr></thead>
                <tbody>
                  {d.evaluations.map((e) => (
                    <tr key={e.id}>
                      <td><div className="flex items-center gap-2"><span className="chip">{archLabel(e.arch)}</span><span className="max-w-[260px] truncate">{e.experiment_name}</span>{e.is_fixture && <span className="chip text-info">fixture</span>}</div></td>
                      <td className="num text-ink-2">{e.split}{e.split === 'test' ? ` · ${e.label_variant}` : ''}</td>
                      <td className="num text-right text-ink-2">{e.images}</td>
                      <td className="num text-right">{ratio(e.pixel_accuracy, 1)}</td>
                      <td className="num text-right font-medium text-accent">{ratio(e.mean_iou, 1)}</td>
                      <td className="num text-right">{ratio(e.mean_dice, 1)}</td>
                      <td className="num text-right text-ink-3">{ago(e.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>

        <Panel title="Recent analyses" icon={Mountain} actions={<Link to="/analysis" className="btn-ghost text-xs">Analyse <ArrowRight className="h-3.5 w-3.5" /></Link>}>
          {d.recent_analyses.length === 0 ? (
            <p className="py-6 text-center text-sm text-ink-3">No images analysed yet.</p>
          ) : (
            <div className="space-y-1">
              {d.recent_analyses.slice(0, 6).map((a) => (
                <Link key={a.id} to={`/terrain?id=${a.id}`} className="block">
                  <AnalysisThumb a={a} />
                  <CompositionBar percent={a.class_percent} height={3} className="-mt-1 mb-1 ml-16 w-auto" />
                </Link>
              ))}
            </div>
          )}
        </Panel>
      </div>

      <Panel title="Available models" icon={BrainCircuit} actions={<Link to="/experiments" className="btn-ghost text-xs">Experiments <ArrowRight className="h-3.5 w-3.5" /></Link>}>
        {d.models.length === 0 ? (
          <EmptyState icon={BrainCircuit} title="No checkpoints yet" action={<Link to={d.dataset.indexed ? '/training' : '/dataset'} className="btn-primary">{d.dataset.indexed ? 'Start training' : 'Load the dataset'}</Link>}>
            {d.dataset.indexed ? 'The dataset is ready. Train U-Net and DeepLabV3 to compare them.' : 'Begin by loading the AI4Mars dataset, then train a model.'}
          </EmptyState>
        ) : (
          <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
            {d.models.slice(0, 8).map((m) => (
              <div key={m.id}>
                <ModelCard model={m} />
                <div className="mt-1 flex items-center justify-between px-1 text-2xs text-ink-3">
                  <StatusBadge state={m.status} />
                  <span className="num">{m.latest_test_evaluation ? `test mIoU ${ratio(m.latest_test_evaluation.mean_iou, 1)} (${m.latest_test_evaluation.label_variant})` : 'not evaluated'}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </Panel>

      {d.latest_experiment && (
        <p className="text-2xs text-ink-3">
          Latest experiment: <Link className="text-ink-2 underline decoration-line-strong underline-offset-2 hover:text-accent" to={`/experiments?id=${d.latest_experiment.id}`}>{d.latest_experiment.name}</Link>{' '}
          · {d.latest_experiment.status} · {d.latest_experiment.epochs_completed}/{d.latest_experiment.epochs_total} epochs · {ago(d.latest_experiment.created_at)}.
          Class order is fixed: {Object.values(CLASS_LABEL).join(', ')}.
        </p>
      )}
    </div>
  );
}

/** A real prediction bundled with the app: DeepLabV3 on a held-out AI4Mars test frame (Curiosity NavCam). */
const SAMPLE = {
  frame: 'NLA_409036068EDR',
  image: `${import.meta.env.BASE_URL}samples/navcam-NLA_409036068.jpg`,
  classes: `${import.meta.env.BASE_URL}samples/navcam-NLA_409036068-classes.png`,
  percent: { soil: 0, bedrock: 68.4, sand: 15.5, big_rock: 16.1 } as Record<ClassKey, number>,
};

function hexRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.replace('#', ''), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Draws the bundled NavCam frame with its class map blended on top, in the user's class colours. */
function SampleOverlay({ showMask }: { showMask: boolean }) {
  const { colors } = useApp();
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    let alive = true;
    const load = (src: string) => new Promise<HTMLImageElement>((res, rej) => {
      const img = new Image(); img.onload = () => res(img); img.onerror = rej; img.src = src;
    });
    Promise.all([load(SAMPLE.image), load(SAMPLE.classes)]).then(([img, cls]) => {
      const c = canvas.current;
      if (!alive || !c) return;
      const w = img.naturalWidth, h = img.naturalHeight;
      c.width = w; c.height = h;
      const ctx = c.getContext('2d');
      if (!ctx) return;
      ctx.drawImage(img, 0, 0);
      if (!showMask) return;
      const base = ctx.getImageData(0, 0, w, h);
      const tmp = document.createElement('canvas'); tmp.width = w; tmp.height = h;
      const tctx = tmp.getContext('2d');
      if (!tctx) return;
      tctx.drawImage(cls, 0, 0, w, h);
      const ids = tctx.getImageData(0, 0, w, h).data;
      const pal = CLASS_KEYS.map((k) => hexRgb(colors[k]));
      const px = base.data;
      for (let i = 0; i < px.length; i += 4) {
        const rgb = pal[ids[i]];
        if (!rgb) continue;
        px[i] = px[i] * 0.5 + rgb[0] * 0.5; px[i + 1] = px[i + 1] * 0.5 + rgb[1] * 0.5; px[i + 2] = px[i + 2] * 0.5 + rgb[2] * 0.5;
      }
      ctx.putImageData(base, 0, 0);
    }).catch(() => undefined);
    return () => { alive = false; };
  }, [colors, showMask]);
  return <canvas ref={canvas} role="img" aria-label={`Rover NavCam frame ${SAMPLE.frame}${showMask ? ' with predicted terrain classes' : ''}`}
    className="block aspect-square w-full bg-sunken" />;
}

function Hero({ latestId, latestName, latestFixture }: { latestId?: string; latestName?: string; latestFixture?: boolean }) {
  const [view, setView] = useState<'overlay' | 'frame'>('overlay');
  return (
    <section className="panel relative overflow-hidden">
      <div className="relative grid gap-8 p-6 md:p-8 lg:grid-cols-[1.2fr_1fr] lg:items-center">
        <div>
          <div className="eyebrow mb-4">AI4Mars · semantic segmentation · runs on your machine</div>
          <h1 className="text-balance text-3xl font-medium leading-tight text-ink md:text-4xl">
            Segment Mars rover images into soil, bedrock, sand and big rock
          </h1>
          <p className="mt-4 max-w-xl text-sm leading-relaxed text-ink-2">
            Train U-Net and DeepLabV3 on NASA JPL&apos;s AI4Mars dataset, measure them on expert-labelled held-out images, and
            explore how terrain maps could inform route planning. Everything runs on your machine; every number shown is computed
            here. <Link to="/docs" className="text-ink underline decoration-line-strong underline-offset-2 hover:decoration-ink-2">Read the methodology</Link>
          </p>
          <div className="mt-6 flex flex-wrap gap-2">
            <Link to="/analysis" className="btn-primary"><ScanSearch className="h-4 w-4" strokeWidth={1.5} /> Analyse an image</Link>
            <Link to="/dataset" className="btn-secondary"><Database className="h-4 w-4" strokeWidth={1.5} /> Explore the dataset</Link>
          </div>
        </div>
        <figure className="w-full lg:max-w-[440px] lg:justify-self-end">
          {latestId ? (
            <Link to={`/terrain?id=${latestId}`} className="block">
              <div className="panel-ticks relative rounded-xl border border-line-strong p-1">
                <img src={api.analysisImageUrl(latestId, 'overlay', { alpha: 0.55 })} alt="Latest segmentation result" className="aspect-square w-full rounded-lg object-cover" />
              </div>
              <figcaption className="mt-3 flex items-center justify-between gap-3 text-xs text-ink-3">
                <span className="num truncate">{latestName}</span>
                <span>{latestFixture ? 'Test fixture' : 'Latest result'}</span>
              </figcaption>
            </Link>
          ) : (
            <>
              <div className="panel-ticks relative rounded-xl border border-line-strong p-1">
                <div className="overflow-hidden rounded-lg"><SampleOverlay showMask={view === 'overlay'} /></div>
              </div>
              <figcaption className="mt-3 space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <ClassLegend values={{ soil: `${SAMPLE.percent.soil}%`, bedrock: `${SAMPLE.percent.bedrock}%`, sand: `${SAMPLE.percent.sand}%`, big_rock: `${SAMPLE.percent.big_rock}%` }} />
                  <Segmented size="sm" ariaLabel="Sample view" value={view} onChange={setView}
                    options={[{ value: 'overlay', label: 'Prediction' }, { value: 'frame', label: 'NavCam frame' }]} />
                </div>
                <p className="text-xs leading-snug text-ink-3">
                  Sample DeepLabV3 prediction, held-out test frame <span className="num">{SAMPLE.frame}</span>
                </p>
              </figcaption>
            </>
          )}
        </figure>
      </div>
    </section>
  );
}
