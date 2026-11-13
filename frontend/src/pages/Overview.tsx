import {
  ArrowRight, BrainCircuit, CheckCircle2, Circle, Cpu, Database, Gauge, Loader2, Map, Mountain, ScanSearch, Sparkles,
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { CompositionBar, EmptyState, ErrorBanner, Loading, Panel, ProgressBar, Stat, StatusBadge } from '../components/ui';
import { AnalysisThumb, ModelCard } from '../components/shared';
import { useApp } from '../context/AppContext';
import { useRemote } from '../hooks/usePolling';
import { api } from '../services/api';
import { ago, archLabel, CLASS_LABEL, cx, duration, int, ratio } from '../utils/format';

function Step({ n, title, done, active, to, detail, icon: Icon }: {
  n: number; title: string; done: boolean; active?: boolean; to: string; detail: string; icon: typeof Database;
}) {
  return (
    <Link to={to} className={cx('group relative flex min-w-0 flex-1 flex-col rounded-lg border p-3 transition-colors',
      done ? 'border-good/30 bg-good/[0.05]' : active ? 'border-accent/50 bg-accent/[0.06]' : 'border-line bg-sunken/50 hover:border-line-strong')}>
      <div className="flex items-center justify-between">
        <span className="num text-2xs text-ink-3">STEP {String(n).padStart(2, '0')}</span>
        {done ? <CheckCircle2 className="h-4 w-4 text-good" /> : active ? <Loader2 className="h-4 w-4 animate-spin text-accent" /> : <Circle className="h-4 w-4 text-ink-3" />}
      </div>
      <div className="mt-2 flex items-center gap-2 text-sm font-medium text-ink">
        <Icon className="h-4 w-4 text-ink-3 group-hover:text-accent" strokeWidth={1.75} /> {title}
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

      <Panel title="Research pipeline" eyebrow="mission checklist" icon={Sparkles}>
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

function Hero({ latestId, latestName, latestFixture }: { latestId?: string; latestName?: string; latestFixture?: boolean }) {
  return (
    <section className="panel panel-ticks relative overflow-hidden">
      <div className="absolute inset-0 bg-grid opacity-70" />
      <div className="absolute -right-32 -top-40 h-96 w-96 rounded-full bg-accent/[0.08] blur-3xl" />
      <div className="relative grid gap-6 p-6 md:p-8 lg:grid-cols-[1.25fr_1fr] lg:items-center">
        <div>
          <div className="eyebrow mb-3 flex items-center gap-2 text-accent">
            <span className="h-1.5 w-1.5 rounded-full bg-accent animate-pulse2" /> AI4Mars · semantic segmentation · local inference
          </div>
          <h1 className="text-balance text-3xl font-semibold leading-[1.1] text-ink md:text-[2.5rem]">
            Pixel-level terrain intelligence for Mars rover imagery
          </h1>
          <p className="mt-4 max-w-xl text-sm leading-relaxed text-ink-2">
            Train U-Net and DeepLabV3 on NASA JPL&apos;s AI4Mars dataset, segment rover photographs into soil, bedrock, sand and big
            rocks, measure performance on expert-labelled held-out images, and explore how terrain maps could inform route planning.
            Everything runs on your machine; every number shown is computed here.
          </p>
          <div className="mt-6 flex flex-wrap gap-2">
            <Link to="/analysis" className="btn-primary"><ScanSearch className="h-4 w-4" /> Analyse an image</Link>
            <Link to="/dataset" className="btn-secondary"><Database className="h-4 w-4" /> Explore the dataset</Link>
            <Link to="/docs" className="btn-ghost">Methodology <ArrowRight className="h-4 w-4" /></Link>
          </div>
        </div>
        <div className="relative">
          {latestId ? (
            <Link to={`/terrain?id=${latestId}`} className="group block">
              <div className="relative overflow-hidden rounded-xl border border-line-strong">
                <img src={api.analysisImageUrl(latestId, 'overlay', { alpha: 0.55 })} alt="Latest segmentation result" className="aspect-square w-full object-cover transition-transform duration-700 group-hover:scale-[1.03]" />
                <div className="pointer-events-none absolute inset-x-0 h-1/3 bg-gradient-to-b from-transparent via-accent/10 to-transparent animate-scan" />
                <div className="absolute inset-x-0 bottom-0 flex items-center justify-between bg-gradient-to-t from-black/80 to-transparent p-3 text-xs text-white">
                  <span className="truncate font-mono">{latestName}</span>
                  <span className="chip border-white/20 bg-black/40 text-white">{latestFixture ? 'fixture' : 'latest result'}</span>
                </div>
              </div>
            </Link>
          ) : (
            <div className="grid aspect-[4/3] place-items-center rounded-xl border border-dashed border-line-strong bg-sunken/70 p-6 text-center">
              <div>
                <ScanSearch className="mx-auto h-8 w-8 text-ink-3" strokeWidth={1.4} />
                <p className="mt-3 text-sm text-ink-2">Your latest segmentation will appear here.</p>
                <p className="mt-1 text-xs text-ink-3">No analysis has been run yet.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
