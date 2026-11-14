import {
  BrainCircuit, Calculator, ChevronDown, Cpu, Pause, Play, Rocket, Square, Terminal, Timer, TrendingDown,
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { LossChart, PerClassIouHistory, ValMetricChart } from '../charts/Charts';
import {
  Callout, EmptyState, ErrorBanner, Field, KV, Loading, PageHeader, Panel, ProgressBar, Segmented, Stat, StatusBadge, Toggle,
} from '../components/ui';
import { useApp } from '../context/AppContext';
import { useRemote } from '../hooks/usePolling';
import { api, toTrainRequest } from '../services/api';
import type { Estimate, TrainConfig, TrainingStatus } from '../types/api';
import { archLabel, compact, cx, duration, fixed, int, ratio } from '../utils/format';
import { IS_DEMO } from '../demo/env';

const ARCH_INFO = {
  unet: {
    title: 'U-Net',
    blurb: 'Encoder–decoder with skip connections (Ronneberger et al., 2015). Trained from scratch; size set by base channels and depth.',
  },
  deeplabv3: {
    title: 'DeepLabV3',
    blurb: 'Atrous convolutions + ASPP (Chen et al., 2017) via Torchvision. Pretrained backbone with a new 4-class head.',
  },
} as const;

const LRS = [3e-4, 5e-4, 1e-3, 2e-3, 5e-3];

export default function ModelTraining() {
  const { connection } = useApp();
  const defaults = useRemote(() => api.trainingDefaults(), [connection]);
  const [status, setStatus] = useState<TrainingStatus>();
  const [statusError, setStatusError] = useState<Error>();
  const active = !!status?.active;

  useEffect(() => {
    let alive = true;
    const tick = () => api.trainingStatus().then((s) => { if (alive) { setStatus(s); setStatusError(undefined); } }).catch((e) => alive && setStatusError(e));
    tick();
    const t = window.setInterval(tick, active ? 1500 : 5000);
    return () => { alive = false; window.clearInterval(t); };
  }, [active, connection]);

  return (
    <div>
      <PageHeader eyebrow="04 · learning" title="Model Training"
        description="Configure and run real training jobs. Training runs in a separate background process, so this page stays responsive; a checkpoint is saved after every completed epoch and is kept if you stop the run." />
      {defaults.error && <ErrorBanner error={defaults.error} onRetry={defaults.reload} className="mb-4" />}
      {!defaults.data ? (defaults.error ? null : <Loading />) : defaults.data.train_available < 2 ? (
        <EmptyState icon={BrainCircuit} title="Load the dataset first" action={<Link to="/dataset" className="btn-primary">Open Dataset Explorer</Link>}>
          Training needs the validated AI4Mars train/validation splits. Download a subset or import the archive, then come back.
        </EmptyState>
      ) : (
        <div className="grid gap-6 xl:grid-cols-[400px_minmax(0,1fr)]">
          {IS_DEMO ? (
            <Panel title="Training needs the full app" icon={BrainCircuit} ticks>
              <p className="text-sm leading-relaxed text-ink-2">Training runs on the Python backend with the AI4Mars training images, which are not part of this browser edition. On the right is the most recent real training run from the local platform: its loss curves, validation metrics and full log.</p>
              <p className="mt-3 text-sm text-ink-2">To train your own models, install the app from <a className="text-accent underline" href="https://github.com/aarushagrawal267-maker/AI4Mars" target="_blank" rel="noreferrer">GitHub</a> and run <code className="num">setup.ps1</code>.</p>
            </Panel>
          ) : (
            <ConfigPanel defaults={defaults.data} disabled={active} onStarted={() => api.trainingStatus().then(setStatus)} />
          )}
          <Monitor status={status} error={statusError} />
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------- config form */

function ConfigPanel({ defaults, disabled, onStarted }: {
  defaults: NonNullable<ReturnType<typeof useRemote<Awaited<ReturnType<typeof api.trainingDefaults>>>>['data']>;
  disabled: boolean;
  onStarted: () => void;
}) {
  const [cfg, setCfg] = useState<TrainConfig>(() => ({
    ...defaults.config,
    model_config: defaults.config.arch === 'unet' ? { base_channels: 16, depth: 4 } : {},
  }));
  const [name, setName] = useState('');
  const [useSubset, setUseSubset] = useState(defaults.config.max_train_samples !== null);
  const [advanced, setAdvanced] = useState(false);
  const [estimate, setEstimate] = useState<Estimate>();
  const [estimating, setEstimating] = useState(false);
  const [confirmLarge, setConfirmLarge] = useState(false);
  const [error, setError] = useState<Error>();
  const [starting, setStarting] = useState(false);

  const set = <K extends keyof TrainConfig>(k: K, v: TrainConfig[K]) => {
    setCfg((c) => ({ ...c, [k]: v }));
    setEstimate(undefined);
  };
  const setOpt = (k: string, v: unknown) => {
    setCfg((c) => ({ ...c, model_config: { ...c.model_config, [k]: v } }));
    setEstimate(undefined);
  };
  const setArch = (arch: TrainConfig['arch']) => {
    setCfg((c) => ({ ...c, arch, model_config: arch === 'unet' ? { base_channels: 16, depth: 4 } : { backbone: 'mobilenet_v3_large', pretrained: 'coco' } }));
    setEstimate(undefined);
  };

  const effective: TrainConfig = useMemo(() => ({
    ...cfg,
    max_train_samples: useSubset ? cfg.max_train_samples ?? Math.min(1000, defaults.train_available) : null,
    max_val_samples: useSubset ? cfg.max_val_samples ?? Math.min(200, defaults.val_available) : null,
  }), [cfg, useSubset, defaults]);

  async function doEstimate() {
    setEstimating(true);
    setError(undefined);
    try {
      setEstimate(await api.trainingEstimate(toTrainRequest(effective)));
    } catch (e) {
      setError(e as Error);
    } finally {
      setEstimating(false);
    }
  }

  async function start() {
    setStarting(true);
    setError(undefined);
    try {
      await api.trainingStart({ ...toTrainRequest(effective), name: name.trim() || undefined, confirm_large: confirmLarge });
      setEstimate(undefined);
      setConfirmLarge(false);
      onStarted();
    } catch (e) {
      setError(e as Error);
    } finally {
      setStarting(false);
    }
  }

  const mc = cfg.model_config as Record<string, unknown>;
  return (
    <Panel title="Experiment configuration" icon={BrainCircuit} ticks>
      <fieldset disabled={disabled} className="space-y-5 disabled:opacity-60">
        <Callout tone="info">{defaults.reason}</Callout>

        <div>
          <div className="mb-1.5 text-xs font-medium text-ink-2">Architecture</div>
          <div className="grid grid-cols-2 gap-2">
            {(['unet', 'deeplabv3'] as const).map((a) => (
              <button key={a} type="button" onClick={() => setArch(a)} aria-pressed={cfg.arch === a}
                className={cx('rounded-lg border p-3 text-left transition-colors', cfg.arch === a ? 'border-accent/70 bg-accent/[0.07]' : 'border-line bg-sunken/50 hover:border-line-strong')}>
                <div className="flex items-center justify-between"><span className="text-sm font-semibold">{ARCH_INFO[a].title}</span>
                  <span className={cx('h-3 w-3 rounded-full border-2', cfg.arch === a ? 'border-accent bg-accent' : 'border-line-strong')} /></div>
                <p className="mt-1 text-2xs leading-snug text-ink-3">{ARCH_INFO[a].blurb}</p>
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          {cfg.arch === 'unet' ? (
            <>
              <Field label="Base channels" hint="16 light · 32 default · 64 original">
                <select className="field" value={String(mc.base_channels ?? 32)} onChange={(e) => setOpt('base_channels', Number(e.target.value))}>
                  {[16, 32, 64].map((v) => <option key={v} value={v}>{v}</option>)}
                </select>
              </Field>
              <Field label="Depth" hint="down/up-sampling stages">
                <select className="field" value={String(mc.depth ?? 4)} onChange={(e) => setOpt('depth', Number(e.target.value))}>
                  {[3, 4, 5].map((v) => <option key={v} value={v}>{v}</option>)}
                </select>
              </Field>
            </>
          ) : (
            <>
              <Field label="Backbone" hint="MobileNetV3 is laptop friendly">
                <select className="field" value={String(mc.backbone ?? 'mobilenet_v3_large')} onChange={(e) => setOpt('backbone', e.target.value)}>
                  <option value="mobilenet_v3_large">MobileNetV3-Large</option>
                  <option value="resnet50">ResNet-50</option>
                </select>
              </Field>
              <Field label="Initialisation" hint="pretrained weights download once">
                <select className="field" value={String(mc.pretrained ?? 'coco')} onChange={(e) => setOpt('pretrained', e.target.value)}>
                  <option value="coco">Pretrained (COCO)</option>
                  <option value="none">Random (from scratch)</option>
                </select>
              </Field>
            </>
          )}
        </div>

        <div className="space-y-2.5 rounded-lg border border-line bg-sunken/40 p-3">
          <Toggle checked={useSubset} onChange={(v) => { setUseSubset(v); setEstimate(undefined); }} label="Use a subset of the training split"
            description={`${int(defaults.train_available)} training and ${int(defaults.val_available)} validation pairs available`} />
          {useSubset && (
            <div className="grid grid-cols-2 gap-3 pt-1">
              <Field label="Train samples">
                <input type="number" className="field num" min={2} max={defaults.train_available} value={effective.max_train_samples ?? ''}
                  onChange={(e) => set('max_train_samples', Math.max(2, Math.min(defaults.train_available, Number(e.target.value) || 2)))} />
              </Field>
              <Field label="Val samples">
                <input type="number" className="field num" min={2} max={defaults.val_available} value={effective.max_val_samples ?? ''}
                  onChange={(e) => set('max_val_samples', Math.max(2, Math.min(defaults.val_available, Number(e.target.value) || 2)))} />
              </Field>
            </div>
          )}
        </div>

        <div className="grid grid-cols-2 gap-3">
          <Field label={<span className="flex justify-between"><span>Epochs</span><span className="num text-ink">{cfg.epochs}</span></span>}>
            <input type="range" min={1} max={60} value={cfg.epochs} onChange={(e) => set('epochs', Number(e.target.value))} className="w-full" />
          </Field>
          <Field label="Batch size">
            <select className="field" value={cfg.batch_size} onChange={(e) => set('batch_size', Number(e.target.value))}>
              {[2, 4, 8, 12, 16, 24, 32].map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </Field>
          <Field label="Learning rate">
            <select className="field num" value={cfg.learning_rate} onChange={(e) => set('learning_rate', Number(e.target.value))}>
              {(LRS.includes(cfg.learning_rate) ? LRS : [...LRS, cfg.learning_rate]).map((v) => <option key={v} value={v}>{v.toExponential(0)}</option>)}
            </select>
          </Field>
          <Field label="Input resolution">
            <select className="field" value={cfg.resolution} onChange={(e) => set('resolution', Number(e.target.value))}>
              {defaults.options.resolutions.map((r) => <option key={r} value={r}>{r} × {r}px</option>)}
            </select>
          </Field>
        </div>
        <Field label="Device" hint={defaults.cuda_available ? 'A CUDA GPU was detected.' : 'No CUDA GPU detected: CPU only.'}>
          <Segmented value={cfg.device} onChange={(v) => set('device', v)} options={defaults.options.devices.map((d) => ({ value: d as TrainConfig['device'], label: d.toUpperCase() }))} />
        </Field>

        <button type="button" className="flex w-full items-center justify-between text-xs font-medium text-ink-2 hover:text-ink" onClick={() => setAdvanced(!advanced)}>
          Loss, schedule & regularisation <ChevronDown className={cx('h-4 w-4 transition-transform', advanced && 'rotate-180')} />
        </button>
        {advanced && (
          <div className="grid grid-cols-2 gap-3 animate-fade-in">
            <Field label="Dice loss weight" hint="0 = cross-entropy only">
              <input type="number" step={0.1} min={0} max={5} className="field num" value={cfg.dice_weight} onChange={(e) => set('dice_weight', Number(e.target.value))} />
            </Field>
            <Field label="Class weighting" hint="counter big-rock imbalance">
              <select className="field" value={cfg.class_weighting} onChange={(e) => set('class_weighting', e.target.value as TrainConfig['class_weighting'])}>
                <option value="none">None</option><option value="inverse_sqrt">Inverse √frequency</option>
              </select>
            </Field>
            <Field label="LR schedule">
              <select className="field" value={cfg.scheduler} onChange={(e) => set('scheduler', e.target.value as TrainConfig['scheduler'])}>
                <option value="cosine">Cosine annealing</option><option value="plateau">Reduce on plateau</option><option value="none">Constant</option>
              </select>
            </Field>
            <Field label="Early stopping patience" hint="epochs without val mIoU gain; 0 = off">
              <input type="number" min={0} max={100} className="field num" value={cfg.early_stopping_patience} onChange={(e) => set('early_stopping_patience', Number(e.target.value))} />
            </Field>
            <Field label="Weight decay">
              <input type="number" step={0.0001} min={0} max={1} className="field num" value={cfg.weight_decay} onChange={(e) => set('weight_decay', Number(e.target.value))} />
            </Field>
            <Field label="Random seed">
              <input type="number" min={0} className="field num" value={cfg.seed} onChange={(e) => set('seed', Number(e.target.value))} />
            </Field>
            <div className="col-span-2"><Toggle checked={cfg.augment} onChange={(v) => set('augment', v)} label="Data augmentation" description="horizontal flip, scale-crop, brightness/contrast, noise" /></div>
          </div>
        )}

        <Field label="Experiment name (optional)">
          <input className="field" maxLength={120} placeholder={`${archLabel(cfg.arch)} ${cfg.resolution}px`} value={name} onChange={(e) => setName(e.target.value)} />
        </Field>

        <div className="space-y-3 border-t border-line/70 pt-4">
          <button type="button" className="btn-secondary w-full" onClick={doEstimate} disabled={estimating}>
            <Calculator className="h-4 w-4" /> {estimating ? 'Measuring on this machine…' : 'Estimate workload'}
          </button>
          {estimate && <EstimateCard e={estimate} />}
          {estimate?.large_job && (
            <label className="flex items-start gap-2 text-xs text-warn"><input type="checkbox" className="mt-0.5" checked={confirmLarge} onChange={(e) => setConfirmLarge(e.target.checked)} />
              I understand this run is estimated to take {duration(estimate.estimated_total_seconds)} and want to start it.</label>
          )}
          <button type="button" className="btn-primary w-full py-2.5" onClick={start}
            disabled={starting || !estimate || !estimate.fits_in_memory || (estimate.large_job && !confirmLarge)} data-testid="start-training">
            <Rocket className="h-4 w-4" /> {starting ? 'Starting…' : 'Start training'}
          </button>
          {!estimate && <p className="text-center text-2xs text-ink-3">Estimate the workload first — large jobs are never started silently.</p>}
          {error && <ErrorBanner error={error.message.replace(/^LARGE_JOB: /, '')} />}
        </div>
      </fieldset>
      {disabled && <p className="mt-3 text-center text-xs text-ink-3">A run is in progress. Stop it to start another.</p>}
    </Panel>
  );
}

function EstimateCard({ e }: { e: Estimate }) {
  return (
    <div className={cx('rounded-lg border p-3 animate-fade-in', e.large_job ? 'border-warn/40 bg-warn/[0.06]' : 'border-line bg-sunken/60')}>
      {e.error ? <p className="text-xs text-bad">{e.error}</p> : (
        <>
          <div className="flex items-baseline justify-between">
            <span className="eyebrow">estimated duration</span>
            <span className={cx('num text-xl', e.large_job ? 'text-warn' : 'text-ink')}>{duration(e.estimated_total_seconds)}</span>
          </div>
          <KV className="mt-2 text-xs" items={[
            ['Device', e.device],
            ['Parameters', compact(e.parameters)],
            ['Train / val samples', `${int(e.train_samples)} / ${int(e.val_samples)}`],
            ['Steps', `${int(e.batches_per_epoch)} per epoch · ${int(e.total_steps)} total`],
            ['Measured step time', e.seconds_per_step ? `${e.seconds_per_step.toFixed(2)} s` : '—'],
            ['Per epoch', duration(e.estimated_seconds_per_epoch)],
          ]} />
          <p className="mt-2 text-2xs leading-snug text-ink-3">{e.note}</p>
        </>
      )}
    </div>
  );
}

/* ----------------------------------------------------------------- monitor */

function Monitor({ status, error }: { status?: TrainingStatus; error?: Error }) {
  const [busy, setBusy] = useState(false);
  const [ctrlError, setCtrlError] = useState<Error>();
  const s = status;
  const control = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setCtrlError(undefined);
    try { await fn(); } catch (e) { setCtrlError(e as Error); } finally { setBusy(false); }
  };
  const history = s?.history ?? [];
  const last = history[history.length - 1];

  if (!s) return error ? <ErrorBanner error={error} /> : <Loading />;
  if (!s.experiment_id) {
    return (
      <EmptyState icon={TrendingDown} title="No training runs yet">
        Configure an experiment on the left, estimate its workload, and start it. Loss curves, validation metrics and logs stream here live.
      </EmptyState>
    );
  }
  return (
    <div className="min-w-0 space-y-6">
      <Panel ticks title={<span className="flex items-center gap-2">{s.name} <StatusBadge state={s.state} /></span>} eyebrow={s.active ? 'live run' : 'most recent run'} icon={Timer}
        actions={s.active ? (
          <>
            {s.state === 'paused'
              ? <button className="btn-secondary text-xs" disabled={busy} onClick={() => control(api.trainingResume)}><Play className="h-3.5 w-3.5" /> Resume</button>
              : <button className="btn-secondary text-xs" disabled={busy || s.state === 'preparing'} onClick={() => control(api.trainingPause)}><Pause className="h-3.5 w-3.5" /> Pause</button>}
            <button className="btn-danger text-xs" disabled={busy} onClick={() => control(api.trainingStop)} data-testid="stop-training"><Square className="h-3.5 w-3.5" /> Stop</button>
          </>
        ) : <Link to={`/experiments?id=${s.experiment_id}`} className="btn-ghost text-xs">View experiment</Link>}>
        <div className="flex items-center gap-4">
          <ProgressBar value={s.progress ?? 0} className="h-2 flex-1" tone={s.state === 'failed' ? 'bad' : s.state === 'completed' ? 'good' : s.state === 'paused' || s.state === 'stopped' ? 'warn' : 'accent'} />
          <span className="num w-12 text-right text-sm">{Math.round((s.progress ?? 0) * 100)}%</span>
        </div>
        <div className="mt-2 text-xs text-ink-2" aria-live="polite">{s.message}</div>
        {s.error && <ErrorBanner error={s.error} className="mt-3" />}
        {ctrlError && <ErrorBanner error={ctrlError} className="mt-3" />}
        <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-3 2xl:grid-cols-6">
          <Stat label="Epoch" value={`${s.epoch ?? 0}/${s.epochs_total ?? '—'}`} sub={s.batch ? `batch ${s.batch}/${s.batches_per_epoch}` : undefined} />
          <Stat label="Elapsed" value={duration(s.elapsed_seconds)} sub={s.active ? `ETA ${duration(s.eta_seconds)}` : undefined} />
          <Stat label="Train loss" value={fixed(s.active ? s.train_loss_running ?? last?.train_loss : last?.train_loss, 4)} sub={s.active ? 'running mean, this epoch' : 'last epoch'} />
          <Stat label="Val loss" value={fixed(last?.val_loss, 4)} sub="last epoch" />
          <Stat label="Best val mIoU" value={ratio(s.best_val_miou, 2)} tone="accent" sub={s.best_epoch ? `epoch ${s.best_epoch}` : undefined} />
          <Stat label="Model" value={archLabel(s.arch)} sub={`${compact(s.parameters)} params · ${s.device ?? '—'}`} icon={Cpu} />
        </div>
        {s.arch === 'deeplabv3' && s.pretrained_loaded === false && (
          <Callout tone="warn" className="mt-3">Pretrained DeepLabV3 weights could not be loaded, so this run started from random initialisation. This is recorded with the experiment.</Callout>
        )}
      </Panel>

      {history.length === 0 ? (
        <Panel><p className="py-8 text-center text-sm text-ink-3">Curves appear after the first epoch completes{s.active ? ' (preprocessing is cached the first time a resolution is used).' : '.'}</p></Panel>
      ) : (
        <>
          <div className="grid gap-6 xl:grid-cols-2">
            <Panel title="Loss" eyebrow="cross-entropy + dice · per epoch"><LossChart history={history} /></Panel>
            <Panel title="Validation metrics" eyebrow="held-out validation split"><ValMetricChart history={history} /></Panel>
          </div>
          <Panel title="Validation IoU per class" eyebrow="watch big rock — it is rare"><PerClassIouHistory history={history} /></Panel>
        </>
      )}
      <LogConsole experimentId={s.experiment_id} live={s.active} />
    </div>
  );
}

function LogConsole({ experimentId, live }: { experimentId: string; live: boolean }) {
  const logs = useRemote(() => api.trainingLogs(experimentId, 400), [experimentId], live ? 2000 : null);
  const box = useRef<HTMLPreElement>(null);
  const [follow, setFollow] = useState(true);
  useEffect(() => {
    if (follow && box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [logs.data, follow]);
  return (
    <Panel title="Training log" icon={Terminal} padded={false}
      actions={<label className="flex items-center gap-1.5 text-2xs text-ink-3"><input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> follow</label>}>
      <pre ref={box} className="max-h-72 overflow-auto bg-sunken/70 p-4 font-mono text-[11.5px] leading-relaxed text-ink-2" data-testid="training-log">
        {(logs.data?.lines ?? []).length === 0 ? 'No log output yet.' : logs.data!.lines.map((l, i) => (
          <div key={i} className={cx(l.includes('[best]') && 'text-good', /error|failed|Traceback/i.test(l) && 'text-bad')}>{l}</div>
        ))}
      </pre>
    </Panel>
  );
}
