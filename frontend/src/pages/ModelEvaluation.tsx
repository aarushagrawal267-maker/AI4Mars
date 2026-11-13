import { BarChart3, Download, Gauge, GitCompare, Play, Scale, Trash2 } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { archColor, ConfusionMatrix, GroupedClassBars, MULTI, MultiCurve, SERIES } from '../charts/Charts';
import { ModelSelect, NoModels } from '../components/shared';
import {
  Callout, ClassSwatch, EmptyState, ErrorBanner, Field, Loading, PageHeader, Panel, ProgressBar, Segmented, Stat, StatusBadge,
} from '../components/ui';
import { useApp } from '../context/AppContext';
import { useRemote } from '../hooks/usePolling';
import { api } from '../services/api';
import type { Evaluation, ExperimentDetail, Job } from '../types/api';
import { archLabel, CLASS_KEYS, CLASS_LABEL, compact, cx, int, ms, ratio, when } from '../utils/format';

export default function ModelEvaluation() {
  const { connection } = useApp();
  const models = useRemote(() => api.models(), [connection]);
  const evals = useRemote(() => api.evaluations(), [connection]);
  const ds = useRemote(() => api.datasetStatus(), [connection]);
  const [selected, setSelected] = useState<number[]>([]);
  const [focus, setFocus] = useState<number | null>(null);

  const list = evals.data?.evaluations ?? [];
  useEffect(() => {
    if (focus === null && list.length) setFocus(list[0].id);
  }, [list, focus]);
  const focused = list.find((e) => e.id === focus) ?? null;
  const compared = list.filter((e) => selected.includes(e.id));

  const toggle = (id: number) => setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));

  return (
    <div className="space-y-6">
      <PageHeader eyebrow="05 · measurement" title="Model Evaluation"
        description="Score trained models on genuinely held-out data: the official expert-labelled test set (or the validation split). Metrics are computed at native label resolution from a confusion matrix over labelled pixels only." />
      {models.error && <ErrorBanner error={models.error} onRetry={models.reload} />}
      {!models.data ? <Loading /> : models.data.models.length === 0 ? <NoModels /> : (
        <>
          <RunPanel models={models.data.models} indexed={!!ds.data?.indexed} onDone={async (id) => { await evals.reload(); setFocus(id); }} />
          <Panel title="Stored evaluations" icon={BarChart3} padded={false}
            actions={selected.length > 0 && (
              <>
                <span className="text-2xs text-ink-3">{selected.length} selected</span>
                <a className="btn-ghost text-xs" href={api.evaluationReportUrl(selected, 'csv')} download><Download className="h-3.5 w-3.5" /> CSV</a>
                <a className="btn-ghost text-xs" href={api.evaluationReportUrl(selected, 'json')} download><Download className="h-3.5 w-3.5" /> JSON</a>
              </>
            )}>
            {list.length === 0 ? (
              <div className="p-4"><EmptyState icon={Gauge} title="No evaluations yet">Run an evaluation above. Results are stored in the local SQLite database and can be compared and downloaded.</EmptyState></div>
            ) : (
              <div className="overflow-x-auto">
                <table className="table-clean">
                  <thead><tr><th className="w-8" title="Select for comparison / download">⇄</th><th>Model</th><th>Split</th><th className="text-right">Images</th><th className="text-right">Pixel acc.</th><th className="text-right">mIoU</th><th className="text-right">Dice</th><th className="text-right">Forward</th><th className="text-right">When</th><th /></tr></thead>
                  <tbody>
                    {list.map((e) => (
                      <tr key={e.id} className={cx('cursor-pointer', focus === e.id && 'bg-accent/[0.06]')} onClick={() => setFocus(e.id)}>
                        <td onClick={(ev) => ev.stopPropagation()}><input type="checkbox" aria-label={`Select evaluation ${e.id}`} checked={selected.includes(e.id)} onChange={() => toggle(e.id)} /></td>
                        <td><div className="flex items-center gap-2"><span className="h-2 w-2 rounded-full" style={{ background: archColor(e.arch) }} /><span className="chip">{archLabel(e.arch)}</span><span className="max-w-[280px] truncate">{e.experiment_name}</span>{e.is_fixture && <span className="chip text-info">fixture</span>}</div></td>
                        <td className="num text-ink-2">{e.split}{e.split === 'test' ? ` · ${e.label_variant}` : ''}{e.metrics.subset_of_split ? ' (subset)' : ''}</td>
                        <td className="num text-right">{e.images}</td>
                        <td className="num text-right">{ratio(e.metrics.pixel_accuracy, 1)}</td>
                        <td className="num text-right font-medium text-accent">{ratio(e.metrics.mean_iou, 1)}</td>
                        <td className="num text-right">{ratio(e.metrics.mean_dice, 1)}</td>
                        <td className="num text-right text-ink-2">{ms(e.metrics.inference.forward_ms_median)}</td>
                        <td className="num text-right text-ink-3">{when(e.created_at)}</td>
                        <td onClick={(ev) => ev.stopPropagation()}>
                          <button className="btn-ghost p-1" aria-label="Delete evaluation" onClick={async () => { await api.deleteEvaluation(e.id); setSelected((s) => s.filter((x) => x !== e.id)); if (focus === e.id) setFocus(null); void evals.reload(); }}>
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Panel>

          {compared.length >= 2 && <Comparison evals={compared} />}
          {compared.length === 1 && <p className="text-center text-xs text-ink-3">Select one more evaluation to compare architectures side by side.</p>}
          {focused && <EvaluationDetail ev={focused} />}
        </>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------- runner */

function RunPanel({ models, indexed, onDone }: { models: Parameters<typeof ModelSelect>[0]['models']; indexed: boolean; onDone: (id: number) => void }) {
  const real = models.filter((m) => !m.is_fixture);
  const [modelId, setModelId] = useState((real[0] ?? models[0]).id);
  const [split, setSplit] = useState<'test' | 'val'>('test');
  const [variant, setVariant] = useState('min3');
  const [limit, setLimit] = useState<string>('');
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<Error>();
  const running = job?.status === 'running' || job?.status === 'queued';

  useEffect(() => {
    api.evaluateStatus().then((r) => r.job && (r.job.status === 'running' || r.job.status === 'queued') && setJob(r.job)).catch(() => undefined);
  }, []);
  useEffect(() => {
    if (!running) return;
    const t = window.setInterval(async () => {
      const r = await api.evaluateStatus().catch(() => null);
      if (!r?.job) return;
      setJob(r.job);
      if (r.job.status === 'completed') onDone((r.job.result as { evaluation_id: number }).evaluation_id);
    }, 1000);
    return () => window.clearInterval(t);
  }, [running, onDone]);

  async function run() {
    setError(undefined);
    try {
      setJob(await api.evaluate({ model_id: modelId, split, label_variant: split === 'test' ? variant : null, max_images: limit ? Number(limit) : null }));
    } catch (e) {
      setError(e as Error);
    }
  }
  return (
    <Panel title="Run an evaluation" icon={Play} ticks>
      {!indexed ? <Callout tone="warn">Load the dataset first: evaluation needs the held-out labelled images.</Callout> : (
        <div className="grid items-end gap-3 md:grid-cols-[2fr_auto_auto_140px_auto]">
          <Field label="Model"><ModelSelect models={models} value={modelId} onChange={setModelId} /></Field>
          <Field label="Held-out split"><Segmented value={split} onChange={setSplit} options={[{ value: 'test', label: 'Test (expert)' }, { value: 'val', label: 'Validation' }]} /></Field>
          <Field label="Label agreement">
            <Segmented value={variant} onChange={setVariant} options={['min1', 'min2', 'min3'].map((v) => ({ value: v, label: v, disabled: split !== 'test' }))} />
          </Field>
          <Field label="Max images" hint="blank = whole split"><input className="field num" type="number" min={1} placeholder="all" value={limit} onChange={(e) => setLimit(e.target.value)} /></Field>
          <div className="flex gap-2">
            <button className="btn-primary" disabled={running || !modelId} onClick={run}><Play className="h-4 w-4" /> Evaluate</button>
            {running && <button className="btn-danger" onClick={() => api.evaluateCancel()}>Cancel</button>}
          </div>
        </div>
      )}
      {job && (
        <div className="mt-4 rounded-lg border border-line bg-sunken/60 p-3" aria-live="polite">
          <div className="mb-2 flex items-center justify-between text-xs"><StatusBadge state={job.status} /><span className="num text-ink-2">{Math.round(job.progress * 100)}%</span></div>
          <ProgressBar value={job.progress} tone={job.status === 'failed' ? 'bad' : job.status === 'completed' ? 'good' : 'accent'} />
          <div className="mt-1.5 text-2xs text-ink-3">{job.error ?? job.message}</div>
        </div>
      )}
      {error && <ErrorBanner error={error} className="mt-3" />}
      <p className="mt-3 text-2xs text-ink-3">Training data is never used for evaluation. min3 (all three experts agree) is the strictest and sparsest test label set; min1 covers more pixels.</p>
    </Panel>
  );
}

/* ------------------------------------------------------------------- detail */

function EvaluationDetail({ ev }: { ev: Evaluation }) {
  const m = ev.metrics;
  const [norm, setNorm] = useState(true);
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-lg font-semibold">{ev.experiment_name}</h2>
        <span className="chip">{archLabel(ev.arch)}</span>
        <span className="chip">{ev.split}{ev.split === 'test' ? ` · ${ev.label_variant}` : ''} · {ev.images} images</span>
        {ev.is_fixture && <span className="chip text-info">TEST FIXTURE · untrained</span>}
        <div className="ml-auto flex gap-1">
          <a className="btn-secondary text-xs" href={api.evaluationReportUrl([ev.id], 'csv')} download><Download className="h-3.5 w-3.5" /> CSV</a>
          <a className="btn-secondary text-xs" href={api.evaluationReportUrl([ev.id], 'json')} download><Download className="h-3.5 w-3.5" /> JSON</a>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-8">
        <Stat label="Pixel accuracy" value={ratio(m.pixel_accuracy, 1)} />
        <Stat label="Mean IoU" value={ratio(m.mean_iou, 1)} tone="accent" />
        <Stat label="Mean Dice" value={ratio(m.mean_dice, 1)} />
        <Stat label="Mean precision" value={ratio(m.mean_precision, 1)} />
        <Stat label="Mean recall" value={ratio(m.mean_recall, 1)} />
        <Stat label="Forward pass" value={ms(m.inference.forward_ms_median)} sub={`median · ${m.inference.device} · ${m.resolution}px`} />
        <Stat label="Parameters" value={compact(m.model.parameters)} />
        <Stat label="Checkpoint" value={`${m.model.checkpoint_mb.toFixed(1)} MB`} sub={m.model.checkpoint_epoch ? `epoch ${m.model.checkpoint_epoch}` : undefined} />
      </div>
      <div className="grid gap-6 xl:grid-cols-[1.1fr_1fr]">
        <Panel title="Per-class performance" icon={Scale} padded={false}>
          <table className="table-clean">
            <thead><tr><th>Class</th><th className="text-right">IoU</th><th className="text-right">Dice</th><th className="text-right">Precision</th><th className="text-right">Recall</th><th className="text-right">GT pixels</th></tr></thead>
            <tbody>
              {CLASS_KEYS.map((k) => {
                const c = m.per_class[k];
                return (
                  <tr key={k}>
                    <td><span className="flex items-center gap-2"><ClassSwatch k={k} /> {CLASS_LABEL[k]}</span></td>
                    <td className="num text-right font-medium">{c.iou === null ? 'n/a' : ratio(c.iou, 1)}</td>
                    <td className="num text-right">{c.dice === null ? 'n/a' : ratio(c.dice, 1)}</td>
                    <td className="num text-right">{c.precision === null ? 'n/a' : ratio(c.precision, 1)}</td>
                    <td className="num text-right">{c.recall === null ? 'n/a' : ratio(c.recall, 1)}</td>
                    <td className="num text-right text-ink-3">{int(c.support_pixels)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="p-4">
            <GroupedClassBars height={220} groups={[
              { id: 'precision', name: 'Precision', color: SERIES.train, values: Object.fromEntries(CLASS_KEYS.map((k) => [k, m.per_class[k].precision])) as never },
              { id: 'recall', name: 'Recall', color: SERIES.val, values: Object.fromEntries(CLASS_KEYS.map((k) => [k, m.per_class[k].recall])) as never },
            ]} />
            <p className="mt-2 text-2xs text-ink-3">n/a = undefined (the class was never predicted or never present), not zero. Mean metrics average over classes present in the ground truth ({m.classes_present.map((k) => CLASS_LABEL[k]).join(', ')}). {int(m.evaluated_pixels)} labelled pixels evaluated.</p>
          </div>
        </Panel>
        <Panel title="Confusion matrix" icon={GitCompare} actions={<Segmented size="sm" value={norm ? 'row' : 'count'} onChange={(v) => setNorm(v === 'row')} options={[{ value: 'row', label: 'row %' }, { value: 'count', label: 'pixels' }]} />}>
          <ConfusionMatrix matrix={m.confusion_matrix} normalize={norm} />
          <p className="mt-3 text-2xs text-ink-3">{m.inference.note} Timed on {m.inference.timed_images} images; p95 {ms(m.inference.forward_ms_p95)}, end-to-end mean {ms(m.inference.end_to_end_ms_mean)}.</p>
        </Panel>
      </div>
    </div>
  );
}

/* --------------------------------------------------------------- comparison */

function Comparison({ evals }: { evals: Evaluation[] }) {
  const comparable = new Set(evals.map((e) => `${e.split}|${e.label_variant}|${e.images}`)).size === 1;
  const color = (e: Evaluation, i: number) => {
    const sameArch = evals.filter((x) => x.arch === e.arch).length > 1;
    return sameArch ? MULTI[i % MULTI.length] : archColor(e.arch);
  };
  const short = (e: Evaluation) => `${archLabel(e.arch)} #${e.id}`;
  const [details, setDetails] = useState<Record<string, ExperimentDetail>>({});
  const expIds = useMemo(() => [...new Set(evals.map((e) => e.experiment_id))], [evals]);
  useEffect(() => {
    expIds.forEach((id) => !details[id] && api.experiment(id).then((d) => setDetails((s) => ({ ...s, [id]: d }))).catch(() => undefined));
  }, [expIds, details]);

  const summary = (['mean_iou', 'pixel_accuracy', 'mean_dice'] as const).map((k) => ({
    metric: { mean_iou: 'mIoU', pixel_accuracy: 'Pixel acc.', mean_dice: 'Mean Dice' }[k],
    ...Object.fromEntries(evals.map((e) => [String(e.id), e.metrics[k]])),
  }));
  const curves = evals.filter((e, i, a) => a.findIndex((x) => x.experiment_id === e.experiment_id) === i)
    .map((e) => ({ id: e.experiment_id, name: e.experiment_name ?? e.experiment_id, color: color(e, evals.indexOf(e)), history: details[e.experiment_id]?.history ?? [] }));

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2"><GitCompare className="h-4 w-4 text-accent" /><h2 className="text-lg font-semibold">Architecture comparison</h2></div>
      {!comparable && <Callout tone="warn" title="Not directly comparable">The selected evaluations used different splits, label variants or image counts. Compare models on the same test partition for a fair result.</Callout>}
      <Panel padded={false} title="Side by side" icon={Scale}>
        <div className="overflow-x-auto">
          <table className="table-clean">
            <thead><tr><th>Metric</th>{evals.map((e, i) => <th key={e.id} className="text-right"><span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: color(e, i) }} />{short(e)}</span></th>)}</tr></thead>
            <tbody>
              {([
                ['Experiment', (e: Evaluation) => <span className="block max-w-[200px] truncate">{e.experiment_name}</span>],
                ['Evaluated on', (e: Evaluation) => `${e.split}${e.split === 'test' ? ` · ${e.label_variant}` : ''} · ${e.images} imgs`],
                ['Mean IoU', (e: Evaluation) => <b className="text-accent">{ratio(e.metrics.mean_iou, 2)}</b>],
                ['Pixel accuracy', (e: Evaluation) => ratio(e.metrics.pixel_accuracy, 2)],
                ['Mean Dice', (e: Evaluation) => ratio(e.metrics.mean_dice, 2)],
                ...CLASS_KEYS.map((k) => [`IoU · ${CLASS_LABEL[k]}`, (e: Evaluation) => ratio(e.metrics.per_class[k].iou, 1)] as const),
                ['Parameters', (e: Evaluation) => compact(e.metrics.model.parameters)],
                ['Checkpoint size', (e: Evaluation) => `${e.metrics.model.checkpoint_mb.toFixed(1)} MB`],
                ['Input resolution', (e: Evaluation) => `${e.metrics.resolution}px`],
                ['Forward pass (median)', (e: Evaluation) => `${ms(e.metrics.inference.forward_ms_median)} · ${e.metrics.inference.device}`],
                ['Images / second', (e: Evaluation) => e.metrics.inference.images_per_second?.toFixed(1) ?? '—'],
                ['Training samples', (e: Evaluation) => int(details[e.experiment_id]?.n_train)],
                ['Epochs (best)', (e: Evaluation) => details[e.experiment_id] ? `${details[e.experiment_id].epochs_completed} (${details[e.experiment_id].best_epoch ?? '—'})` : '…'],
                ['Pretrained init', (e: Evaluation) => (e.arch === 'deeplabv3' ? (details[e.experiment_id]?.pretrained_loaded ? 'COCO' : 'no') : 'from scratch')],
              ] as [string, (e: Evaluation) => React.ReactNode][]).map(([label, fn]) => (
                <tr key={label}><td className="text-ink-2">{label}</td>{evals.map((e) => <td key={e.id} className="num text-right">{fn(e)}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      <div className="grid gap-6 xl:grid-cols-2">
        <Panel title="Headline metrics">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={summary} margin={{ top: 18, right: 8, left: 0, bottom: 4 }} barGap={2}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="metric" tickLine={false} axisLine={false} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tickLine={false} axisLine={false} width={44} />
              <Tooltip formatter={(v: number) => ratio(v, 2)} contentStyle={{ background: 'rgb(var(--surface))', border: '1px solid rgb(var(--line-strong))', borderRadius: 8, fontSize: 12 }} />
              {evals.map((e, i) => (
                <Bar key={e.id} dataKey={String(e.id)} name={short(e)} fill={color(e, i)} radius={[4, 4, 0, 0]} isAnimationActive={false}>
                  {evals.length <= 3 && <LabelList dataKey={String(e.id)} position="top" formatter={(v: number) => `${(v * 100).toFixed(1)}`} style={{ fill: 'rgb(var(--ink-3))', fontSize: 10, fontFamily: 'JetBrains Mono' }} />}
                </Bar>
              ))}
            </BarChart>
          </ResponsiveContainer>
          <div className="mt-1 flex flex-wrap gap-3 text-2xs text-ink-2">{evals.map((e, i) => <span key={e.id} className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm" style={{ background: color(e, i) }} />{short(e)} — {e.experiment_name}</span>)}</div>
        </Panel>
        <Panel title="Per-class IoU">
          <GroupedClassBars groups={evals.map((e, i) => ({ id: String(e.id), name: short(e), color: color(e, i),
            values: Object.fromEntries(CLASS_KEYS.map((k) => [k, e.metrics.per_class[k].iou])) as never }))} />
        </Panel>
        <Panel title="Inference speed" eyebrow="median forward pass, lower is faster">
          <ResponsiveContainer width="100%" height={60 + evals.length * 34}>
            <BarChart layout="vertical" data={evals.map((e, i) => ({ name: short(e), v: e.metrics.inference.forward_ms_median, c: color(e, i) }))} margin={{ left: 8, right: 64 }}>
              <XAxis type="number" hide />
              <YAxis type="category" dataKey="name" width={110} tickLine={false} axisLine={false} />
              <Bar dataKey="v" radius={[0, 4, 4, 0]} isAnimationActive={false}>
                {evals.map((e, i) => <Cell key={e.id} fill={color(e, i)} />)}
                <LabelList dataKey="v" position="right" formatter={(v: number) => ms(v)} style={{ fill: 'rgb(var(--ink-2))', fontSize: 11, fontFamily: 'JetBrains Mono' }} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="text-2xs text-ink-3">Measured during evaluation on the device shown in the table; not a promise of real-time performance elsewhere.</p>
        </Panel>
        <Panel title="Validation mIoU during training">
          <MultiCurve series={curves} metric="val_mean_iou" percent height={220} />
        </Panel>
      </div>
      <Panel title="Training loss curves" eyebrow="train loss (one axis, same loss definition only if configs match)">
        <MultiCurve series={curves} metric="train_loss" height={220} />
      </Panel>
    </div>
  );
}
