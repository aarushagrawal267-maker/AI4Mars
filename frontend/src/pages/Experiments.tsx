import { Download, FlaskConical, GitCompare, Pencil, Save, Trash2 } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { archColor, LossChart, MULTI, MultiCurve, PerClassIouHistory, ValMetricChart } from '../charts/Charts';
import { Callout, EmptyState, ErrorBanner, KV, Loading, PageHeader, Panel, Segmented, StatusBadge } from '../components/ui';
import { useApp } from '../context/AppContext';
import { useRemote } from '../hooks/usePolling';
import { api } from '../services/api';
import type { Experiment, ExperimentDetail } from '../types/api';
import { archLabel, compact, cx, duration, fixed, int, ratio, when } from '../utils/format';
import { IS_DEMO } from '../demo/env';

export default function Experiments() {
  const { connection } = useApp();
  const [params, setParams] = useSearchParams();
  const list = useRemote(() => api.experiments(), [connection], 8000);
  const [filter, setFilter] = useState<'all' | 'unet' | 'deeplabv3'>('all');
  const [compare, setCompare] = useState<string[]>([]);
  const exps = (list.data?.experiments ?? []).filter((e) => filter === 'all' || e.arch === filter);
  const selectedId = params.get('id') ?? exps.find((e) => !e.is_fixture)?.id ?? exps[0]?.id ?? null;

  return (
    <div className="space-y-6">
      <PageHeader eyebrow="08 · record" title="Experiments"
        description="Every training run is recorded in SQLite — including stopped, failed and interrupted runs, because failed experiments are part of the research record. Compare configurations and curves, keep notes, and download reports." />
      {list.error && <ErrorBanner error={list.error} onRetry={list.reload} />}
      {!list.data ? <Loading /> : list.data.experiments.length === 0 ? (
        <EmptyState icon={FlaskConical} title="No experiments yet" action={<Link to="/training" className="btn-primary">Start a training run</Link>}>
          Experiments appear here as soon as a training run starts.
        </EmptyState>
      ) : (
        <>
          <Panel title="Experiment history" icon={FlaskConical} padded={false}
            actions={<>
              <Segmented size="sm" value={filter} onChange={setFilter} options={[{ value: 'all', label: 'all' }, { value: 'unet', label: 'U-Net' }, { value: 'deeplabv3', label: 'DeepLabV3' }]} />
              {compare.length >= 2 && <span className="chip text-accent"><GitCompare className="h-3 w-3" /> comparing {compare.length}</span>}
            </>}>
            <div className="overflow-x-auto">
              <table className="table-clean">
                <thead><tr><th className="w-8" title="Compare">⇄</th><th>Name</th><th>Status</th><th className="text-right">Epochs</th><th className="text-right">Best val mIoU</th><th className="text-right">Train / val</th><th className="text-right">Res.</th><th className="text-right">Params</th><th className="text-right">Time</th><th className="text-right">Started</th></tr></thead>
                <tbody>
                  {exps.map((e) => (
                    <tr key={e.id} className={cx('cursor-pointer', selectedId === e.id && 'bg-accent/[0.06]')} onClick={() => setParams({ id: e.id })}>
                      <td onClick={(ev) => ev.stopPropagation()}>
                        <input type="checkbox" aria-label={`Compare ${e.name}`} checked={compare.includes(e.id)}
                          onChange={() => setCompare((c) => (c.includes(e.id) ? c.filter((x) => x !== e.id) : [...c, e.id]))} />
                      </td>
                      <td><div className="flex items-center gap-2"><span className="h-2 w-2 shrink-0 rounded-full" style={{ background: archColor(e.arch) }} /><span className="chip">{archLabel(e.arch)}</span><span className="max-w-[320px] truncate">{e.name}</span></div></td>
                      <td><StatusBadge state={e.status} /></td>
                      <td className="num text-right">{e.epochs_completed}/{e.config.epochs ?? '—'}</td>
                      <td className="num text-right font-medium text-accent">{ratio(e.best_val_miou, 2)}</td>
                      <td className="num text-right text-ink-2">{e.n_train ? `${int(e.n_train)} / ${int(e.n_val)}` : '—'}</td>
                      <td className="num text-right text-ink-2">{e.config.resolution ? `${e.config.resolution}` : '—'}</td>
                      <td className="num text-right text-ink-2">{compact(e.parameters)}</td>
                      <td className="num text-right text-ink-2">{duration(e.elapsed_seconds)}</td>
                      <td className="num text-right text-ink-3">{when(e.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
          {compare.length >= 2 && <CompareExperiments ids={compare} all={list.data.experiments} />}
          {selectedId && <ExperimentView id={selectedId} onChanged={list.reload} onDeleted={() => { setParams({}); void list.reload(); }} />}
        </>
      )}
    </div>
  );
}

const CONFIG_ROWS: [string, (e: Experiment) => string][] = [
  ['Architecture', (e) => archLabel(e.arch)],
  ['Model options', (e) => Object.entries(e.config.model_config ?? {}).map(([k, v]) => `${k}=${v}`).join(', ') || '—'],
  ['Epochs', (e) => String(e.config.epochs)],
  ['Batch size', (e) => String(e.config.batch_size)],
  ['Learning rate', (e) => String(e.config.learning_rate)],
  ['Weight decay', (e) => String(e.config.weight_decay)],
  ['Resolution', (e) => `${e.config.resolution}px`],
  ['Train subset', (e) => (e.config.max_train_samples ? int(e.config.max_train_samples) : 'full split')],
  ['Val subset', (e) => (e.config.max_val_samples ? int(e.config.max_val_samples) : 'full split')],
  ['Loss', (e) => `CE (ignore 255)${e.config.dice_weight ? ` + ${e.config.dice_weight}×Dice` : ''}`],
  ['Class weighting', (e) => e.config.class_weighting],
  ['Scheduler', (e) => e.config.scheduler],
  ['Early stopping', (e) => (e.config.early_stopping_patience ? `${e.config.early_stopping_patience} epochs` : 'off')],
  ['Augmentation', (e) => (e.config.augment ? 'on' : 'off')],
  ['Seed', (e) => String(e.config.seed)],
  ['Device', (e) => e.device ?? e.config.device],
  ['Pretrained init', (e) => (e.arch === 'deeplabv3' ? (e.pretrained_loaded ? 'loaded (COCO)' : e.pretrained_loaded === false ? 'not loaded' : '—') : 'n/a')],
  ['Dataset fingerprint', (e) => e.dataset_fingerprint ?? '—'],
];

function ExperimentView({ id, onChanged, onDeleted }: { id: string; onChanged: () => void; onDeleted: () => void }) {
  const exp = useRemote(() => api.experiment(id), [id], 6000);
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState('');
  const [notes, setNotes] = useState('');
  const [dirty, setDirty] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [error, setError] = useState<Error>();
  const e = exp.data;
  useEffect(() => {
    if (e && !dirty) { setName(e.name); setNotes(e.notes); }
  }, [e, dirty]);
  useEffect(() => { setDirty(false); setEditing(false); setConfirmDelete(false); }, [id]);

  if (exp.error) return <ErrorBanner error={exp.error} />;
  if (!e) return <Loading />;
  async function save() {
    try {
      exp.setData(await api.updateExperiment(id, { name, notes }));
      setDirty(false);
      setEditing(false);
      onChanged();
    } catch (err) { setError(err as Error); }
  }
  async function remove() {
    try { await api.deleteExperiment(id); onDeleted(); } catch (err) { setError(err as Error); }
  }
  return (
    <div className="space-y-6">
      <Panel ticks icon={FlaskConical} eyebrow={e.id}
        title={editing ? <input className="field py-1 text-sm" value={name} onChange={(ev) => { setName(ev.target.value); setDirty(true); }} aria-label="Experiment name" /> : <span className="flex items-center gap-2">{e.name} <StatusBadge state={e.status} /></span>}
        actions={IS_DEMO ? undefined : <>
          {editing ? <button className="btn-primary text-xs" onClick={save}><Save className="h-3.5 w-3.5" /> Save</button>
            : <button className="btn-ghost text-xs" onClick={() => setEditing(true)}><Pencil className="h-3.5 w-3.5" /> Rename</button>}
          <a className="btn-ghost text-xs" href={api.experimentReportUrl(e.id, 'json')} download><Download className="h-3.5 w-3.5" /> JSON</a>
          <a className="btn-ghost text-xs" href={api.experimentReportUrl(e.id, 'csv')} download><Download className="h-3.5 w-3.5" /> CSV</a>
          {!e.is_fixture && (confirmDelete
            ? <button className="btn-danger text-xs" onClick={remove}>Confirm delete</button>
            : <button className="btn-ghost p-1.5" onClick={() => setConfirmDelete(true)} aria-label="Delete experiment"><Trash2 className="h-3.5 w-3.5" /></button>)}
        </>}>
        {e.message && <p className="mb-3 text-sm text-ink-2">{e.message}</p>}
        {e.error && <Callout tone="bad" className="mb-3" title="Run error">{e.error}</Callout>}
        {e.is_fixture && <Callout tone="info" className="mb-3">This is the untrained test fixture used by automated UI tests. It is not a trained model.</Callout>}
        {error && <ErrorBanner error={error} className="mb-3" />}
        <div className="grid gap-6 lg:grid-cols-[1fr_1fr_1.2fr]">
          <KV items={CONFIG_ROWS.slice(0, 9).map(([k, f]) => [k, f(e)])} />
          <KV items={CONFIG_ROWS.slice(9).map(([k, f]) => [k, f(e)])} />
          <div className="space-y-3">
            <KV items={[
              ['Epochs completed', `${e.epochs_completed} / ${e.config.epochs}`],
              ['Best val mIoU', `${ratio(e.best_val_miou, 2)}${e.best_epoch ? ` (epoch ${e.best_epoch})` : ''}`],
              ['Final train / val loss', `${fixed(e.final_train_loss, 4)} / ${fixed(e.final_val_loss, 4)}`],
              ['Training / validation pairs', e.n_train ? `${int(e.n_train)} / ${int(e.n_val)}` : '—'],
              ['Parameters', compact(e.parameters)],
              ['Wall time', duration(e.elapsed_seconds)],
              ['Checkpoint', e.has_checkpoint ? 'best.pt + last.pt saved' : 'none'],
            ]} />
            <div>
              <label className="mb-1 block text-xs font-medium text-ink-2" htmlFor="notes">Research notes</label>
              <textarea id="notes" className="field min-h-[84px] text-xs" value={notes} placeholder="Observations, anomalies, follow-ups…"
                onChange={(ev) => { setNotes(ev.target.value); setDirty(true); }} />
              {dirty && !editing && <button className="btn-secondary mt-2 text-xs" onClick={save}><Save className="h-3.5 w-3.5" /> Save notes</button>}
            </div>
          </div>
        </div>
      </Panel>
      {e.history.length > 0 && (
        <>
          <div className="grid gap-6 xl:grid-cols-2">
            <Panel title="Loss curves" eyebrow="training vs validation"><LossChart history={e.history} /></Panel>
            <Panel title="Validation metrics"><ValMetricChart history={e.history} /></Panel>
          </div>
          <Panel title="Validation IoU per class"><PerClassIouHistory history={e.history} /></Panel>
        </>
      )}
      <EvaluationsOf exp={e} />
    </div>
  );
}

function EvaluationsOf({ exp }: { exp: ExperimentDetail }) {
  if (!exp.evaluations.length) {
    return (
      <Panel title="Evaluations">
        <p className="text-sm text-ink-3">Not evaluated yet. {exp.has_checkpoint && <Link className="text-accent hover:underline" to="/evaluation">Evaluate this model on the held-out test set →</Link>}</p>
      </Panel>
    );
  }
  return (
    <Panel title="Evaluations of this model" padded={false}
      actions={<a className="btn-ghost text-xs" href={api.evaluationReportUrl(exp.evaluations.map((v) => v.id), 'csv')} download><Download className="h-3.5 w-3.5" /> All as CSV</a>}>
      <table className="table-clean">
        <thead><tr><th>Split</th><th className="text-right">Images</th><th className="text-right">Pixel acc.</th><th className="text-right">mIoU</th><th className="text-right">Dice</th><th className="text-right">Soil</th><th className="text-right">Bedrock</th><th className="text-right">Sand</th><th className="text-right">Big rock</th><th className="text-right">When</th></tr></thead>
        <tbody>
          {exp.evaluations.map((v) => (
            <tr key={v.id}>
              <td className="num">{v.split}{v.split === 'test' ? ` · ${v.label_variant}` : ''}</td>
              <td className="num text-right">{v.images}</td>
              <td className="num text-right">{ratio(v.metrics.pixel_accuracy, 1)}</td>
              <td className="num text-right font-medium text-accent">{ratio(v.metrics.mean_iou, 1)}</td>
              <td className="num text-right">{ratio(v.metrics.mean_dice, 1)}</td>
              {(['soil', 'bedrock', 'sand', 'big_rock'] as const).map((k) => <td key={k} className="num text-right text-ink-2">{ratio(v.metrics.per_class[k].iou, 1)}</td>)}
              <td className="num text-right text-ink-3">{when(v.created_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="px-4 py-3 text-2xs text-ink-3">Validation-split numbers guided checkpoint selection; only test-split numbers are unbiased estimates of held-out performance.</p>
    </Panel>
  );
}

function CompareExperiments({ ids, all }: { ids: string[]; all: Experiment[] }) {
  const [details, setDetails] = useState<Record<string, ExperimentDetail>>({});
  useEffect(() => {
    ids.forEach((id) => api.experiment(id).then((d) => setDetails((s) => ({ ...s, [id]: d }))).catch(() => undefined));
  }, [ids]);
  const exps = ids.map((id) => all.find((e) => e.id === id)).filter(Boolean) as Experiment[];
  const series = useMemo(() => exps.map((e, i) => ({ id: e.id, name: e.name, color: MULTI[i % MULTI.length], history: details[e.id]?.history ?? [] })), [exps, details]);
  const differing = CONFIG_ROWS.filter(([, f]) => new Set(exps.map(f)).size > 1);
  return (
    <div className="space-y-6">
      <div className="grid gap-6 xl:grid-cols-2">
        <Panel title="Validation mIoU" icon={GitCompare}><MultiCurve series={series} metric="val_mean_iou" percent /></Panel>
        <Panel title="Validation loss" icon={GitCompare}><MultiCurve series={series} metric="val_loss" /></Panel>
      </div>
      <Panel title="Configuration differences" padded={false}>
        <table className="table-clean">
          <thead><tr><th>Setting</th>{exps.map((e, i) => <th key={e.id}><span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: MULTI[i % MULTI.length] }} /><span className="max-w-[200px] truncate normal-case">{e.name}</span></span></th>)}</tr></thead>
          <tbody>
            {(differing.length ? differing : CONFIG_ROWS.slice(0, 1)).map(([k, f]) => (
              <tr key={k}><td className="text-ink-2">{k}</td>{exps.map((e) => <td key={e.id} className="num">{f(e)}</td>)}</tr>
            ))}
            <tr><td className="text-ink-2">Best val mIoU</td>{exps.map((e) => <td key={e.id} className="num text-accent">{ratio(e.best_val_miou, 2)}</td>)}</tr>
          </tbody>
        </table>
        <p className="px-4 py-3 text-2xs text-ink-3">{differing.length ? `${differing.length} setting(s) differ; everything else is identical.` : 'All recorded settings are identical.'} For a fair architecture comparison only the architecture should differ.</p>
      </Panel>
    </div>
  );
}
