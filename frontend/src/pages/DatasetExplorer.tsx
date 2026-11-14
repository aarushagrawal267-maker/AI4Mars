import {
  AlertTriangle, CheckCircle2, ChevronLeft, ChevronRight, CloudDownload, Database, ExternalLink, FileArchive, FolderTree,
  Images, Layers, RefreshCw, ScanSearch, ShieldCheck, XCircle,
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { ClassBars } from '../charts/Charts';
import {
  Callout, ClassLegend, CompositionBar, EmptyState, ErrorBanner, Field, KV, Loading, Modal, PageHeader, Panel, ProgressBar,
  Segmented, Stat, StatusBadge,
} from '../components/ui';
import { useApp } from '../context/AppContext';
import { useRemote } from '../hooks/usePolling';
import { api } from '../services/api';
import type { Analysis, ClassKey, DatasetStatistics, Sample, SplitStats, ValidationReport } from '../types/api';
import { bytes, CLASS_KEYS, CLASS_LABEL, cx, int, pct } from '../utils/format';
import { IS_DEMO } from '../demo/env';

export default function DatasetExplorer() {
  const { connection } = useApp();
  const status = useRemote(() => api.datasetStatus(), [connection]);
  const job = status.data?.job;
  const jobRunning = job?.status === 'running' || job?.status === 'queued';
  // poll quickly while a dataset job runs
  useEffect(() => {
    if (!jobRunning) return;
    const t = window.setInterval(() => void status.reload(), 1200);
    return () => window.clearInterval(t);
  }, [jobRunning, status]);

  const indexed = !!status.data?.indexed;
  const key = status.data?.fingerprint ?? 'none';
  const stats = useRemote(() => (indexed ? api.datasetStatistics() : Promise.resolve(null)), [key, indexed]);
  const validation = useRemote(() => (indexed ? api.datasetValidation() : Promise.resolve(null)), [key, indexed]);
  const tree = useRemote(() => api.datasetStructure(), [key, status.data?.dataset_found]);

  return (
    <div className="space-y-6">
      <PageHeader eyebrow="03 · data" title="Dataset Explorer"
        description="NASA JPL's AI4Mars dataset (Curiosity NavCam subset). Every count, percentage and validation result on this page is computed from the files on disk; nothing is estimated." />
      {status.error && <ErrorBanner error={status.error} onRetry={status.reload} />}
      {!status.data ? (status.error ? null : <Loading />) : (
        <>
          {IS_DEMO ? (
            <Callout tone="info" title="Dataset in the browser edition">Statistics, the validation report and the folder structure describe the full local dataset (AI4Mars v0.6, Curiosity NavCam). The gallery includes 120 of the 322 expert-labelled test images (min1 labels), including every one that contains big rocks.</Callout>
          ) : (
            <Acquisition onChanged={status.reload} />
          )}
          {!indexed ? (
            status.data.dataset_found ? (
              <EmptyState icon={Database} title="Dataset found but not validated yet" action={<RescanButton onDone={status.reload} />}>
                AI4Mars files were found in <code className="num">{status.data.raw_dir}</code>. Validate and index them to explore statistics and samples.
              </EmptyState>
            ) : null
          ) : stats.data && validation.data ? (
            <>
              <StatsSection stats={stats.data} />
              <div className="grid gap-6 xl:grid-cols-2">
                <ValidationSection report={validation.data} />
                <TestVariants stats={stats.data} />
              </div>
              <Gallery testVariants={stats.data.statistics.test_variants} />
              <Panel title="Dataset structure" icon={FolderTree} padded={false}>
                <div className="max-h-[360px] overflow-auto">
                  <table className="table-clean">
                    <thead><tr><th>Folder (relative to {tree.data?.root ? 'data/raw' : 'dataset root'})</th><th className="text-right">Files</th><th>Types</th><th className="text-right">Size</th></tr></thead>
                    <tbody>
                      {tree.data?.tree.filter((n) => n.files > 0 || n.depth < 4).map((n) => (
                        <tr key={n.path}>
                          <td className="font-mono text-xs" style={{ paddingLeft: 12 + n.depth * 16 }}>
                            <span className="text-ink-3">{n.depth > 0 ? '└ ' : ''}</span>{n.path.split('/').pop()}
                          </td>
                          <td className="num text-right">{n.files ? int(n.files) : ''}</td>
                          <td className="text-2xs text-ink-3">{Object.entries(n.extensions).map(([e, c]) => `${e} ${int(c)}`).join(' · ')}</td>
                          <td className="num text-right text-ink-2">{n.bytes ? bytes(n.bytes) : ''}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Panel>
            </>
          ) : <Loading label="Loading dataset statistics…" />}
        </>
      )}
    </div>
  );
}

function RescanButton({ onDone }: { onDone: () => void }) {
  const [err, setErr] = useState<Error>();
  return (
    <div className="flex flex-col items-center gap-2">
      <button className="btn-primary" onClick={() => api.datasetLoad().then(onDone).catch(setErr)}><ShieldCheck className="h-4 w-4" /> Validate & index</button>
      {err && <ErrorBanner error={err} />}
    </div>
  );
}

/* ------------------------------------------------------------ acquisition */

function Acquisition({ onChanged }: { onChanged: () => void }) {
  const { connection } = useApp();
  const status = useRemote(() => api.datasetStatus(), [connection], 3000);
  const s = status.data;
  const [mode, setMode] = useState<'subset' | 'full' | 'import'>('subset');
  const [count, setCount] = useState(3000);
  const [all, setAll] = useState(false);
  const [confirmFull, setConfirmFull] = useState(false);
  const [archive, setArchive] = useState('');
  const [error, setError] = useState<Error>();
  const job = s?.job;
  const running = job?.status === 'running' || job?.status === 'queued';
  const [wasRunning, setWasRunning] = useState(false);
  useEffect(() => {
    if (wasRunning && !running) onChanged();
    setWasRunning(running);
  }, [running, wasRunning, onChanged]);
  useEffect(() => {
    if (!archive && s?.local_archives.length) setArchive(s.local_archives[0].name);
  }, [s?.local_archives, archive]);

  const act = async (fn: () => Promise<unknown>) => {
    setError(undefined);
    try {
      await fn();
      await status.reload();
      onChanged();
    } catch (e) {
      setError(e as Error);
    }
  };
  if (!s) return null;
  const approxMb = all ? 5500 : Math.round(count * 0.345 + 125);

  return (
    <div className="grid gap-6 xl:grid-cols-[1fr_1.2fr]">
      <Panel title="Data source" icon={Database} ticks>
        <KV items={[
          ['Dataset', <a key="d" className="inline-flex items-center gap-1 text-ink hover:text-accent" href={s.source.record_url} target="_blank" rel="noreferrer">AI4Mars merged v0.6 <ExternalLink className="h-3 w-3" /></a>],
          ['DOI', s.source.doi],
          ['Licence', s.source.license],
          ['Archive', `${s.source.archive} (${bytes(s.source.archive_bytes)})`],
          ['Subset used', 'Curiosity (MSL) NavCam'],
          ['Local folder', <span key="f" title={s.raw_dir}>{s.raw_dir.split(/[\\/]/).slice(-2).join('/')}</span>],
          ['Status', <span key="s" className="inline-flex gap-1.5">{s.indexed ? <StatusBadge state="completed" label="validated" /> : s.dataset_found ? <StatusBadge state="paused" label="not indexed" /> : <StatusBadge state="idle" label="not downloaded" />}</span>],
        ]} />
        <p className="mt-4 text-2xs leading-relaxed text-ink-3">
          Label encoding (from the dataset&apos;s own label_keys.json, verified on real files): 0 soil · 1 bedrock · 2 sand · 3 big rock · 255 unlabelled.
          Unlabelled pixels are excluded from training loss and from every metric. The Kaggle mirror can be imported as a local ZIP.
        </p>
      </Panel>

      <Panel title="Acquire or refresh" icon={CloudDownload}>
        <div className="space-y-4">
          <Segmented value={mode} onChange={setMode} ariaLabel="Acquisition mode" options={[
            { value: 'subset', label: 'Download subset' }, { value: 'full', label: 'Full archive' }, { value: 'import', label: 'Import local ZIP' },
          ]} />
          {mode === 'subset' && (
            <div className="space-y-3">
              <p className="text-xs text-ink-3">Fetches only the selected NavCam training pairs plus the complete 322-image expert test set directly from Zenodo (HTTP range requests, CRC-checked, resumable). Whole rover positions are chosen so frames from one location stay together.</p>
              <Field label={<span className="flex justify-between"><span>Training pairs</span><span className="num text-ink">{all ? 'all 16,064' : int(count)}</span></span>}
                hint={`≈ ${approxMb >= 1000 ? `${(approxMb / 1000).toFixed(1)} GB` : `${approxMb} MB`} download. Laptops: 1,000–4,000 is a good start.`}>
                <input type="range" min={200} max={16000} step={100} value={count} disabled={all} onChange={(e) => setCount(Number(e.target.value))} className="w-full" />
              </Field>
              <label className="flex items-center gap-2 text-xs text-ink-2"><input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} /> All Curiosity NavCam training pairs</label>
              <button className="btn-primary" disabled={running || connection !== 'online'} onClick={() => act(() => api.datasetDownload('subset', all ? null : count))}>
                <CloudDownload className="h-4 w-4" /> Download & validate
              </button>
            </div>
          )}
          {mode === 'full' && (
            <div className="space-y-3">
              <Callout tone="warn" title="16.2 GB download">Downloads the complete archive (all missions), verifies its MD5 checksum, then extracts only the Curiosity NavCam folders. Needs ~25 GB free disk space during extraction.</Callout>
              <label className="flex items-center gap-2 text-xs text-ink-2"><input type="checkbox" checked={confirmFull} onChange={(e) => setConfirmFull(e.target.checked)} /> I understand this downloads 16 GB</label>
              <button className="btn-primary" disabled={!confirmFull || running} onClick={() => act(() => api.datasetDownload('full', null, true))}><CloudDownload className="h-4 w-4" /> Download full archive</button>
            </div>
          )}
          {mode === 'import' && (
            <div className="space-y-3">
              <p className="text-xs text-ink-3">Place an AI4Mars ZIP (from Zenodo or the Kaggle mirror) in <code className="num text-ink-2">{s.imports_dir}</code>, then pick it here. Only the MSL image/label folders are extracted; paths that escape the folder are rejected.</p>
              {s.local_archives.length === 0 ? (
                <p className="rounded-lg border border-dashed border-line-strong px-3 py-4 text-center text-xs text-ink-3">No .zip files in the imports folder yet.</p>
              ) : (
                <select className="field" value={archive} onChange={(e) => setArchive(e.target.value)}>
                  {s.local_archives.map((a) => <option key={a.name} value={a.name}>{a.name} ({bytes(a.bytes)})</option>)}
                </select>
              )}
              <button className="btn-primary" disabled={!archive || running} onClick={() => act(() => api.datasetImport(archive))}><FileArchive className="h-4 w-4" /> Extract & validate</button>
            </div>
          )}
          <div className="flex flex-wrap items-center gap-2 border-t border-line/70 pt-3">
            <button className="btn-secondary text-xs" disabled={running || !s.dataset_found} onClick={() => act(() => api.datasetLoad())}><RefreshCw className="h-3.5 w-3.5" /> Re-scan & validate files on disk</button>
            {running && <button className="btn-danger text-xs" onClick={() => act(() => api.datasetCancel())}>Cancel</button>}
          </div>
          {job && (
            <div className="rounded-lg border border-line bg-sunken/60 p-3" aria-live="polite">
              <div className="mb-2 flex items-center justify-between text-xs">
                <span className="flex items-center gap-2"><StatusBadge state={job.status} /> <span className="text-ink-2">{String(job.params.action ?? job.kind)} · {job.stage}</span></span>
                <span className="num text-ink-2">{Math.round(job.progress * 100)}%</span>
              </div>
              <ProgressBar value={job.progress} tone={job.status === 'failed' ? 'bad' : job.status === 'completed' ? 'good' : 'accent'} />
              <div className="mt-1.5 truncate text-2xs text-ink-3">{job.error ?? job.message}</div>
            </div>
          )}
          {error && <ErrorBanner error={error} />}
        </div>
      </Panel>
    </div>
  );
}

/* -------------------------------------------------------------- statistics */

function StatsSection({ stats }: { stats: DatasetStatistics }) {
  const st = stats.statistics;
  const [split, setSplit] = useState<string>('train');
  const testVariant = st.test_variants.includes('min3') ? 'min3' : st.test_variants[0];
  const splitData: Record<string, SplitStats | undefined> = {
    train: st.splits.train, val: st.splits.val, ...Object.fromEntries(st.test_variants.map((v) => [`test-${v}`, st.splits.test[v]])),
  };
  const current = splitData[split] ?? st.splits.train;
  return (
    <>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <Stat label="Valid pairs" value={int(st.total_images)} sub={`${st.rover_positions} rover positions`} icon={Images} />
        <Stat label="Train" value={int(st.splits.train.images)} sub="crowdsourced labels" />
        <Stat label="Validation" value={int(st.splits.val.images)} sub={`${Math.round(stats.split_config.val_fraction * 100)}% · grouped by position`} />
        <Stat label="Test" value={int(st.splits.test[testVariant]?.images)} sub="expert labels (official)" tone="accent" />
        <Stat label="Unlabelled (train)" value={pct(st.splits.train.unlabeled_percent)} sub="excluded from loss" tone="warn" />
        <Stat label="Big rock (train)" value={pct(st.splits.train.class_percent_of_labeled.big_rock, 2)} sub="of labelled pixels" tone="bad" />
      </div>
      <div className="grid gap-6 xl:grid-cols-[1.1fr_1fr]">
        <Panel title="Class distribution" eyebrow="share of labelled pixels" icon={Layers}
          actions={<Segmented size="sm" value={split} onChange={setSplit} options={Object.keys(splitData).map((k) => ({ value: k, label: k }))} />}>
          <ClassBars percent={current.class_percent_of_labeled as Record<ClassKey, number>} height={190} valueDigits={2} />
          <div className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1 text-xs sm:grid-cols-4">
            {CLASS_KEYS.map((k) => (
              <div key={k}><div className="text-ink-3">{CLASS_LABEL[k]} in</div><div className="num text-ink">{int(current.images_containing_class[k])} <span className="text-ink-3">/ {int(current.images)} images</span></div></div>
            ))}
          </div>
        </Panel>
        <Panel title="Splits at a glance" eyebrow="labelled-pixel composition" icon={Layers}>
          <div className="space-y-3">
            {Object.entries(splitData).map(([name, s]) => s && (
              <div key={name}>
                <div className="mb-1 flex items-center justify-between text-xs">
                  <span className="font-mono text-ink-2">{name}</span>
                  <span className="num text-ink-3">{int(s.images)} imgs · {pct(s.unlabeled_percent)} unlabelled</span>
                </div>
                <CompositionBar percent={s.class_percent_of_labeled} height={12} />
              </div>
            ))}
            <ClassLegend className="pt-1" />
            <Callout tone="warn" title="Class imbalance">
              Big rock is {pct(st.splits.train.class_percent_of_labeled.big_rock, 2)} of labelled training pixels and {pct(st.splits.test[testVariant]?.class_percent_of_labeled.big_rock ?? null, 3)} of test ({testVariant}) pixels. Scores for this class are noisy and per-class metrics matter more than pixel accuracy.
            </Callout>
          </div>
        </Panel>
      </div>
    </>
  );
}

function TestVariants({ stats }: { stats: DatasetStatistics }) {
  const st = stats.statistics;
  return (
    <Panel title="Expert test-label variants" icon={ShieldCheck} padded={false}>
      <div className="overflow-x-auto">
        <table className="table-clean">
          <thead><tr><th>Variant</th><th className="text-right">Images</th><th className="text-right">Unlabelled</th>{CLASS_KEYS.map((k) => <th key={k} className="text-right">{CLASS_LABEL[k]}</th>)}</tr></thead>
          <tbody>
            {st.test_variants.map((v) => {
              const s = st.splits.test[v];
              return (
                <tr key={v}>
                  <td className="font-mono">{v}</td>
                  <td className="num text-right">{s.images}</td>
                  <td className="num text-right text-ink-2">{pct(s.unlabeled_percent)}</td>
                  {CLASS_KEYS.map((k) => <td key={k} className="num text-right">{pct(s.class_percent_of_labeled[k], 2)}</td>)}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="px-4 py-3 text-2xs leading-relaxed text-ink-3">
        minN = a pixel is labelled only if at least N of 3 expert labellers marked it with full agreement. min3 is the sparsest and strictest.
        Leakage check: {stats.leakage_check.train_val_shared_positions} rover positions shared between train and validation;{' '}
        {stats.leakage_check.test_images_sharing_position_with_train} of {stats.leakage_check.test_images} test images share a rover position with the training data (reported, not hidden — the official test partition is used unchanged).
      </p>
    </Panel>
  );
}

function ValidationSection({ report }: { report: ValidationReport }) {
  const v = report.validation;
  const checks: { label: string; ok: boolean; detail: string; warn?: boolean }[] = [
    { label: 'Every label has a matching image', ok: v.labels_without_image_count === 0, detail: `${v.labels_without_image_count} missing` },
    { label: 'All files readable', ok: v.unreadable_count === 0, detail: `${v.unreadable_count} unreadable` },
    { label: 'Image and label sizes match', ok: v.size_mismatch_count === 0, detail: `${v.size_mismatch_count} mismatched` },
    { label: 'Only documented label values (0–3, 255)', ok: v.unexpected_values_count === 0, detail: `observed {${v.observed_label_values.join(', ')}}` },
    { label: 'No train/test filename overlap', ok: v.train_test_filename_overlap.length === 0, detail: `${v.train_test_filename_overlap.length} overlapping` },
    { label: 'Fully unlabelled training masks excluded', ok: true, warn: v.empty_labels_count > 0, detail: `${v.empty_labels_count} excluded` },
    { label: 'Images without any label (unused)', ok: true, warn: v.images_without_label_count > 0, detail: `${v.images_without_label_count} images` },
  ];
  const problems = [...v.labels_without_image.map((f) => `missing image for ${f}`), ...v.unreadable.map((u) => `${u.file}: ${u.error}`),
    ...v.size_mismatch.map((m) => `${m.image}: image ${m.image_size.join('×')} vs label ${m.label_size.join('×')}`),
    ...v.unexpected_values.map((u) => `${u.file}: values ${u.values.join(', ')}`)];
  return (
    <Panel title="Validation report" icon={ShieldCheck} eyebrow={`scanned in ${report.scan_seconds}s`}>
      <ul className="space-y-2">
        {checks.map((c) => (
          <li key={c.label} className="flex items-center justify-between gap-3 text-sm">
            <span className="flex items-center gap-2">
              {!c.ok ? <XCircle className="h-4 w-4 text-bad" /> : c.warn ? <AlertTriangle className="h-4 w-4 text-warn" /> : <CheckCircle2 className="h-4 w-4 text-good" />}
              {c.label}
            </span>
            <span className="num text-xs text-ink-3">{c.detail}</span>
          </li>
        ))}
      </ul>
      <div className="mt-4 grid grid-cols-3 gap-2 text-center">
        {[['images found', v.images_found], ['train labels', v.train_labels_found], ['valid pairs', v.valid_pairs]].map(([k, n]) => (
          <div key={k as string} className="rounded-lg border border-line bg-sunken/60 py-2"><div className="num text-lg">{int(n as number)}</div><div className="eyebrow">{k}</div></div>
        ))}
      </div>
      {problems.length > 0 && (
        <details className="mt-3 text-xs"><summary className="cursor-pointer text-ink-2">Show {problems.length} problem file(s)</summary>
          <ul className="mt-2 max-h-40 overflow-auto font-mono text-2xs text-ink-3">{problems.map((p) => <li key={p}>{p}</li>)}</ul>
        </details>
      )}
    </Panel>
  );
}

/* ----------------------------------------------------------------- gallery */

function Gallery({ testVariants }: { testVariants: string[] }) {
  const [split, setSplit] = useState<string>('test');
  const [contains, setContains] = useState<string>('');
  const [view, setView] = useState<'image' | 'overlay' | 'label'>('overlay');
  const [page, setPage] = useState(0);
  const [open, setOpen] = useState<Sample | null>(null);
  const limit = 12;
  useEffect(() => setPage(0), [split, contains]);
  const samples = useRemote(() => api.datasetSamples({ split, contains, offset: page * limit, limit }), [split, contains, page]);
  const total = samples.data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / limit));
  const variantFor = (s: Sample) => (s.split === 'test' ? (testVariants.includes('min1') ? 'min1' : s.variants[0]) : 'train');
  return (
    <Panel title="Image–mask pairs" icon={Images}
      actions={<span className="num text-xs text-ink-3">{int(total)} samples</span>}>
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <Segmented size="sm" value={split} onChange={setSplit} ariaLabel="Split" options={[{ value: 'train', label: 'train' }, { value: 'val', label: 'val' }, { value: 'test', label: 'test' }]} />
        <select className="field w-auto py-1 text-xs" value={contains} onChange={(e) => setContains(e.target.value)} aria-label="Filter by class">
          <option value="">Any class</option>
          {CLASS_KEYS.map((k) => <option key={k} value={k}>Contains {CLASS_LABEL[k].toLowerCase()}</option>)}
        </select>
        <Segmented size="sm" value={view} onChange={setView} ariaLabel="Thumbnail view" options={[{ value: 'image', label: 'image' }, { value: 'overlay', label: 'overlay' }, { value: 'label', label: 'mask' }]} />
        <div className="ml-auto flex items-center gap-1">
          <button className="btn-ghost p-1.5" disabled={page === 0} onClick={() => setPage(page - 1)} aria-label="Previous page"><ChevronLeft className="h-4 w-4" /></button>
          <span className="num text-xs text-ink-3">{page + 1} / {pages}</span>
          <button className="btn-ghost p-1.5" disabled={page + 1 >= pages} onClick={() => setPage(page + 1)} aria-label="Next page"><ChevronRight className="h-4 w-4" /></button>
        </div>
      </div>
      {samples.error && <ErrorBanner error={samples.error} />}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 2xl:grid-cols-6">
        {samples.data?.items.map((s) => {
          const variant = variantFor(s);
          const lab = s.labels[variant];
          return (
            <button key={s.id} onClick={() => setOpen(s)} className="group overflow-hidden rounded-lg border border-line bg-sunken text-left transition-colors hover:border-accent/60">
              <div className="relative aspect-square overflow-hidden">
                <img src={api.sampleImageUrl(s.id, view, view === 'image' ? undefined : variant, 320)} alt={s.id} loading="lazy" className={cx('h-full w-full object-cover transition-transform duration-500 group-hover:scale-105', view === 'label' && 'pixelated')} />
                <span className="absolute left-1.5 top-1.5 rounded bg-black/60 px-1 font-mono text-[9px] text-white">{s.split}{s.split === 'test' ? ` · ${variant}` : ''}</span>
              </div>
              <div className="p-2">
                <div className="truncate font-mono text-[10px] text-ink-3">{s.id}</div>
                {lab && <CompositionBar percent={lab.class_percent} height={4} className="mt-1.5" />}
              </div>
            </button>
          );
        })}
      </div>
      {open && <SampleModal sample={open} onClose={() => setOpen(null)} />}
    </Panel>
  );
}

function SampleModal({ sample, onClose }: { sample: Sample; onClose: () => void }) {
  const { colorVersion } = useApp();
  const [variant, setVariant] = useState(sample.variants.includes('min1') ? 'min1' : sample.variants[0]);
  const models = useRemote(() => api.models(), []);
  const real = useMemo(() => (models.data?.models ?? []).filter((m) => !m.is_fixture).sort((a, b) => (b.best_val_miou ?? 0) - (a.best_val_miou ?? 0)), [models.data]);
  const [modelId, setModelId] = useState('');
  useEffect(() => { if (!modelId && real[0]) setModelId(real[0].id); }, [real, modelId]);
  const [pred, setPred] = useState<Analysis | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<Error>();
  const lab = sample.labels[variant];
  async function predict() {
    setBusy(true);
    setErr(undefined);
    try {
      setPred(await api.predictSample(modelId, sample.id));
    } catch (e) {
      setErr(e as Error);
    } finally {
      setBusy(false);
    }
  }
  return (
    <Modal open onClose={onClose} wide title={<span className="font-mono text-xs">{sample.id}</span>}>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
        <span className="chip">{sample.split}</span>
        <span className="chip">camera {sample.camera}</span>
        <span className="chip">site {sample.site} · drive {sample.drive}</span>
        <span className="chip">{sample.width}×{sample.height}</span>
        {sample.variants.length > 1 && <Segmented size="sm" value={variant} onChange={setVariant} options={sample.variants.map((v) => ({ value: v, label: v }))} />}
      </div>
      <div className={cx('grid gap-3', pred ? 'md:grid-cols-4' : 'md:grid-cols-3')}>
        {(['image', 'label', 'overlay'] as const).map((k) => (
          <figure key={k}>
            <img src={api.sampleImageUrl(sample.id, k, k === 'image' ? undefined : variant, 768, colorVersion)} alt={`${k}`} className={cx('w-full rounded-md border border-line', k === 'label' && 'pixelated')} />
            <figcaption className="eyebrow mt-1.5">{k === 'image' ? 'original' : k === 'label' ? `ground truth (${variant})` : 'ground-truth overlay'}</figcaption>
          </figure>
        ))}
        {pred && (
          <figure>
            <img src={api.analysisImageUrl(pred.id, 'mask', { v: colorVersion })} alt="Prediction" className="w-full rounded-md border border-accent/50" />
            <figcaption className="eyebrow mt-1.5 text-accent">prediction · {pred.model_name}</figcaption>
          </figure>
        )}
      </div>
      <ClassLegend className="mt-3" unlabeled />
      {lab && (
        <div className="mt-3 grid gap-4 md:grid-cols-2">
          <div>
            <div className="eyebrow mb-1.5">ground-truth composition ({variant}) · {pct(lab.unlabeled_percent)} unlabelled</div>
            <CompositionBar percent={lab.class_percent} height={10} />
            <div className="num mt-1.5 flex flex-wrap gap-x-3 text-2xs text-ink-2">{CLASS_KEYS.map((k) => <span key={k}>{CLASS_LABEL[k]} {pct(lab.class_percent[k])}</span>)}</div>
          </div>
          {pred && (
            <div>
              <div className="eyebrow mb-1.5">predicted composition (all pixels, including those unlabelled in ground truth)</div>
              <CompositionBar percent={pred.class_percent} height={10} />
              <div className="num mt-1.5 flex flex-wrap gap-x-3 text-2xs text-ink-2">{CLASS_KEYS.map((k) => <span key={k}>{CLASS_LABEL[k]} {pct(pred.class_percent[k])}</span>)}</div>
            </div>
          )}
        </div>
      )}
      <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-line pt-3">
        {real.length ? (
          <>
            <select className="field w-auto py-1.5 text-xs" value={modelId} onChange={(e) => setModelId(e.target.value)}>
              {real.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
            <button className="btn-primary text-xs" onClick={predict} disabled={busy || !modelId}><ScanSearch className="h-3.5 w-3.5" /> {busy ? 'Predicting…' : 'Compare with model prediction'}</button>
            {pred && <Link to={`/terrain?id=${pred.id}`} className="btn-ghost text-xs">Open in Terrain Intelligence</Link>}
          </>
        ) : <span className="text-xs text-ink-3">Train a model to compare its prediction with this ground truth.</span>}
        {err && <ErrorBanner error={err} className="w-full" />}
      </div>
    </Modal>
  );
}
