import { Cpu, FolderOpen, Info, Moon, Palette, RotateCcw, Save, Server, Sun } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Callout, ClassSwatch, ErrorBanner, KV, Loading, PageHeader, Panel, Segmented, StatusBadge } from '../components/ui';
import { useApp } from '../context/AppContext';
import { useRemote } from '../hooks/usePolling';
import { api } from '../services/api';
import type { PerClass } from '../types/api';
import { CLASS_KEYS, CLASS_LABEL, CVD_COLORS, DEFAULT_COLORS } from '../utils/format';

export default function Settings() {
  const { connection, health, theme, setTheme, colors, setColors } = useApp();
  const sys = useRemote(() => api.systemInfo(), [connection]);
  const [draft, setDraft] = useState<PerClass<string>>(colors);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<Error>();
  useEffect(() => setDraft(colors), [colors]);

  async function save(c: PerClass<string>) {
    setError(undefined);
    try {
      const r = await api.saveColors(c);
      setColors(r.colors);
      setSaved(true);
      window.setTimeout(() => setSaved(false), 1500);
    } catch (e) {
      setError(e as Error);
    }
  }
  const s = sys.data;
  return (
    <div className="space-y-6">
      <PageHeader eyebrow="10 · system" title="Settings" description="Connection, hardware, storage locations and display preferences." />
      <div className="grid gap-6 xl:grid-cols-2">
        <Panel title="Backend connection" icon={Server}>
          <KV items={[
            ['Status', <StatusBadge key="s" state={connection} />],
            ['API', <a key="a" className="text-accent hover:underline" href="/api/health" target="_blank" rel="noreferrer">/api → http://127.0.0.1:8000</a>],
            ['Interactive API docs', <a key="d" className="text-accent hover:underline" href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">:8000/docs</a>],
            ['Application version', health?.version ?? '—'],
            ['Python / PyTorch', s ? `${s.python} / ${s.torch}` : '—'],
            ['Torchvision', s?.torchvision ?? '—'],
            ['Platform', <span key="p" title={s?.platform}>{s?.platform ?? '—'}</span>],
          ]} />
          {sys.error && <ErrorBanner error={sys.error} className="mt-3" onRetry={sys.reload} />}
        </Panel>
        <Panel title="Compute device" icon={Cpu}>
          {!s ? <Loading /> : (
            <>
              <KV items={[
                ['Selected device', s.hardware.selected_device.toUpperCase()],
                ['CUDA available', s.hardware.cuda_available ? 'yes' : 'no'],
                ...s.hardware.gpus.map((g) => [`GPU ${g.index}`, `${g.name} · ${g.memory_free_gb}/${g.memory_total_gb} GB free`] as [string, string]),
                ['CPU', s.hardware.cpu || '—'],
                ['CPU threads', `${s.hardware.cpu_cores_logical} logical · ${s.hardware.cpu_cores_physical ?? '?'} physical · torch uses ${s.hardware.torch_threads}`],
                ['RAM', `${s.hardware.ram_available_gb} GB free of ${s.hardware.ram_total_gb} GB`],
                ['Disk free (data dir)', `${s.disk_free_gb} GB`],
              ]} />
              <Callout tone={s.hardware.cuda_available ? 'good' : 'info'} className="mt-3">{s.hardware.diagnosis}</Callout>
            </>
          )}
        </Panel>
        <Panel title="Storage locations" icon={FolderOpen}>
          {!s ? <Loading /> : (
            <>
              <KV items={[
                ['Dataset directory', s.paths.dataset_dir],
                ['Import ZIPs here', s.paths.imports_dir],
                ['Processed cache', s.paths.processed_dir],
                ['Model directory', s.paths.model_dir],
                ['Experiment runs', s.paths.experiments_dir],
                ['Database', s.paths.database],
              ]} />
              <p className="mt-3 text-2xs text-ink-3">Change these with <code className="num">MTI_DATA_DIR</code>, <code className="num">MTI_CHECKPOINT_DIR</code> and <code className="num">MTI_EXPERIMENTS_DIR</code> in <code className="num">.env</code>, then restart. Upload limit: {s.limits.max_upload_mb} MB. Stored: {s.counts.experiments} experiments, {s.counts.evaluations} evaluations, {s.counts.analyses} analyses.</p>
              {s.test_fixture_enabled && <Callout tone="info" className="mt-3">Test-fixture mode is ON (MTI_ENABLE_TEST_FIXTURE). An untrained, clearly labelled fixture model is registered for automated tests.</Callout>}
            </>
          )}
        </Panel>
        <Panel title="Theme" icon={theme === 'dark' ? Moon : Sun}>
          <Segmented value={theme} onChange={setTheme} options={[{ value: 'dark', label: 'Dark' }, { value: 'light', label: 'Light' }]} />
          <p className="mt-3 text-xs text-ink-3">Stored in this browser only.</p>
        </Panel>
      </div>

      <Panel title="Terrain class colours" icon={Palette}
        actions={<>
          <button className="btn-ghost text-xs" onClick={() => setDraft(CVD_COLORS)}>Colour-blind safe preset</button>
          <button className="btn-ghost text-xs" onClick={() => setDraft(DEFAULT_COLORS)}><RotateCcw className="h-3.5 w-3.5" /> Default</button>
          <button className="btn-primary text-xs" onClick={() => save(draft)}><Save className="h-3.5 w-3.5" /> {saved ? 'Saved' : 'Apply'}</button>
        </>}>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {CLASS_KEYS.map((k, i) => (
            <div key={k} className="rounded-lg border border-line bg-sunken/50 p-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">{CLASS_LABEL[k]}</span>
                <span className="num text-2xs text-ink-3">class id {i}</span>
              </div>
              <div className="mt-3 flex items-center gap-3">
                <input type="color" value={draft[k]} onChange={(e) => setDraft({ ...draft, [k]: e.target.value.toUpperCase() })} aria-label={`${CLASS_LABEL[k]} colour`}
                  className="h-10 w-14 cursor-pointer rounded border border-line bg-transparent" />
                <input className="field num py-1.5 text-xs" value={draft[k]} maxLength={7} onChange={(e) => setDraft({ ...draft, [k]: e.target.value })} aria-label={`${CLASS_LABEL[k]} hex`} />
              </div>
              <div className="mt-3 flex items-center gap-2 text-2xs text-ink-3"><ClassSwatch k={k} /> current</div>
            </div>
          ))}
        </div>
        {error && <ErrorBanner error={error} className="mt-3" />}
        <p className="mt-3 flex items-start gap-2 text-2xs text-ink-3"><Info className="mt-0.5 h-3 w-3 shrink-0" /> Colours change only how masks, overlays and charts are drawn. Class ids (0 soil, 1 bedrock, 2 sand, 3 big rock) and every stored prediction stay exactly the same. Blue and purple are close for some colour-vision deficiencies, so class names always accompany colours; the preset uses the Okabe–Ito palette.</p>
      </Panel>
    </div>
  );
}
