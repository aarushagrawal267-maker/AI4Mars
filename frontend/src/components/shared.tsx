import { BrainCircuit, ImagePlus, UploadCloud } from 'lucide-react';
import { useRef, useState, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../services/api';
import type { AnalysisSummary, ModelInfo } from '../types/api';
import { ago, archLabel, compact, cx, ratio } from '../utils/format';
import { EmptyState } from './ui';

export function ModelSelect({ models, value, onChange, id, includeFixtures = true }: {
  models: ModelInfo[];
  value: string;
  onChange: (id: string) => void;
  id?: string;
  includeFixtures?: boolean;
}) {
  const list = models.filter((m) => includeFixtures || !m.is_fixture);
  return (
    <select id={id} className="field" value={value} onChange={(e) => onChange(e.target.value)} aria-label="Model">
      {list.length === 0 && <option value="">No trained models</option>}
      {list.map((m) => (
        <option key={m.id} value={m.id}>
          {m.is_fixture ? '[TEST FIXTURE] ' : ''}{archLabel(m.arch)} · {m.name}
          {m.best_val_miou !== null ? ` · val mIoU ${ratio(m.best_val_miou, 1)}` : ''}
        </option>
      ))}
    </select>
  );
}

export function ModelCard({ model, selected, onClick }: { model: ModelInfo; selected?: boolean; onClick?: () => void }) {
  return (
    <button type="button" onClick={onClick}
      className={cx('w-full rounded-lg border p-3 text-left transition-colors',
        selected ? 'border-accent/60 bg-accent/[0.06]' : 'border-line bg-sunken/50 hover:border-line-strong')}>
      <div className="flex items-center justify-between gap-2">
        <span className="chip">{archLabel(model.arch)}</span>
        {model.is_fixture ? <span className="chip text-info">fixture</span> : <span className="num text-2xs text-ink-3">{ago(model.created_at)}</span>}
      </div>
      <div className="mt-2 truncate text-sm font-medium text-ink">{model.name}</div>
      <div className="num mt-1 flex flex-wrap gap-x-3 text-2xs text-ink-3">
        <span>{compact(model.parameters)} params</span>
        <span>{model.resolution}px</span>
        <span>{model.epochs_completed} ep</span>
        {model.best_val_miou !== null && <span className="text-ink-2">val mIoU {ratio(model.best_val_miou, 1)}</span>}
      </div>
    </button>
  );
}

export function NoModels() {
  return (
    <EmptyState icon={BrainCircuit} title="No trained model yet"
      action={<Link to="/training" className="btn-primary">Train a model</Link>}>
      Inference only runs with a model you trained locally. Load the dataset in the Dataset Explorer, then train U-Net or
      DeepLabV3 in Model Training. Nothing is faked while no model exists.
    </EmptyState>
  );
}

export function Dropzone({ onFile, disabled, maxMb = 20, children }: {
  onFile: (f: File) => void;
  disabled?: boolean;
  maxMb?: number;
  children?: ReactNode;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  return (
    <div
      role="button"
      tabIndex={0}
      aria-label="Upload a Mars image"
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
      onClick={() => !disabled && input.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const f = e.dataTransfer.files?.[0];
        if (f && !disabled) onFile(f);
      }}
      className={cx(
        'group relative flex cursor-pointer flex-col items-center justify-center overflow-hidden rounded-xl border border-dashed px-6 py-9 text-center transition-all',
        over ? 'border-accent bg-accent/[0.07]' : 'border-line-strong bg-sunken/60 hover:border-accent/60 hover:bg-accent/[0.03]',
        disabled && 'pointer-events-none opacity-50',
      )}>
      <div className="absolute inset-0 bg-grid opacity-60" />
      <div className="relative">
        <div className="mx-auto mb-3 grid h-11 w-11 place-items-center rounded-xl border border-line bg-raised transition-transform group-hover:-translate-y-0.5">
          {over ? <ImagePlus className="h-5 w-5 text-accent" /> : <UploadCloud className="h-5 w-5 text-accent" />}
        </div>
        <div className="text-sm font-medium text-ink">Drop a Mars rover image here, or click to browse</div>
        <div className="mt-1 text-xs text-ink-3">JPG, JPEG or PNG · up to {maxMb} MB · grayscale NavCam frames match the training data best</div>
        {children}
      </div>
      <input ref={input} type="file" accept=".jpg,.jpeg,.png,image/jpeg,image/png" className="hidden" data-testid="file-input"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = '';
        }} />
    </div>
  );
}

export function AnalysisThumb({ a, active, onClick }: { a: AnalysisSummary; active?: boolean; onClick?: () => void }) {
  return (
    <button type="button" onClick={onClick}
      className={cx('group flex w-full items-center gap-3 rounded-lg border p-2 text-left transition-colors',
        active ? 'border-accent/60 bg-accent/[0.06]' : 'border-transparent hover:border-line hover:bg-raised/60')}>
      <img src={api.analysisImageUrl(a.id, 'overlay', { alpha: 0.55 })} alt="" loading="lazy"
        className="h-11 w-11 shrink-0 rounded-md border border-line object-cover" />
      <div className="min-w-0 flex-1">
        <div className="truncate text-xs font-medium text-ink">{a.filename}</div>
        <div className="truncate text-2xs text-ink-3">
          {a.is_fixture ? 'fixture · ' : ''}{archLabel(a.arch)} · {ago(a.created_at)}
        </div>
      </div>
    </button>
  );
}
