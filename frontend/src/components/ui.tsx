import { AlertTriangle, CheckCircle2, Info, Loader2, RefreshCw, X, XCircle, type LucideIcon } from 'lucide-react';
import { useEffect, type ReactNode } from 'react';
import { useApp } from '../context/AppContext';
import type { ClassKey } from '../types/api';
import { CLASS_KEYS, CLASS_LABEL, cx } from '../utils/format';

/* ------------------------------------------------------------------ layout */

export function PageHeader({ eyebrow, title, description, actions }: {
  eyebrow?: string;
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="mb-6 flex flex-wrap items-end justify-between gap-4 animate-fade-in">
      <div className="min-w-0">
        {eyebrow && <div className="eyebrow mb-1.5 text-accent/90">{eyebrow}</div>}
        <h1 className="text-[1.65rem] font-semibold leading-tight text-ink">{title}</h1>
        {description && <p className="mt-1.5 max-w-3xl text-sm leading-relaxed text-ink-2">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

export function Panel({ title, eyebrow, icon: Icon, actions, children, className, bodyClassName, ticks, padded = true, id }: {
  title?: ReactNode;
  eyebrow?: string;
  icon?: LucideIcon;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  ticks?: boolean;
  padded?: boolean;
  id?: string;
}) {
  return (
    <section id={id} className={cx('panel animate-fade-in', ticks && 'panel-ticks', className)}>
      {(title || actions || eyebrow) && (
        <div className="flex items-center justify-between gap-3 border-b border-line/70 px-4 py-3">
          <div className="flex min-w-0 items-center gap-2.5">
            {Icon && <Icon className="h-4 w-4 shrink-0 text-accent" strokeWidth={1.75} />}
            <div className="min-w-0">
              {eyebrow && <div className="eyebrow leading-none">{eyebrow}</div>}
              {title && <h2 className="truncate text-sm font-semibold text-ink">{title}</h2>}
            </div>
          </div>
          {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
        </div>
      )}
      <div className={cx(padded && 'p-4', bodyClassName)}>{children}</div>
    </section>
  );
}

/* ------------------------------------------------------------------- stats */

type Tone = 'default' | 'accent' | 'good' | 'warn' | 'bad' | 'info';
const toneText: Record<Tone, string> = {
  default: 'text-ink',
  accent: 'text-accent',
  good: 'text-good',
  warn: 'text-warn',
  bad: 'text-bad',
  info: 'text-info',
};

export function Stat({ label, value, sub, tone = 'default', icon: Icon, className }: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  tone?: Tone;
  icon?: LucideIcon;
  className?: string;
}) {
  return (
    <div className={cx('panel px-4 py-3.5', className)}>
      <div className="flex items-center justify-between gap-2">
        <span className="eyebrow">{label}</span>
        {Icon && <Icon className="h-3.5 w-3.5 text-ink-3" strokeWidth={1.75} />}
      </div>
      <div className={cx('num mt-1.5 text-2xl font-medium leading-none', toneText[tone])}>{value}</div>
      {sub && <div className="mt-1.5 truncate text-xs text-ink-3">{sub}</div>}
    </div>
  );
}

export function KV({ items, className }: { items: [ReactNode, ReactNode][]; className?: string }) {
  return (
    <dl className={cx('grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm', className)}>
      {items.map(([k, v], i) => (
        <div key={i} className="contents">
          <dt className="text-ink-3">{k}</dt>
          <dd className="num min-w-0 truncate text-right text-ink">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

/* --------------------------------------------------------------- feedback */

const STATE_TONE: Record<string, Tone> = {
  completed: 'good',
  online: 'good',
  running: 'accent',
  preparing: 'accent',
  queued: 'info',
  paused: 'warn',
  stopped: 'warn',
  interrupted: 'warn',
  cancelled: 'warn',
  failed: 'bad',
  offline: 'bad',
  fixture: 'info',
  idle: 'default',
};

export function StatusBadge({ state, label }: { state: string; label?: string }) {
  const tone = STATE_TONE[state] ?? 'default';
  const dot: Record<Tone, string> = {
    default: 'bg-ink-3',
    accent: 'bg-accent animate-pulse2',
    good: 'bg-good',
    warn: 'bg-warn',
    bad: 'bg-bad',
    info: 'bg-info',
  };
  return (
    <span className={cx('chip capitalize', toneText[tone])}>
      <span className={cx('h-1.5 w-1.5 rounded-full', dot[tone])} />
      {label ?? state}
    </span>
  );
}

export function Callout({ tone = 'info', title, children, className, icon }: {
  tone?: 'info' | 'warn' | 'bad' | 'good';
  title?: ReactNode;
  children: ReactNode;
  className?: string;
  icon?: LucideIcon;
}) {
  const Icon = icon ?? { info: Info, warn: AlertTriangle, bad: XCircle, good: CheckCircle2 }[tone];
  const styles = {
    info: 'border-info/30 bg-info/[0.07] text-info',
    warn: 'border-warn/35 bg-warn/[0.07] text-warn',
    bad: 'border-bad/35 bg-bad/[0.08] text-bad',
    good: 'border-good/30 bg-good/[0.07] text-good',
  }[tone];
  return (
    <div className={cx('flex gap-3 rounded-lg border px-3.5 py-3 text-sm', styles, className)} role="note">
      <Icon className="mt-0.5 h-4 w-4 shrink-0" strokeWidth={1.9} />
      <div className="min-w-0 text-ink-2">
        {title && <div className="mb-0.5 font-medium text-ink">{title}</div>}
        {children}
      </div>
    </div>
  );
}

export function ErrorBanner({ error, onRetry, className }: { error: Error | string | undefined; onRetry?: () => void; className?: string }) {
  if (!error) return null;
  const message = typeof error === 'string' ? error : error.message;
  return (
    <div role="alert" className={cx('flex items-start gap-3 rounded-lg border border-bad/35 bg-bad/[0.08] px-3.5 py-3 text-sm', className)}>
      <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-bad" />
      <div className="min-w-0 flex-1 break-words text-ink-2">{message}</div>
      {onRetry && (
        <button className="btn-ghost -my-1 px-2 py-1 text-xs" onClick={onRetry}>
          <RefreshCw className="h-3.5 w-3.5" /> Retry
        </button>
      )}
    </div>
  );
}

export function EmptyState({ icon: Icon, title, children, action, className }: {
  icon: LucideIcon;
  title: string;
  children?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cx('flex flex-col items-center justify-center rounded-xl border border-dashed border-line-strong/70 bg-grid px-6 py-12 text-center', className)}>
      <div className="mb-4 grid h-12 w-12 place-items-center rounded-xl border border-line bg-raised">
        <Icon className="h-5 w-5 text-accent" strokeWidth={1.6} />
      </div>
      <h3 className="text-base font-semibold text-ink">{title}</h3>
      {children && <div className="mt-1.5 max-w-md text-sm leading-relaxed text-ink-2">{children}</div>}
      {action && <div className="mt-5 flex flex-wrap justify-center gap-2">{action}</div>}
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return <Loader2 className={cx('h-4 w-4 animate-spin', className)} aria-label="Loading" />;
}

export function Loading({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="flex items-center gap-2.5 py-10 text-sm text-ink-3 justify-center">
      <Spinner /> {label}
    </div>
  );
}

export function ProgressBar({ value, tone = 'accent', className, label }: {
  value: number;
  tone?: 'accent' | 'good' | 'warn' | 'bad';
  className?: string;
  label?: string;
}) {
  const v = Math.max(0, Math.min(1, value || 0));
  const bar = { accent: 'bg-accent', good: 'bg-good', warn: 'bg-warn', bad: 'bg-bad' }[tone];
  return (
    <div className={cx('h-1.5 w-full overflow-hidden rounded-full bg-sunken ring-1 ring-inset ring-line', className)}
      role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(v * 100)} aria-label={label}>
      <div className={cx('h-full rounded-full transition-[width] duration-500 ease-out', bar)} style={{ width: `${v * 100}%` }} />
    </div>
  );
}

/* ---------------------------------------------------------------- controls */

export function Field({ label, hint, children, className, htmlFor }: {
  label: ReactNode;
  hint?: ReactNode;
  children: ReactNode;
  className?: string;
  htmlFor?: string;
}) {
  return (
    <div className={cx('min-w-0', className)}>
      <label htmlFor={htmlFor} className="mb-1.5 block text-xs font-medium text-ink-2">{label}</label>
      {children}
      {hint && <div className="mt-1 text-2xs leading-snug text-ink-3">{hint}</div>}
    </div>
  );
}

export function Segmented<T extends string>({ options, value, onChange, size = 'md', className, ariaLabel }: {
  options: { value: T; label: ReactNode; title?: string; disabled?: boolean }[];
  value: T;
  onChange: (v: T) => void;
  size?: 'sm' | 'md';
  className?: string;
  ariaLabel?: string;
}) {
  return (
    <div role="tablist" aria-label={ariaLabel} className={cx('inline-flex rounded-lg border border-line bg-sunken p-0.5', className)}>
      {options.map((o) => (
        <button key={o.value} role="tab" type="button" aria-selected={value === o.value} title={o.title} disabled={o.disabled}
          onClick={() => onChange(o.value)}
          className={cx(
            'rounded-md font-medium transition-colors disabled:opacity-40',
            size === 'sm' ? 'px-2 py-1 text-2xs' : 'px-3 py-1.5 text-xs',
            value === o.value ? 'bg-raised text-ink shadow-sm ring-1 ring-line-strong' : 'text-ink-3 hover:text-ink',
          )}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Toggle({ checked, onChange, label, description, disabled }: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: ReactNode;
  description?: ReactNode;
  disabled?: boolean;
}) {
  return (
    <label className={cx('flex cursor-pointer items-start gap-3', disabled && 'cursor-not-allowed opacity-50')}>
      <button type="button" role="switch" aria-checked={checked} disabled={disabled} onClick={() => onChange(!checked)}
        className={cx('relative mt-0.5 h-5 w-9 shrink-0 rounded-full border transition-colors',
          checked ? 'border-accent bg-accent' : 'border-line-strong bg-sunken')}>
        <span className={cx('absolute top-0.5 h-3.5 w-3.5 rounded-full bg-white shadow transition-all', checked ? 'left-[18px]' : 'left-0.5')} />
      </button>
      <span className="min-w-0">
        <span className="block text-sm text-ink">{label}</span>
        {description && <span className="block text-2xs text-ink-3">{description}</span>}
      </span>
    </label>
  );
}

export function Modal({ open, onClose, title, children, wide }: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  children: ReactNode;
  wide?: boolean;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/60 p-4 backdrop-blur-sm animate-fade-in" onMouseDown={onClose}>
      <div role="dialog" aria-modal="true" className={cx('panel panel-ticks max-h-[92vh] w-full overflow-auto', wide ? 'max-w-6xl' : 'max-w-lg')}
        onMouseDown={(e) => e.stopPropagation()}>
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-line bg-surface/95 px-4 py-3 backdrop-blur">
          <h2 className="truncate text-sm font-semibold">{title}</h2>
          <button className="btn-ghost p-1.5" onClick={onClose} aria-label="Close"><X className="h-4 w-4" /></button>
        </div>
        <div className="p-4">{children}</div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------- terrain */

export function ClassSwatch({ k, size = 10 }: { k: ClassKey; size?: number }) {
  const { colorOf } = useApp();
  return <span className="inline-block shrink-0 rounded-[3px] ring-1 ring-black/20" style={{ width: size, height: size, background: colorOf(k) }} />;
}

export function ClassLegend({ values, className, unlabeled }: {
  values?: Partial<Record<ClassKey, ReactNode>>;
  className?: string;
  unlabeled?: boolean;
}) {
  return (
    <ul className={cx('flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs', className)} aria-label="Terrain classes">
      {CLASS_KEYS.map((k) => (
        <li key={k} className="flex items-center gap-1.5">
          <ClassSwatch k={k} />
          <span className="text-ink-2">{CLASS_LABEL[k]}</span>
          {values?.[k] !== undefined && <span className="num text-ink">{values[k]}</span>}
        </li>
      ))}
      {unlabeled && (
        <li className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-[3px] bg-[#1F2328] ring-1 ring-line-strong" />
          <span className="text-ink-3">Unlabelled</span>
        </li>
      )}
    </ul>
  );
}

/** Horizontal 100% composition strip with a 2px gap between segments. */
export function CompositionBar({ percent, height = 10, className }: {
  percent: Partial<Record<ClassKey, number | null>>;
  height?: number;
  className?: string;
}) {
  const { colorOf } = useApp();
  const parts = CLASS_KEYS.map((k) => ({ k, v: Math.max(0, percent[k] ?? 0) })).filter((p) => p.v > 0);
  const total = parts.reduce((s, p) => s + p.v, 0) || 1;
  return (
    <div className={cx('flex w-full gap-[2px] overflow-hidden rounded-full bg-sunken', className)} style={{ height }}>
      {parts.map((p) => (
        <div key={p.k} title={`${CLASS_LABEL[p.k]} ${p.v.toFixed(2)}%`} style={{ width: `${(p.v / total) * 100}%`, background: colorOf(p.k) }}
          className="h-full min-w-[2px]" />
      ))}
    </div>
  );
}

export function FixtureBadge() {
  return (
    <span className="chip border-info/40 bg-info/10 text-info" title="Untrained deterministic fixture used for automated tests. Not a real prediction.">
      TEST FIXTURE · not a real prediction
    </span>
  );
}
