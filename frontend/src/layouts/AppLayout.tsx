import {
  Activity, BookOpen, BrainCircuit, Cpu, Database, FlaskConical, Gauge, LayoutDashboard, Map, Menu, Moon, Mountain,
  ScanSearch, Settings as SettingsIcon, Sun, Telescope, X, type LucideIcon,
} from 'lucide-react';
import { useEffect, useState } from 'react';
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom';
import { useApp } from '../context/AppContext';
import { useRemote } from '../hooks/usePolling';
import { api } from '../services/api';
import { cx, duration } from '../utils/format';

export interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  group: string;
}

export const NAV: NavItem[] = [
  { to: '/', label: 'Overview', icon: LayoutDashboard, group: 'Mission' },
  { to: '/analysis', label: 'Image Analysis', icon: ScanSearch, group: 'Analyse' },
  { to: '/dataset', label: 'Dataset Explorer', icon: Database, group: 'Research' },
  { to: '/training', label: 'Model Training', icon: BrainCircuit, group: 'Research' },
  { to: '/evaluation', label: 'Model Evaluation', icon: Gauge, group: 'Research' },
  { to: '/terrain', label: 'Terrain Intelligence', icon: Mountain, group: 'Analyse' },
  { to: '/navigation', label: 'Navigation Simulator', icon: Map, group: 'Analyse' },
  { to: '/experiments', label: 'Experiments', icon: FlaskConical, group: 'Research' },
  { to: '/docs', label: 'Documentation', icon: BookOpen, group: 'Reference' },
  { to: '/settings', label: 'Settings', icon: SettingsIcon, group: 'Reference' },
];

export function Logo({ compact = false }: { compact?: boolean }) {
  return (
    <Link to="/" className="flex items-center gap-3" aria-label="Mars Terrain Intelligence home">
      <svg viewBox="0 0 40 40" className="h-9 w-9 shrink-0" aria-hidden="true">
        <defs>
          <clipPath id="mti-planet"><circle cx="20" cy="20" r="13" /></clipPath>
        </defs>
        <circle cx="20" cy="20" r="13" fill="#B9532A" />
        <g clipPath="url(#mti-planet)">
          <rect x="7" y="7" width="13" height="13" fill="#D7773F" />
          <rect x="20" y="20" width="13" height="13" fill="#8E3B1C" />
          <rect x="20" y="7" width="13" height="13" fill="#B44BD8" opacity="0.55" />
          <rect x="7" y="20" width="13" height="13" fill="#35A3E8" opacity="0.5" />
          <path d="M7 20h26M20 7v26" stroke="#0b0c0e" strokeWidth="1.2" />
        </g>
        <circle cx="20" cy="20" r="13" fill="none" stroke="#E8743B" strokeWidth="1" />
        <ellipse cx="20" cy="20" rx="19" ry="6.5" fill="none" stroke="#E8743B" strokeWidth="1" opacity="0.6" transform="rotate(-20 20 20)" />
      </svg>
      {!compact && (
        <div className="leading-none">
          <div className="font-display text-[13px] font-semibold tracking-[0.16em] text-ink">MARS TERRAIN</div>
          <div className="mt-1 font-mono text-[10px] tracking-[0.32em] text-accent">INTELLIGENCE</div>
        </div>
      )}
    </Link>
  );
}

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  const { connection, health } = useApp();
  const training = useRemote(() => api.trainingStatus(), [], connection === 'online' ? 4000 : null);
  const t = training.data;
  let lastGroup = '';
  return (
    <div className="flex h-full flex-col">
      <div className="px-5 pb-5 pt-6">
        <Logo />
      </div>
      <nav className="flex-1 overflow-y-auto px-3" aria-label="Main">
        {NAV.map((item, i) => {
          const showGroup = item.group !== lastGroup;
          lastGroup = item.group;
          return (
            <div key={item.to}>
              {showGroup && i > 0 && <div className="mx-2 my-2 h-px bg-line/70" />}
              <NavLink to={item.to} end={item.to === '/'} onClick={onNavigate}
                className={({ isActive }) => cx(
                  'group relative flex items-center gap-3 rounded-lg px-2.5 py-2 text-[13px] transition-colors',
                  isActive ? 'bg-raised text-ink ring-1 ring-line' : 'text-ink-2 hover:bg-raised/60 hover:text-ink',
                )}>
                {({ isActive }) => (
                  <>
                    {isActive && <span className="absolute -left-3 top-1/2 h-5 w-[3px] -translate-y-1/2 rounded-r bg-accent" />}
                    <span className={cx('num w-5 text-[10px]', isActive ? 'text-accent' : 'text-ink-3')}>{String(i + 1).padStart(2, '0')}</span>
                    <item.icon className={cx('h-4 w-4', isActive ? 'text-accent' : 'text-ink-3 group-hover:text-ink-2')} strokeWidth={1.75} />
                    <span className="truncate">{item.label}</span>
                  </>
                )}
              </NavLink>
            </div>
          );
        })}
      </nav>
      <div className="space-y-2.5 border-t border-line/70 p-4">
        {t?.active && (
          <Link to="/training" onClick={onNavigate} className="block rounded-lg border border-accent/30 bg-accent/[0.07] p-2.5 transition-colors hover:bg-accent/[0.12]">
            <div className="flex items-center justify-between text-2xs">
              <span className="flex items-center gap-1.5 font-medium text-accent">
                <Activity className="h-3 w-3" /> {t.state === 'paused' ? 'Training paused' : 'Training'}
              </span>
              <span className="num text-ink-2">{Math.round((t.progress ?? 0) * 100)}%</span>
            </div>
            <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-sunken">
              <div className="h-full bg-accent transition-[width] duration-700" style={{ width: `${(t.progress ?? 0) * 100}%` }} />
            </div>
            <div className="mt-1 truncate text-2xs text-ink-3">
              {t.arch === 'deeplabv3' ? 'DeepLabV3' : 'U-Net'} · epoch {t.epoch ?? 0}/{t.epochs_total} · ETA {duration(t.eta_seconds)}
            </div>
          </Link>
        )}
        <div className="flex items-center justify-between text-2xs">
          <span className="flex items-center gap-1.5 text-ink-3">
            <span className={cx('h-1.5 w-1.5 rounded-full', connection === 'online' ? 'bg-good' : connection === 'offline' ? 'bg-bad' : 'bg-warn animate-pulse2')} />
            Backend {connection}
          </span>
          <span className="num text-ink-3">v{health?.version ?? '—'}</span>
        </div>
        <p className="text-[10px] leading-snug text-ink-3/80">
          Independent student research project. Not affiliated with or endorsed by NASA/JPL.
        </p>
      </div>
    </div>
  );
}

function Topbar({ onMenu }: { onMenu: () => void }) {
  const { connection, theme, setTheme } = useApp();
  const location = useLocation();
  const current = NAV.find((n) => (n.to === '/' ? location.pathname === '/' : location.pathname.startsWith(n.to)));
  const sys = useRemote(() => api.systemInfo(), [connection], null);
  const hw = sys.data?.hardware;
  return (
    <div className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b border-line/70 bg-bg/80 px-4 backdrop-blur-md md:px-8">
      <button className="btn-ghost -ml-2 p-2 lg:hidden" onClick={onMenu} aria-label="Open navigation"><Menu className="h-5 w-5" /></button>
      <div className="flex min-w-0 items-center gap-2 text-sm">
        <Telescope className="hidden h-4 w-4 text-ink-3 sm:block" strokeWidth={1.6} />
        <span className="hidden text-ink-3 sm:inline">MTI</span>
        <span className="hidden text-ink-3 sm:inline">/</span>
        <span className="truncate font-medium text-ink">{current?.label ?? 'Not found'}</span>
      </div>
      <div className="ml-auto flex items-center gap-2">
        {hw && (
          <span className="chip hidden md:inline-flex" title={hw.diagnosis}>
            <Cpu className="h-3 w-3" />
            {hw.cuda_available ? `CUDA · ${hw.gpus[0]?.name ?? 'GPU'}` : `CPU · ${hw.cpu_cores_logical ?? '?'} threads`}
          </span>
        )}
        <span className={cx('chip', connection === 'online' ? 'text-good' : connection === 'offline' ? 'text-bad' : 'text-warn')}>
          <span className={cx('h-1.5 w-1.5 rounded-full', connection === 'online' ? 'bg-good' : connection === 'offline' ? 'bg-bad' : 'bg-warn')} />
          {connection === 'online' ? 'API online' : connection === 'offline' ? 'API offline' : 'Connecting'}
        </span>
        <button className="btn-ghost p-2" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`} title="Toggle theme">
          {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </button>
      </div>
    </div>
  );
}

export default function AppLayout() {
  const [open, setOpen] = useState(false);
  const { connection } = useApp();
  const location = useLocation();
  useEffect(() => setOpen(false), [location.pathname]);
  return (
    <div className="flex min-h-full">
      <aside className="hidden w-64 shrink-0 border-r border-line/70 bg-surface/60 lg:block">
        <div className="sticky top-0 h-screen"><SidebarContent /></div>
      </aside>
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-black/60" onClick={() => setOpen(false)} />
          <aside className="absolute inset-y-0 left-0 w-72 border-r border-line bg-surface animate-fade-in">
            <button className="btn-ghost absolute right-2 top-2 p-2" onClick={() => setOpen(false)} aria-label="Close navigation"><X className="h-4 w-4" /></button>
            <SidebarContent onNavigate={() => setOpen(false)} />
          </aside>
        </div>
      )}
      <div className="relative flex min-w-0 flex-1 flex-col">
        <div className="pointer-events-none absolute inset-x-0 top-0 h-80 bg-grid [mask-image:linear-gradient(to_bottom,black,transparent)]" />
        <Topbar onMenu={() => setOpen(true)} />
        {connection === 'offline' && (
          <div role="alert" className="relative border-b border-bad/30 bg-bad/[0.08] px-4 py-2.5 text-sm text-ink-2 md:px-8">
            <span className="font-medium text-bad">Backend unreachable.</span> Start it with <code className="num text-ink">.\start.ps1</code> (or{' '}
            <code className="num text-ink">uvicorn backend.app.main:app --port 8000</code>). Data on this page will refresh automatically once it is up.
          </div>
        )}
        <main className="relative mx-auto w-full max-w-[1500px] flex-1 px-4 py-7 md:px-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
