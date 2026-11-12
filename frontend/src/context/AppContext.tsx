import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { api } from '../services/api';
import type { ClassKey, Health, PerClass } from '../types/api';
import { DEFAULT_COLORS } from '../utils/format';

type Theme = 'dark' | 'light';
export type Connection = 'connecting' | 'online' | 'offline';

interface AppState {
  connection: Connection;
  health: Health | undefined;
  colors: PerClass<string>;
  setColors: (c: PerClass<string>) => void;
  colorVersion: number;
  theme: Theme;
  setTheme: (t: Theme) => void;
  colorOf: (k: ClassKey) => string;
}

const Ctx = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [connection, setConnection] = useState<Connection>('connecting');
  const [health, setHealth] = useState<Health>();
  const [colors, setColorsState] = useState<PerClass<string>>(DEFAULT_COLORS);
  const [colorVersion, setColorVersion] = useState(0);
  const [theme, setThemeState] = useState<Theme>(() =>
    (document.documentElement.dataset.theme as Theme) === 'light' ? 'light' : 'dark',
  );

  useEffect(() => {
    let alive = true;
    const ping = async () => {
      try {
        const h = await api.health();
        if (!alive) return;
        setHealth(h);
        setConnection((prev) => {
          if (prev !== 'online') {
            api.colors().then((c) => alive && setColorsState(c.colors)).catch(() => undefined);
          }
          return 'online';
        });
      } catch {
        if (alive) setConnection('offline');
      }
    };
    void ping();
    const t = window.setInterval(ping, 5000);
    return () => {
      alive = false;
      window.clearInterval(t);
    };
  }, []);

  const setTheme = useCallback((t: Theme) => {
    document.documentElement.dataset.theme = t;
    try {
      window.localStorage.setItem('mti.theme', t);
    } catch {
      /* ignore */
    }
    setThemeState(t);
  }, []);

  const setColors = useCallback((c: PerClass<string>) => {
    setColorsState(c);
    setColorVersion((v) => v + 1); // bust cached server-rendered masks
  }, []);

  const value = useMemo<AppState>(
    () => ({ connection, health, colors, setColors, colorVersion, theme, setTheme, colorOf: (k) => colors[k] }),
    [connection, health, colors, setColors, colorVersion, theme, setTheme],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error('useApp must be used inside AppProvider');
  return v;
}
