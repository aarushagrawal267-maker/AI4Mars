import { useCallback, useEffect, useRef, useState } from 'react';

export interface Remote<T> {
  data: T | undefined;
  error: Error | undefined;
  loading: boolean;
  reload: () => Promise<void>;
  setData: (d: T | undefined) => void;
}

/**
 * Fetch data once (or every ``intervalMs`` while mounted). The interval may be changed on the fly,
 * e.g. faster while a job runs. Errors are kept alongside the last good data so the UI does not blank out.
 */
export function useRemote<T>(fetcher: () => Promise<T>, deps: unknown[] = [], intervalMs?: number | null): Remote<T> {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<Error>();
  const [loading, setLoading] = useState(true);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;
  const seq = useRef(0);

  const reload = useCallback(async () => {
    const id = ++seq.current;
    try {
      const d = await fetcherRef.current();
      if (id === seq.current) {
        setData(d);
        setError(undefined);
      }
    } catch (e) {
      if (id === seq.current) setError(e as Error);
    } finally {
      if (id === seq.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    setLoading(true);
    void reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    if (!intervalMs) return;
    const t = window.setInterval(() => void reload(), intervalMs);
    return () => window.clearInterval(t);
  }, [intervalMs, reload]);

  return { data, error, loading, reload, setData };
}

/** Persist a small UI preference in localStorage (never used for research data). */
export function useLocalState<T>(key: string, initial: T): [T, (v: T) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = window.localStorage.getItem(key);
      return raw ? (JSON.parse(raw) as T) : initial;
    } catch {
      return initial;
    }
  });
  const set = useCallback(
    (v: T) => {
      setValue(v);
      try {
        window.localStorage.setItem(key, JSON.stringify(v));
      } catch {
        /* storage unavailable */
      }
    },
    [key],
  );
  return [value, set];
}
