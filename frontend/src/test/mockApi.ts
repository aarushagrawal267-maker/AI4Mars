import { vi } from 'vitest';

type Handler = (url: URL, init?: RequestInit) => unknown | Promise<unknown>;
export interface MockRoute {
  method?: string;
  path: string | RegExp;
  status?: number;
  body: unknown | Handler;
}

/** Replace fetch with a router over mocked API routes. Unmatched calls fail loudly with 404. */
export function mockApi(routes: MockRoute[]) {
  const calls: { method: string; path: string; body?: BodyInit | null }[] = [];
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), 'http://localhost');
    const method = (init?.method ?? 'GET').toUpperCase();
    calls.push({ method, path: url.pathname, body: init?.body });
    const route = routes.find((r) => (r.method ?? 'GET') === method &&
      (typeof r.path === 'string' ? url.pathname === `/api${r.path}` : r.path.test(url.pathname)));
    if (!route) {
      return new Response(JSON.stringify({ detail: `unmocked ${method} ${url.pathname}` }), { status: 404, headers: { 'content-type': 'application/json' } });
    }
    const body = typeof route.body === 'function' ? await (route.body as Handler)(url, init) : route.body;
    return new Response(JSON.stringify(body), { status: route.status ?? 200, headers: { 'content-type': 'application/json' } });
  });
  vi.stubGlobal('fetch', fn);
  return { fn, calls };
}

export const health = { method: 'GET', path: '/health', body: { status: 'ok', app: 'Mars Terrain Intelligence', version: '1.0.0', time: 0 } };
export const colors = {
  method: 'GET', path: '/settings/colors',
  body: {
    colors: { soil: '#B0682C', bedrock: '#35A3E8', sand: '#E6AE35', big_rock: '#B44BD8' },
    defaults: { soil: '#B0682C', bedrock: '#35A3E8', sand: '#E6AE35', big_rock: '#B44BD8' },
    unlabeled: '#1F2328', ignore_index: 255,
    classes: [{ id: 0, key: 'soil', label: 'Soil' }, { id: 1, key: 'bedrock', label: 'Bedrock' }, { id: 2, key: 'sand', label: 'Sand' }, { id: 3, key: 'big_rock', label: 'Big rock' }],
  },
};
export const hardware = {
  cuda_available: false, gpus: [], selected_device: 'cpu', cpu: 'x86_64', cpu_cores_logical: 8, cpu_cores_physical: 4,
  torch_threads: 4, ram_total_gb: 16, ram_available_gb: 10, diagnosis: 'No CUDA GPU detected by PyTorch.',
};
export const systemInfo = {
  method: 'GET', path: '/system/info',
  body: { app: 'MTI', version: '1.0.0', python: '3.11', platform: 'test', torch: '2.x', torchvision: '0.x', hardware,
    paths: { dataset_dir: '/d/raw', imports_dir: '/d/raw/imports', processed_dir: '/d/p', model_dir: '/c', experiments_dir: '/e', database: 'sqlite' },
    disk_free_gb: 100, limits: { max_upload_mb: 20, max_output_side: 1536 }, counts: { experiments: 0, evaluations: 0, analyses: 0 }, test_fixture_enabled: false },
};
export const idleTraining = { method: 'GET', path: '/training/status', body: { active: false, experiment_id: null, state: 'idle' } };

export const pc = (soil: number, bedrock: number, sand: number, big_rock: number) => ({ soil, bedrock, sand, big_rock });

export const model = {
  id: 'unet-1', name: 'U-Net test run', arch: 'unet', status: 'completed', is_fixture: false, created_at: '2026-01-01T00:00:00',
  resolution: 192, model_config: { base_channels: 16 }, epochs_completed: 3, best_epoch: 3, best_val_miou: 0.5, parameters: 1812068,
  n_train: 100, checkpoint_bytes: 7_000_000, pretrained_loaded: null, latest_test_evaluation: null,
};
export const model2 = { ...model, id: 'deeplab-1', name: 'DeepLab test run', arch: 'deeplabv3', best_val_miou: 0.6 };

export const analysis = {
  id: 'abc123', created_at: '2026-01-01T00:00:00', experiment_id: 'unet-1', model_name: 'U-Net test run', arch: 'unet',
  source: 'upload', filename: 'mars.png', sample_id: null, is_fixture: false,
  meta: { input_size: [192, 192], output_size: [100, 100], original_size: [100, 100], inference_ms: 12.5, device: 'cpu', resolution: 192,
    mean_probs: pc(0.4, 0.3, 0.2, 0.1), model_parameters: 1812068, color_input: false },
  dominant_class: 'soil', class_percent: pc(50, 30, 20, 0), mean_confidence: 0.8,
  stats: {
    height: 100, width: 100, valid_pixels: 10000, unlabeled_pixels: 0,
    class_pixels: pc(5000, 3000, 2000, 0), class_percent: pc(50, 30, 20, 0), dominant_class: 'soil',
    spatial: Object.fromEntries(['soil', 'bedrock', 'sand', 'big_rock'].map((k) => [k, { pixels: k === 'big_rock' ? 0 : 1, centroid: null, bbox: null, regions: 1,
      largest_region_percent_of_class: 100, largest_region_percent_of_image: 10, zone_share_percent: { upper_left: 100 } }])),
    zone_composition: Object.fromEntries(['upper', 'middle', 'lower'].flatMap((r) => ['left', 'centre', 'right'].map((c) => [`${r}_${c}`, { percent: pc(50, 30, 20, 0), dominant: 'soil' }]))),
    band_profile: [{ band: 1, from_top_percent: 0, ...pc(50, 30, 20, 0) }],
    confidence: { mean: 0.8, median: 0.82, min: 0.3, per_class_mean: pc(0.9, 0.7, 0.6, null),
      histogram: Array.from({ length: 10 }, (_, i) => ({ from: i / 10, to: (i + 1) / 10, pixels: 1000, percent: 10 })),
      low_confidence_percent: 5, high_confidence_percent: 40, calibrated: false, note: 'uncalibrated' },
    uncertainty: { mean_entropy: 0.3, high_uncertainty_percent: 10, per_class_mean_entropy: pc(0.2, 0.3, 0.4, null), definition: 'entropy' },
    summary: 'The predicted terrain is dominated by soil (50.0% of classified pixels).',
  },
};
