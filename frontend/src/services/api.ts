/**
 * REST client for the FastAPI backend. Every value shown in the UI comes through here.
 * In development Vite proxies /api to http://127.0.0.1:8000 (see vite.config.ts).
 */
import type {
  Analysis, AnalysisSummary, ColorsResponse, CostsResponse, DatasetStatistics, DatasetStatus, Estimate, Evaluation,
  Experiment, ExperimentDetail, Health, Job, ModelInfo, Overview, PerClass, PlanResult, SamplePage, SystemInfo,
  TerrainCost, TerrainGrid, TrainConfig, TrainingDefaults, TrainingStatus, TreeNode, ValidationReport,
} from '../types/api';

export const API_BASE: string = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '') ?? '';

import { demoApi } from '../demo/demoApi';
import { IS_DEMO } from '../demo/env';
import { ApiError } from './apiError';

export { ApiError };

function describe(detail: unknown): string {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    // FastAPI validation errors: [{loc, msg}]
    return detail
      .map((d) => {
        const loc = Array.isArray(d?.loc) ? d.loc.filter((x: unknown) => x !== 'body').join('.') : '';
        return loc ? `${loc}: ${d?.msg}` : String(d?.msg ?? d);
      })
      .join('; ');
  }
  return 'Request failed';
}

export const ADMIN_TOKEN_KEY = 'mti.adminToken';

function adminToken(): string {
  try {
    return window.localStorage.getItem(ADMIN_TOKEN_KEY) ?? '';
  } catch {
    return '';
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  const token = adminToken();
  if (token) init = { ...init, headers: { ...(init.headers as Record<string, string> | undefined), 'X-Admin-Token': token } };
  try {
    res = await fetch(`${API_BASE}/api${path}`, init);
  } catch {
    throw new ApiError(0, 'Cannot reach the backend at port 8000. Is it running? (start.ps1 starts it)');
  }
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) message = describe(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, message);
  }
  const type = res.headers.get('content-type') ?? '';
  return (type.includes('application/json') ? res.json() : res.text()) as Promise<T>;
}

const json = (method: string, body?: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: body === undefined ? undefined : JSON.stringify(body),
});

/** Absolute URL for <img>/<a href> targets served by the API. */
export const apiUrl = (path: string) => `${API_BASE}/api${path}`;

export type RenderKind = 'original' | 'mask' | 'overlay' | 'side_by_side' | 'confidence' | 'uncertainty' | 'class_ids' | 'confidence_raw';

const httpApi = {
  // system
  health: () => request<Health>('/health'),
  systemInfo: () => request<SystemInfo>('/system/info'),
  overview: () => request<Overview>('/overview'),

  // dataset
  datasetStatus: () => request<DatasetStatus>('/dataset/status'),
  datasetLoad: () => request<Job>('/dataset/load', json('POST')),
  datasetDownload: (mode: 'subset' | 'full', train_count: number | null, confirm_full = false) =>
    request<Job>('/dataset/download', json('POST', { mode, train_count, confirm_full })),
  datasetImport: (archive: string) => request<Job>('/dataset/import', json('POST', { archive })),
  datasetCancel: () => request<{ cancelled: boolean }>('/dataset/cancel', json('POST')),
  datasetStatistics: () => request<DatasetStatistics>('/dataset/statistics'),
  datasetValidation: () => request<ValidationReport>('/dataset/validation'),
  datasetStructure: () => request<{ root: string; tree: TreeNode[] }>('/dataset/structure'),
  datasetSamples: (params: { split?: string; offset?: number; limit?: number; contains?: string }) => {
    const q = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v !== undefined && v !== '' && q.set(k, String(v)));
    return request<SamplePage>(`/dataset/samples?${q}`);
  },
  sampleImageUrl: (id: string, kind: 'image' | 'label' | 'overlay', variant?: string, size = 512, v = 0) =>
    apiUrl(`/dataset/samples/${encodeURIComponent(id)}/${kind}?size=${size}${variant ? `&variant=${variant}` : ''}&v=${v}`),

  // training
  trainingDefaults: () => request<TrainingDefaults>('/training/defaults'),
  trainingEstimate: (cfg: TrainRequest) => request<Estimate>('/training/estimate', json('POST', cfg)),
  trainingStart: (cfg: TrainRequest & { name?: string; confirm_large?: boolean }) =>
    request<{ experiment_id: string; estimate: Estimate }>('/training/start', json('POST', cfg)),
  trainingStop: () => request<{ ok: boolean }>('/training/stop', json('POST')),
  trainingPause: () => request<{ ok: boolean }>('/training/pause', json('POST')),
  trainingResume: () => request<{ ok: boolean }>('/training/resume', json('POST')),
  trainingStatus: () => request<TrainingStatus>('/training/status'),
  trainingLogs: (experimentId?: string, tail = 300) =>
    request<{ experiment_id: string | null; lines: string[] }>(
      `/training/logs?tail=${tail}${experimentId ? `&experiment_id=${encodeURIComponent(experimentId)}` : ''}`,
    ),

  // models & experiments
  models: () => request<{ models: ModelInfo[] }>('/models'),
  experiments: () => request<{ experiments: Experiment[] }>('/experiments'),
  experiment: (id: string) => request<ExperimentDetail>(`/experiments/${encodeURIComponent(id)}`),
  updateExperiment: (id: string, body: { name?: string; notes?: string }) =>
    request<ExperimentDetail>(`/experiments/${encodeURIComponent(id)}`, json('PATCH', body)),
  deleteExperiment: (id: string) => request<{ deleted: string }>(`/experiments/${encodeURIComponent(id)}`, json('DELETE')),
  experimentReportUrl: (id: string, format: 'json' | 'csv') =>
    apiUrl(`/experiments/${encodeURIComponent(id)}/report?format=${format}`),

  // inference
  predictUpload: (modelId: string, file: File) => {
    const form = new FormData();
    form.append('model_id', modelId);
    form.append('file', file);
    return request<Analysis>('/predict', { method: 'POST', body: form });
  },
  predictSample: (modelId: string, sampleId: string) => {
    const form = new FormData();
    form.append('model_id', modelId);
    form.append('sample_id', sampleId);
    return request<Analysis>('/predict', { method: 'POST', body: form });
  },
  analyses: (limit = 50) => request<{ analyses: AnalysisSummary[] }>(`/analyses?limit=${limit}`),
  analysis: (id: string) => request<Analysis>(`/analyses/${encodeURIComponent(id)}`),
  deleteAnalysis: (id: string) => request<{ deleted: string }>(`/analyses/${encodeURIComponent(id)}`, json('DELETE')),
  analysisImageUrl: (id: string, kind: RenderKind, opts: { alpha?: number; download?: boolean; v?: number } = {}) =>
    apiUrl(
      `/analyses/${encodeURIComponent(id)}/image/${kind}?alpha=${opts.alpha ?? 0.5}${opts.download ? '&download=true' : ''}&v=${opts.v ?? 0}`,
    ),
  analysisGrid: (id: string, size: number) => request<TerrainGrid>(`/analyses/${encodeURIComponent(id)}/grid?size=${size}`),

  // evaluation
  evaluate: (body: { model_id: string; split: 'test' | 'val'; label_variant?: string | null; max_images?: number | null }) =>
    request<Job>('/evaluate', json('POST', body)),
  evaluateStatus: () => request<{ job: Job | null }>('/evaluate/status'),
  evaluateCancel: () => request<{ cancelled: boolean }>('/evaluate/cancel', json('POST')),
  evaluations: (experimentId?: string) =>
    request<{ evaluations: Evaluation[] }>(
      `/evaluations${experimentId ? `?experiment_id=${encodeURIComponent(experimentId)}` : ''}`,
    ),
  deleteEvaluation: (id: number) => request<{ deleted: number }>(`/evaluations/${id}`, json('DELETE')),
  evaluationReportUrl: (ids: number[], format: 'csv' | 'json') =>
    apiUrl(`/evaluations/report?ids=${ids.join(',')}&format=${format}`),

  // navigation & settings
  costs: () => request<CostsResponse>('/navigation/costs'),
  saveCosts: (costs: PerClass<TerrainCost>) => request<CostsResponse>('/navigation/costs', json('PUT', costs)),
  resetCosts: () => request<CostsResponse>('/navigation/costs/reset', json('POST')),
  plan: (body: {
    analysis_id: string;
    grid_size: number;
    start: [number, number];
    goal: [number, number];
    costs?: PerClass<TerrainCost>;
    blocked?: [number, number][];
    allow_diagonal?: boolean;
    uncertainty_weight?: number;
    alternatives?: boolean;
  }) => request<PlanResult>('/navigation/plan', json('POST', body)),
  colors: () => request<ColorsResponse>('/settings/colors'),
  saveColors: (colors: Partial<PerClass<string>>) => request<ColorsResponse>('/settings/colors', json('PUT', colors)),
  resetColors: () => request<ColorsResponse>('/settings/colors/reset', json('POST')),
};

/** In the browser edition every call is answered locally (exported results + in-browser inference). */
export const api: typeof httpApi = IS_DEMO ? (demoApi as unknown as typeof httpApi) : httpApi;

/** Training request body (the API names the per-architecture options `model_options`). */
export type TrainRequest = Omit<TrainConfig, 'model_config' | 'num_workers'> & { model_options: Record<string, unknown> };

export function toTrainRequest(cfg: TrainConfig): TrainRequest {
  const { model_config, num_workers: _nw, ...rest } = cfg;
  return { ...rest, model_options: model_config ?? {} };
}
