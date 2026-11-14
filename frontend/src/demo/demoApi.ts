/**
 * Browser-edition implementation of the API used by the pages. Results, experiments and dataset statistics come
 * from demo/data.json (exported from the local database); predictions run in the browser with the exported models.
 * Actions that need the Python backend (training, evaluation, dataset changes, deletion of stored results) are refused.
 */
import { ApiError } from '../services/apiError';
import type {
  Analysis, AnalysisSummary, ClassKey, ColorsResponse, CostsResponse, DatasetStatistics, DatasetStatus, Estimate, Evaluation,
  Experiment, ExperimentDetail, Health, Job, ModelInfo, Overview, PerClass, PlanResult, SamplePage, Sample, SystemInfo,
  TerrainCost, TerrainGrid, TrainingDefaults, TrainingStatus, TreeNode, ValidationReport,
} from '../types/api';
import { type BrowserModel, MAX_OUTPUT_SIDE, predict, type Prediction } from './inference';
import { plan as planRoute } from './planner';
import { grayUrl, heatRgba, INFERNO, maskRgba, overlayRgba, sideBySideUrl, toUrl, VIRIDIS } from './render';
import { classGrid, terrainStatistics } from './terrain';

export const DEMO_REFUSAL = 'Not available in the browser edition: training, evaluation and dataset changes need the local '
  + 'Python backend. Install the full app from GitHub to do this.';
const refuse = (): never => { throw new ApiError(403, DEMO_REFUSAL); };

const DEFAULT_COLORS: PerClass<string> = { soil: '#B0682C', bedrock: '#35A3E8', sand: '#E6AE35', big_rock: '#B44BD8' };
const DEFAULT_COSTS: PerClass<TerrainCost> = {
  soil: { cost: 1, passable: true }, bedrock: { cost: 2, passable: true }, sand: { cost: 5, passable: true }, big_rock: { cost: 20, passable: true },
};
const COST_DISCLAIMER = 'Movement costs are experimental assumptions chosen for demonstration. They are not validated measurements of Mars rover '
  + 'mobility. Terrain class alone cannot determine whether a rover can safely drive somewhere: slope, rock size, wheel sinkage, rover state '
  + 'and mission rules all matter. This simulator is not connected to any rover.';

interface DemoExperiment extends Experiment { log: string[]; history: ExperimentDetail['history']; browser_model: boolean }
interface DemoData {
  experiments: DemoExperiment[];
  evaluations: Evaluation[];
  dataset: DatasetStatistics & { validation: ValidationReport['validation']; scan_seconds: number; split_counts: DatasetStatus['split_counts']; samples_included: number };
  tree: TreeNode[];
  colors: PerClass<string>;
}

let dataPromise: Promise<DemoData> | null = null;
const data = (): Promise<DemoData> => (dataPromise ??= fetch('demo/data.json').then((r) => { if (!r.ok) throw new ApiError(500, 'Demo data is missing.'); return r.json(); }));
let modelsPromise: Promise<BrowserModel[]> | null = null;
const browserModels = (): Promise<BrowserModel[]> => (modelsPromise ??= fetch('demo/models/models.json').then((r) => r.json()));
let samplesPromise: Promise<Sample[]> | null = null;
const samples = (): Promise<Sample[]> => (samplesPromise ??= fetch('demo/samples.json').then((r) => r.json()));

const store = {
  get<T>(key: string, fallback: T): T {
    try { const v = window.localStorage.getItem(key); return v ? (JSON.parse(v) as T) : fallback; } catch { return fallback; }
  },
  set(key: string, v: unknown) { try { window.localStorage.setItem(key, JSON.stringify(v)); } catch { /* ignore */ } },
};
let palette: PerClass<string> = store.get('mti.demo.colors', DEFAULT_COLORS);

/* ------------------------------------------------------------ analyses */

interface Stored { summary: Analysis; pred: Prediction; cache: Map<string, string> }
const analyses = new Map<string, Stored>();
const nowIso = () => new Date().toISOString().replace('Z', '');
const newId = () => Math.random().toString(16).slice(2, 10) + Date.now().toString(16).slice(-8);

async function runPrediction(modelId: string, img: CanvasImageSource, w: number, h: number, filename: string, source: 'upload' | 'dataset',
  sampleId: string | null): Promise<Analysis> {
  const models = await browserModels();
  const model = models.find((m) => m.id === modelId);
  if (!model) throw new ApiError(404, `Unknown model: ${modelId}`);
  const exp = (await data()).experiments.find((e) => e.id === modelId);
  const pred = await predict(model, img, w, h, (_f, msg) => window.dispatchEvent(new CustomEvent('mti-demo-progress', { detail: msg })));
  const stats = terrainStatistics(pred.cls, pred.width, pred.height, pred.conf, pred.unc);
  const id = newId();
  const summary: Analysis = {
    id, created_at: nowIso(), experiment_id: modelId, model_name: exp?.name ?? modelId, arch: model.arch, source, filename, sample_id: sampleId,
    is_fixture: false,
    meta: {
      input_size: pred.inputSize, output_size: [pred.height, pred.width], original_size: pred.originalSize,
      inference_ms: Math.round(pred.inferenceMs * 1000) / 1000, device: 'browser (WebAssembly)', resolution: model.resolution,
      mean_probs: Object.fromEntries((['soil', 'bedrock', 'sand', 'big_rock'] as ClassKey[]).map((k, i) => [k, pred.meanProbs[i]])) as PerClass<number>,
      model_parameters: model.parameters, color_input: pred.colorInput,
    },
    dominant_class: stats.dominant_class, class_percent: stats.class_percent, mean_confidence: stats.confidence?.mean ?? null, stats,
  };
  analyses.set(id, { summary, pred, cache: new Map() });
  return summary;
}

function render(id: string, kind: string, alpha: number, v: number): string {
  const a = analyses.get(id);
  if (!a) return '';
  const key = `${kind}|${alpha}|${v}`;
  const hit = a.cache.get(key);
  if (hit) return hit;
  const { cls, conf, unc, rgb, width: w, height: h } = a.pred;
  let url = '';
  switch (kind) {
    case 'original': url = toUrl(new Uint8ClampedArray(rgb), w, h); break;
    case 'mask': url = toUrl(maskRgba(cls, palette), w, h); break;
    case 'overlay': url = toUrl(overlayRgba(rgb, cls, palette, alpha), w, h); break;
    case 'side_by_side': url = sideBySideUrl([new Uint8ClampedArray(rgb), maskRgba(cls, palette), overlayRgba(rgb, cls, palette, alpha)], w, h); break;
    case 'confidence': url = toUrl(heatRgba(conf, VIRIDIS), w, h); break;
    case 'uncertainty': url = toUrl(heatRgba(unc, INFERNO), w, h); break;
    case 'class_ids': url = grayUrl(cls, w, h); break;
    case 'confidence_raw': url = grayUrl(Float32Array.from(conf, (c) => Math.round(c * 255)), w, h); break;
    default: url = '';
  }
  a.cache.set(key, url);
  return url;
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new ApiError(400, 'The image could not be decoded.'));
    img.src = src;
  });
}

/* ---------------------------------------------------------------- helpers */

async function modelInfos(): Promise<ModelInfo[]> {
  const [d, models] = await Promise.all([data(), browserModels()]);
  return models.map((m) => {
    const e = d.experiments.find((x) => x.id === m.id)!;
    const tests = d.evaluations.filter((v) => v.experiment_id === m.id && v.split === 'test');
    const latest = tests[0];
    return {
      id: m.id, name: e.name, arch: m.arch, status: e.status, is_fixture: false, created_at: e.created_at, resolution: m.resolution,
      model_config: e.config.model_config, epochs_completed: e.epochs_completed, best_epoch: e.best_epoch, best_val_miou: e.best_val_miou,
      parameters: m.parameters, n_train: e.n_train, checkpoint_bytes: m.onnx_bytes, pretrained_loaded: e.pretrained_loaded,
      latest_test_evaluation: latest ? { id: latest.id, label_variant: latest.label_variant, images: latest.images, mean_iou: latest.metrics.mean_iou,
        pixel_accuracy: latest.metrics.pixel_accuracy, created_at: latest.created_at } : null,
    };
  });
}

function trainingStatusOf(e: DemoExperiment | undefined): TrainingStatus {
  if (!e) return { active: false, experiment_id: null, state: 'idle' };
  return {
    active: false, experiment_id: e.id, name: e.name, arch: e.arch, config: e.config, state: e.status as TrainingStatus['state'],
    message: e.message ?? undefined, epoch: e.epochs_completed, epochs_total: e.config.epochs, progress: 1, best_val_miou: e.best_val_miou,
    best_epoch: e.best_epoch, elapsed_seconds: e.elapsed_seconds, eta_seconds: 0, device: e.device ?? undefined, parameters: e.parameters ?? undefined,
    n_train: e.n_train ?? undefined, n_val: e.n_val ?? undefined, pretrained_loaded: e.pretrained_loaded, has_checkpoint: e.has_checkpoint,
    error: e.error, history: e.history,
  };
}

const publicExperiment = (e: DemoExperiment): Experiment => {
  const { log: _log, history: _h, browser_model: _b, ...rest } = e;
  const last = e.history[e.history.length - 1];
  return { ...rest, final_train_loss: last?.train_loss ?? null, final_val_loss: last?.val_loss ?? null };
};

const jsonUrl = (obj: unknown) => `data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(obj, null, 2))}`;

/* ------------------------------------------------------------------- API */

export const demoApi = {
  health: async (): Promise<Health> => ({ status: 'ok', app: 'Mars Terrain Intelligence', version: '1.0.0', time: Date.now() / 1000 }),
  systemInfo: async (): Promise<SystemInfo> => {
    const d = await data();
    const nav = navigator as Navigator & { deviceMemory?: number };
    return {
      app: 'Mars Terrain Intelligence', version: '1.0.0 (browser edition)', python: 'n/a (browser)', platform: navigator.userAgent,
      torch: 'models exported to ONNX', torchvision: 'n/a',
      hardware: {
        cuda_available: false, gpus: [], selected_device: 'browser', cpu: 'this device', cpu_cores_logical: navigator.hardwareConcurrency ?? null,
        cpu_cores_physical: null, torch_threads: 1, ram_total_gb: nav.deviceMemory ?? 0, ram_available_gb: nav.deviceMemory ?? 0,
        diagnosis: 'Browser edition: the trained models run on this device in WebAssembly (onnxruntime-web 1.17.3). Nothing is sent to a server.',
      },
      paths: { dataset_dir: 'bundled with this page (120 test images)', imports_dir: 'n/a', processed_dir: 'n/a', model_dir: 'bundled ONNX models', experiments_dir: 'bundled results', database: 'bundled results (read-only)' },
      disk_free_gb: 0, limits: { max_upload_mb: 20, max_output_side: MAX_OUTPUT_SIDE },
      counts: { experiments: d.experiments.length, evaluations: d.evaluations.length, analyses: analyses.size }, test_fixture_enabled: false,
      demo: true,
    } as SystemInfo;
  },
  overview: async (): Promise<Overview> => {
    const [d, models, sys] = await Promise.all([data(), modelInfos(), demoApi.systemInfo()]);
    const recent = [...analyses.values()].map((a) => a.summary).reverse().slice(0, 8);
    const latest = d.experiments[0];
    return {
      dataset: { dataset_found: true, indexed: true, total_images: d.dataset.statistics.total_images, split_counts: d.dataset.split_counts, job: null },
      models, trained_model_count: models.length,
      latest_experiment: latest ? { id: latest.id, name: latest.name, arch: latest.arch, status: latest.status, created_at: latest.created_at,
        epochs_completed: latest.epochs_completed, epochs_total: latest.config.epochs, best_val_miou: latest.best_val_miou, message: latest.message } : null,
      training: trainingStatusOf(undefined), latest_analysis: recent[0] ?? null, recent_analyses: recent,
      evaluations: d.evaluations.map((e) => ({ id: e.id, experiment_id: e.experiment_id, experiment_name: e.experiment_name, arch: e.arch, split: e.split,
        label_variant: e.label_variant, images: e.images, created_at: e.created_at, is_fixture: false, mean_iou: e.metrics.mean_iou,
        pixel_accuracy: e.metrics.pixel_accuracy, mean_dice: e.metrics.mean_dice })),
      hardware: sys.hardware,
    };
  },

  datasetStatus: async (): Promise<DatasetStatus> => {
    const d = await data();
    return {
      raw_dir: 'bundled with this page', imports_dir: 'n/a', dataset_found: true, layout: d.dataset.layout, indexed: true, index_created: d.dataset.created,
      fingerprint: d.dataset.fingerprint, total_images: d.dataset.statistics.total_images, split_counts: d.dataset.split_counts, needs_index: false,
      job: null, local_archives: [],
      source: { name: 'AI4Mars: A Dataset for Terrain-Aware Autonomous Driving on Mars (merged v0.6)', record_url: 'https://zenodo.org/records/15995036',
        doi: '10.5281/zenodo.15995036', archive: 'ai4mars-dataset-merged-0.6.zip', archive_bytes: 16232481989, license: 'CC-BY-4.0',
        kaggle_mirror: 'https://www.kaggle.com/datasets/yash92328/ai4mars-terrainaware-autonomous-driving-on-mars', subset: 'Curiosity (MSL) NavCam' },
    };
  },
  datasetLoad: async (): Promise<Job> => refuse(),
  datasetDownload: async (..._a: unknown[]): Promise<Job> => refuse(),
  datasetImport: async (_a: string): Promise<Job> => refuse(),
  datasetCancel: async () => ({ cancelled: false }),
  datasetStatistics: async (): Promise<DatasetStatistics> => {
    const d = (await data()).dataset;
    return { created: d.created, fingerprint: d.fingerprint, classes: d.classes, ignore_index: d.ignore_index, split_config: d.split_config,
      statistics: d.statistics, leakage_check: d.leakage_check, layout: d.layout };
  },
  datasetValidation: async (): Promise<ValidationReport> => {
    const d = (await data()).dataset;
    return { created: d.created, scan_seconds: d.scan_seconds, validation: d.validation, leakage_check: d.leakage_check };
  },
  datasetStructure: async () => ({ root: 'data/raw (on the machine that trained the models)', tree: (await data()).tree }),
  datasetSamples: async (p: { split?: string; offset?: number; limit?: number; contains?: string }): Promise<SamplePage> => {
    let items = (await samples()).filter((s) => !p.split || s.split === p.split);
    if (p.contains) items = items.filter((s) => Object.values(s.labels).some((l) => (l.class_percent[p.contains as ClassKey] ?? 0) > 0));
    const offset = p.offset ?? 0, limit = p.limit ?? 12;
    return { total: items.length, offset, limit, items: items.slice(offset, offset + limit) };
  },
  sampleImageUrl: (id: string, kind: 'image' | 'label' | 'overlay', variant?: string) => {
    if (kind === 'image') return `demo/samples/${id}_image.jpg`;
    const v = variant && variant !== 'train' ? variant : 'min1';
    return `demo/samples/${id}_${kind}_${v}.${kind === 'label' ? 'png' : 'jpg'}`;
  },

  trainingDefaults: async (): Promise<TrainingDefaults> => {
    const d = await data();
    const e = d.experiments.find((x) => x.arch === 'unet') ?? d.experiments[0];
    return { config: { ...e.config }, reason: DEMO_REFUSAL, cuda_available: false, train_available: d.dataset.split_counts.train,
      val_available: d.dataset.split_counts.val,
      options: { architectures: [], resolutions: [128, 160, 192, 224, 256, 320, 384, 448, 512], devices: ['auto', 'cpu'] } };
  },
  trainingEstimate: async (_c: unknown): Promise<Estimate> => refuse(),
  trainingStart: async (_c: unknown): Promise<{ experiment_id: string; estimate: Estimate }> => refuse(),
  trainingStop: async () => refuse(),
  trainingPause: async () => refuse(),
  trainingResume: async () => refuse(),
  trainingStatus: async (): Promise<TrainingStatus> => trainingStatusOf((await data()).experiments[0]),
  trainingLogs: async (experimentId?: string) => {
    const d = await data();
    const e = d.experiments.find((x) => x.id === experimentId) ?? d.experiments[0];
    return { experiment_id: e?.id ?? null, lines: e?.log ?? [] };
  },

  models: async () => ({ models: await modelInfos() }),
  experiments: async () => ({ experiments: (await data()).experiments.map(publicExperiment) }),
  experiment: async (id: string): Promise<ExperimentDetail> => {
    const d = await data();
    const e = d.experiments.find((x) => x.id === id);
    if (!e) throw new ApiError(404, 'Experiment not found');
    return { ...publicExperiment(e), history: e.history, evaluations: d.evaluations.filter((v) => v.experiment_id === id) };
  },
  updateExperiment: async (_id: string, _b: unknown): Promise<ExperimentDetail> => refuse(),
  deleteExperiment: async (_id: string) => refuse(),
  experimentReportUrl: (id: string, _format: 'json' | 'csv') => jsonUrl({ note: 'Open the full app to download reports.', experiment: id }),

  predictUpload: async (modelId: string, file: File): Promise<Analysis> => {
    let bitmap: ImageBitmap;
    try {
      bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
    } catch {
      throw new ApiError(400, 'The file is not a readable image.');
    }
    if (bitmap.width < 32 || bitmap.height < 32) throw new ApiError(400, 'Image is too small (minimum 32 x 32 pixels).');
    if (bitmap.width * bitmap.height > 40_000_000) throw new ApiError(400, 'Image is too large; maximum is 40 megapixels.');
    return runPrediction(modelId, bitmap, bitmap.width, bitmap.height, file.name.slice(0, 300), 'upload', null);
  },
  predictSample: async (modelId: string, sampleId: string): Promise<Analysis> => {
    const img = await loadImage(`demo/samples/${sampleId}_image.jpg`);
    return runPrediction(modelId, img, img.naturalWidth, img.naturalHeight, `${sampleId}.JPG`, 'dataset', sampleId);
  },
  analyses: async (_limit = 50): Promise<{ analyses: AnalysisSummary[] }> => ({ analyses: [...analyses.values()].map((a) => a.summary).reverse() }),
  analysis: async (id: string): Promise<Analysis> => {
    const a = analyses.get(id);
    if (!a) throw new ApiError(404, 'This analysis is no longer in memory (the browser edition keeps results only until the page is reloaded).');
    return a.summary;
  },
  deleteAnalysis: async (id: string) => { analyses.delete(id); return { deleted: id }; },
  analysisImageUrl: (id: string, kind: string, opts: { alpha?: number; download?: boolean; v?: number } = {}) =>
    render(id, kind, opts.alpha ?? 0.5, opts.v ?? 0),
  analysisGrid: async (id: string, size: number): Promise<TerrainGrid> => {
    const a = analyses.get(id);
    if (!a) throw new ApiError(404, 'Analysis not found');
    return classGrid(a.pred.cls, a.pred.width, a.pred.height, size, a.pred.unc);
  },

  evaluate: async (_b: unknown): Promise<Job> => refuse(),
  evaluateStatus: async () => ({ job: null }),
  evaluateCancel: async () => ({ cancelled: false }),
  evaluations: async (experimentId?: string) => ({ evaluations: (await data()).evaluations.filter((e) => !experimentId || e.experiment_id === experimentId) }),
  deleteEvaluation: async (_id: number) => refuse(),
  evaluationReportUrl: (ids: number[], _format: 'csv' | 'json') => jsonUrl({ note: 'Open the full app to download reports.', evaluations: ids }),

  costs: async (): Promise<CostsResponse> => ({ costs: store.get('mti.demo.costs', DEFAULT_COSTS), defaults: DEFAULT_COSTS, disclaimer: COST_DISCLAIMER }),
  saveCosts: async (costs: PerClass<TerrainCost>): Promise<CostsResponse> => { store.set('mti.demo.costs', costs); return { costs, defaults: DEFAULT_COSTS, disclaimer: COST_DISCLAIMER }; },
  resetCosts: async (): Promise<CostsResponse> => { store.set('mti.demo.costs', DEFAULT_COSTS); return { costs: DEFAULT_COSTS, defaults: DEFAULT_COSTS, disclaimer: COST_DISCLAIMER }; },
  plan: async (b: { analysis_id: string; grid_size: number; start: [number, number]; goal: [number, number]; costs?: PerClass<TerrainCost>;
    blocked?: [number, number][]; allow_diagonal?: boolean; uncertainty_weight?: number; alternatives?: boolean }): Promise<PlanResult> => {
    const g = await demoApi.analysisGrid(b.analysis_id, b.grid_size);
    const inside = ([r, c]: [number, number]) => r >= 0 && r < g.rows && c >= 0 && c < g.cols;
    if (!inside(b.start) || !inside(b.goal)) throw new ApiError(422, `Start and destination must be inside the ${g.rows} x ${g.cols} grid.`);
    if (b.start[0] === b.goal[0] && b.start[1] === b.goal[1]) throw new ApiError(422, 'Start and destination must be different cells.');
    const costs = b.costs ?? store.get('mti.demo.costs', DEFAULT_COSTS);
    const res = planRoute(g.cells, costs, b.start, b.goal, b.blocked ?? [], b.allow_diagonal ?? true, g.uncertainty, b.uncertainty_weight ?? 0, b.alternatives ?? true);
    return { ...res, costs_used: costs } as PlanResult;
  },
  colors: async (): Promise<ColorsResponse> => ({
    colors: palette, defaults: DEFAULT_COLORS, unlabeled: '#1F2328', ignore_index: 255,
    classes: [{ id: 0, key: 'soil', label: 'Soil' }, { id: 1, key: 'bedrock', label: 'Bedrock' }, { id: 2, key: 'sand', label: 'Sand' }, { id: 3, key: 'big_rock', label: 'Big rock' }],
  }),
  saveColors: async (c: Partial<PerClass<string>>): Promise<ColorsResponse> => {
    for (const v of Object.values(c)) if (v && !/^#?[0-9a-fA-F]{6}$/.test(v)) throw new ApiError(422, `Invalid hex colour: ${v}`);
    palette = { ...palette, ...Object.fromEntries(Object.entries(c).filter(([, v]) => v).map(([k, v]) => [k, `#${String(v).replace('#', '').toUpperCase()}`])) } as PerClass<string>;
    store.set('mti.demo.colors', palette);
    return demoApi.colors();
  },
  resetColors: async (): Promise<ColorsResponse> => { palette = { ...DEFAULT_COLORS }; store.set('mti.demo.colors', palette); return demoApi.colors(); },
};
