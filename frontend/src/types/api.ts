/** Types mirroring the FastAPI responses. Only fields the UI uses are listed. */

export type ClassKey = 'soil' | 'bedrock' | 'sand' | 'big_rock';
export type PerClass<T> = Record<ClassKey, T>;

export interface Health {
  status: 'ok';
  app: string;
  version: string;
  time: number;
}

export interface Hardware {
  cuda_available: boolean;
  gpus: { index: number; name: string; memory_total_gb: number; memory_free_gb: number }[];
  selected_device: string;
  cpu: string;
  cpu_cores_logical: number | null;
  cpu_cores_physical: number | null;
  torch_threads: number;
  ram_total_gb: number;
  ram_available_gb: number;
  diagnosis: string;
}

export interface SystemInfo {
  app: string;
  version: string;
  python: string;
  platform: string;
  torch: string;
  torchvision: string;
  hardware: Hardware;
  paths: Record<string, string>;
  disk_free_gb: number;
  limits: { max_upload_mb: number; max_output_side: number };
  counts: { experiments: number; evaluations: number; analyses: number };
  test_fixture_enabled: boolean;
}

export interface Job {
  id: string;
  kind: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  stage: string;
  message: string;
  error: string | null;
  params: Record<string, unknown>;
  created_at: number;
  finished_at: number | null;
  result?: unknown;
}

export interface SplitCounts {
  train: number;
  val: number;
  test: number;
}

export interface DatasetStatus {
  raw_dir: string;
  imports_dir: string;
  dataset_found: boolean;
  layout: Record<string, unknown> | null;
  indexed: boolean;
  index_created: string | null;
  fingerprint: string | null;
  total_images: number;
  split_counts: SplitCounts;
  needs_index: boolean;
  job: Job | null;
  local_archives: { name: string; bytes: number }[];
  source: {
    name: string;
    record_url: string;
    doi: string;
    archive: string;
    archive_bytes: number;
    license: string;
    kaggle_mirror: string;
    subset: string;
  };
}

export interface SplitStats {
  images: number;
  pixels_total: number;
  pixels_labeled: number;
  unlabeled_percent: number | null;
  class_pixels: PerClass<number>;
  class_percent_of_labeled: PerClass<number | null>;
  images_containing_class: PerClass<number>;
}

export interface DatasetStatistics {
  created: string;
  fingerprint: string;
  classes: ClassKey[];
  ignore_index: number;
  split_config: { val_fraction: number; seed: number; grouping: string };
  statistics: {
    splits: { train: SplitStats; val: SplitStats; test: Record<string, SplitStats> };
    test_variants: string[];
    total_images: number;
    cameras: Record<string, number>;
    rover_positions: number;
  };
  leakage_check: {
    train_val_shared_positions: number;
    test_images_sharing_position_with_train: number;
    test_images: number;
    note: string;
  };
  layout: Record<string, unknown>;
}

export interface ValidationReport {
  created: string;
  scan_seconds: number;
  validation: {
    images_found: number;
    train_labels_found: number;
    test_images_with_labels: number;
    test_label_variants: Record<string, number>;
    valid_pairs: number;
    labels_without_image: string[];
    labels_without_image_count: number;
    images_without_label_count: number;
    unreadable: { file: string; error: string }[];
    unreadable_count: number;
    size_mismatch: { image: string; image_size: number[]; label_size: number[] }[];
    size_mismatch_count: number;
    unexpected_values: { file: string; values: number[] }[];
    unexpected_values_count: number;
    empty_labels_count: number;
    train_test_filename_overlap: string[];
    observed_label_values: number[];
  };
  leakage_check: DatasetStatistics['leakage_check'];
}

export interface TreeNode {
  path: string;
  depth: number;
  files: number;
  extensions: Record<string, number>;
  bytes: number;
}

export interface Sample {
  id: string;
  split: 'train' | 'val' | 'test';
  width: number;
  height: number;
  camera: string;
  site: string;
  drive: string;
  sclk: number;
  variants: string[];
  labels: Record<string, { class_percent: PerClass<number>; unlabeled_percent: number }>;
}

export interface SamplePage {
  total: number;
  offset: number;
  limit: number;
  items: Sample[];
}

export interface TrainConfig {
  arch: 'unet' | 'deeplabv3';
  model_config: Record<string, unknown>;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  weight_decay: number;
  resolution: number;
  device: 'auto' | 'cpu' | 'cuda';
  max_train_samples: number | null;
  max_val_samples: number | null;
  dice_weight: number;
  class_weighting: 'none' | 'inverse_sqrt';
  scheduler: 'cosine' | 'plateau' | 'none';
  early_stopping_patience: number;
  augment: boolean;
  seed: number;
  num_workers?: number;
}

export interface TrainingDefaults {
  config: TrainConfig;
  reason: string;
  cuda_available: boolean;
  train_available: number;
  val_available: number;
  options: {
    architectures: { id: string; label: string; model_options: Record<string, unknown[]> }[];
    resolutions: number[];
    devices: string[];
  };
}

export interface Estimate {
  device: string;
  parameters: number;
  train_samples: number;
  val_samples: number;
  train_samples_available: number;
  val_samples_available: number;
  batches_per_epoch: number;
  total_steps: number;
  pixels_per_epoch: number;
  seconds_per_step: number | null;
  estimated_seconds_per_epoch: number | null;
  estimated_total_seconds: number | null;
  large_job: boolean;
  fits_in_memory: boolean;
  error: string | null;
  note: string;
}

export interface HistoryRow {
  epoch: number;
  train_loss: number;
  val_loss: number;
  val_pixel_accuracy: number | null;
  val_mean_iou: number | null;
  val_mean_dice: number | null;
  val_per_class_iou: PerClass<number | null>;
  learning_rate: number;
  epoch_seconds: number;
}

export type RunState = 'idle' | 'queued' | 'preparing' | 'running' | 'paused' | 'completed' | 'stopped' | 'failed'
  | 'interrupted' | 'fixture';

export interface TrainingStatus {
  active: boolean;
  experiment_id: string | null;
  name?: string;
  arch?: string;
  config?: TrainConfig;
  state: RunState;
  message?: string;
  epoch?: number;
  epochs_total?: number;
  batch?: number | null;
  batches_per_epoch?: number | null;
  progress?: number;
  train_loss_running?: number | null;
  best_val_miou?: number | null;
  best_epoch?: number | null;
  elapsed_seconds?: number | null;
  eta_seconds?: number | null;
  device?: string;
  parameters?: number;
  n_train?: number;
  n_val?: number;
  pretrained_loaded?: boolean | null;
  has_checkpoint?: boolean;
  error?: string | null;
  history?: HistoryRow[];
}

export interface ModelInfo {
  id: string;
  name: string;
  arch: string;
  status: string;
  is_fixture: boolean;
  created_at: string;
  resolution: number;
  model_config: Record<string, unknown>;
  epochs_completed: number;
  best_epoch: number | null;
  best_val_miou: number | null;
  parameters: number | null;
  n_train: number | null;
  checkpoint_bytes: number;
  pretrained_loaded: boolean | null;
  latest_test_evaluation: {
    id: number;
    label_variant: string;
    images: number;
    mean_iou: number | null;
    pixel_accuracy: number | null;
    created_at: string;
  } | null;
}

export interface PerClassMetric {
  iou: number | null;
  dice: number | null;
  precision: number | null;
  recall: number | null;
  support_pixels: number;
  predicted_pixels: number;
}

export interface Metrics {
  pixel_accuracy: number | null;
  mean_iou: number | null;
  mean_dice: number | null;
  mean_precision: number | null;
  mean_recall: number | null;
  per_class: PerClass<PerClassMetric>;
  confusion_matrix: number[][];
  evaluated_pixels: number;
  classes_present: ClassKey[];
  split: string;
  label_variant: string;
  images: number;
  resolution: number;
  inference: {
    device: string;
    forward_ms_mean: number;
    forward_ms_median: number;
    forward_ms_p95: number;
    end_to_end_ms_mean: number;
    images_per_second: number | null;
    timed_images: number;
    note: string;
  };
  model: {
    experiment_id: string;
    name: string;
    arch: string;
    is_fixture: boolean;
    parameters: number;
    checkpoint_bytes: number;
    checkpoint_mb: number;
    model_config: Record<string, unknown>;
    checkpoint_epoch: number | null;
  };
  subset_of_split?: boolean;
}

export interface Evaluation {
  id: number;
  experiment_id: string;
  experiment_name: string | null;
  arch: string;
  is_fixture: boolean;
  created_at: string;
  split: string;
  label_variant: string;
  images: number;
  metrics: Metrics;
}

export interface Experiment {
  id: string;
  name: string;
  arch: string;
  status: RunState | string;
  is_fixture: boolean;
  created_at: string;
  finished_at: string | null;
  config: TrainConfig;
  epochs_completed: number;
  best_val_miou: number | null;
  best_epoch: number | null;
  device: string | null;
  parameters: number | null;
  n_train: number | null;
  n_val: number | null;
  elapsed_seconds: number | null;
  pretrained_loaded: boolean | null;
  dataset_fingerprint: string | null;
  message: string | null;
  error: string | null;
  notes: string;
  has_checkpoint: boolean;
  final_train_loss: number | null;
  final_val_loss: number | null;
  evaluation_count: number;
}

export interface ExperimentDetail extends Experiment {
  history: HistoryRow[];
  evaluations: Evaluation[];
}

export interface SpatialEntry {
  pixels: number;
  centroid: [number, number] | null;
  bbox: [number, number, number, number] | null;
  regions: number;
  largest_region_percent_of_class: number;
  largest_region_percent_of_image: number;
  zone_share_percent: Record<string, number>;
}

export interface TerrainStats {
  height: number;
  width: number;
  valid_pixels: number;
  unlabeled_pixels: number;
  class_pixels: PerClass<number>;
  class_percent: PerClass<number>;
  dominant_class: ClassKey | null;
  spatial: PerClass<SpatialEntry>;
  zone_composition: Record<string, { percent: PerClass<number>; dominant: ClassKey | null }>;
  band_profile: ({ band: number; from_top_percent: number } & PerClass<number>)[];
  confidence?: {
    mean: number | null;
    median: number | null;
    min: number | null;
    per_class_mean: PerClass<number | null>;
    histogram: { from: number; to: number; pixels: number; percent: number }[];
    low_confidence_percent: number;
    high_confidence_percent: number;
    calibrated: boolean;
    note: string;
  };
  uncertainty?: {
    mean_entropy: number | null;
    high_uncertainty_percent: number;
    per_class_mean_entropy: PerClass<number | null>;
    definition: string;
  };
  summary: string;
}

export interface AnalysisSummary {
  id: string;
  created_at: string;
  experiment_id: string;
  model_name: string;
  arch: string;
  source: 'upload' | 'dataset';
  filename: string;
  sample_id: string | null;
  is_fixture: boolean;
  meta: {
    input_size: number[];
    output_size: number[];
    original_size: number[];
    inference_ms: number;
    device: string;
    resolution: number;
    mean_probs: PerClass<number>;
    model_parameters: number | null;
    color_input: boolean;
  };
  dominant_class: ClassKey | null;
  class_percent: PerClass<number>;
  mean_confidence: number | null;
}

export interface Analysis extends AnalysisSummary {
  stats: TerrainStats;
}

export interface TerrainGrid {
  rows: number;
  cols: number;
  cells: number[][];
  uncertainty: number[][] | null;
}

export interface TerrainCost {
  cost: number;
  passable: boolean;
}

export interface CostsResponse {
  costs: PerClass<TerrainCost>;
  defaults: PerClass<TerrainCost>;
  disclaimer: string;
}

export interface Route {
  kind: 'terrain_aware' | 'shortest' | 'alternative';
  label: string;
  description: string;
  found: boolean;
  path: [number, number][];
  cost: number | null;
  length: number | null;
  steps: number;
  expanded_nodes: number;
  composition: PerClass<number> | null;
  mean_cost_per_unit: number | null;
}

export interface PlanResult {
  rows: number;
  cols: number;
  start: [number, number];
  goal: [number, number];
  found: boolean;
  reason: string | null;
  routes: Route[];
  blocked_cells: number;
  impassable_classes: ClassKey[];
  algorithm: string;
  units: string;
  disclaimer: string;
}

export interface ColorsResponse {
  colors: PerClass<string>;
  defaults: PerClass<string>;
  unlabeled: string;
  classes: { id: number; key: ClassKey; label: string }[];
  ignore_index: number;
}

export interface Overview {
  dataset: Pick<DatasetStatus, 'dataset_found' | 'indexed' | 'total_images' | 'split_counts' | 'job'>;
  models: ModelInfo[];
  trained_model_count: number;
  latest_experiment: {
    id: string;
    name: string;
    arch: string;
    status: string;
    created_at: string;
    epochs_completed: number;
    epochs_total: number;
    best_val_miou: number | null;
    message: string | null;
  } | null;
  training: TrainingStatus;
  latest_analysis: AnalysisSummary | null;
  recent_analyses: AnalysisSummary[];
  evaluations: {
    id: number;
    experiment_id: string;
    experiment_name: string | null;
    arch: string;
    split: string;
    label_variant: string;
    images: number;
    created_at: string;
    is_fixture: boolean;
    mean_iou: number | null;
    pixel_accuracy: number | null;
    mean_dice: number | null;
  }[];
  hardware: Hardware;
}
