import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { validateFile } from '../pages/ImageAnalysis';
import { ApiError, api } from '../services/api';
import { analysis, colors, hardware, health, idleTraining, mockApi, model, model2, systemInfo } from './mockApi';
import { renderApp } from './render';

const emptyOverview = {
  method: 'GET', path: '/overview',
  body: {
    dataset: { dataset_found: false, indexed: false, total_images: 0, split_counts: { train: 0, val: 0, test: 0 }, job: null },
    models: [], trained_model_count: 0, latest_experiment: null, training: { active: false, experiment_id: null, state: 'idle' },
    latest_analysis: null, recent_analyses: [], evaluations: [], hardware,
  },
};
const base = [health, colors, systemInfo, idleTraining];

describe('navigation', () => {
  it('renders all ten sections in the sidebar and routes between pages', async () => {
    mockApi([...base, emptyOverview, { path: '/models', body: { models: [] } }, { path: '/analyses', body: { analyses: [] } },
      { path: '/dataset/status', body: { indexed: false } }]);
    renderApp('/');
    const nav = screen.getByRole('navigation', { name: 'Main' });
    const labels = ['Overview', 'Image Analysis', 'Dataset Explorer', 'Model Training', 'Model Evaluation', 'Terrain Intelligence',
      'Navigation Simulator', 'Experiments', 'Documentation', 'Settings'];
    for (const l of labels) expect(within(nav).getByRole('link', { name: new RegExp(l) })).toBeInTheDocument();
    await userEvent.click(within(nav).getByRole('link', { name: /Documentation/ }));
    expect(await screen.findByRole('heading', { level: 1, name: 'Documentation' })).toBeInTheDocument();
    await userEvent.click(within(nav).getByRole('link', { name: /Image Analysis/ }));
    expect(await screen.findByRole('heading', { level: 1, name: 'Image Analysis' })).toBeInTheDocument();
  });

  it('shows a not-found page for unknown routes', async () => {
    mockApi(base);
    renderApp('/no-such-page');
    expect(await screen.findByText('Off the map')).toBeInTheDocument();
  });
});

describe('dashboard empty states', () => {
  it('explains how to begin when nothing is loaded or trained, without inventing numbers', async () => {
    mockApi([...base, emptyOverview]);
    renderApp('/');
    expect(await screen.findByText('No checkpoints yet')).toBeInTheDocument();
    expect(screen.getByText('No evaluations yet')).toBeInTheDocument();
    expect(screen.getByText('no test evaluation yet')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Load the dataset' })).toHaveAttribute('href', '/dataset');
    expect(screen.queryByText(/9\d(\.\d)?%/)).not.toBeInTheDocument(); // no fabricated high accuracy anywhere
  });

  it('shows real evaluation values from the API', async () => {
    mockApi([...base, { ...emptyOverview, body: { ...emptyOverview.body, models: [model], trained_model_count: 1,
      evaluations: [{ id: 1, experiment_id: 'unet-1', experiment_name: 'U-Net test run', arch: 'unet', split: 'test', label_variant: 'min3',
        images: 322, created_at: '2026-01-01T00:00:00', is_fixture: false, mean_iou: 0.4321, pixel_accuracy: 0.7, mean_dice: 0.5 }] } }]);
    renderApp('/');
    expect((await screen.findAllByText('43.2%')).length).toBeGreaterThan(0);
  });
});

describe('API error handling', () => {
  it('surfaces FastAPI error details', async () => {
    mockApi([{ path: '/training/status', status: 409, body: { detail: 'A training run is already in progress.' } },
      { path: '/training/estimate', method: 'POST', status: 422, body: { detail: [{ loc: ['body', 'epochs'], msg: 'must be >= 1' }] } }]);
    await expect(api.trainingStatus()).rejects.toThrow('A training run is already in progress.');
    await expect(api.trainingEstimate({} as never)).rejects.toThrow('epochs: must be >= 1');
    await expect(api.trainingStatus()).rejects.toBeInstanceOf(ApiError);
  });

  it('reports an unreachable backend and shows the offline banner', async () => {
    const { vi } = await import('vitest');
    vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('Failed to fetch'); }));
    await expect(api.health()).rejects.toThrow(/Cannot reach the backend/);
    renderApp('/');
    expect(await screen.findByText('Backend unreachable.')).toBeInTheDocument();
  });

  it('shows an error banner with retry when a page request fails', async () => {
    mockApi([...base, { path: '/overview', status: 500, body: { detail: 'Internal server error. See the backend log.' } }]);
    renderApp('/');
    expect(await screen.findByText('Internal server error. See the backend log.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Retry/ })).toBeInTheDocument();
  });
});

describe('image upload', () => {
  it('validates file type and size on the client', () => {
    expect(validateFile(new File(['x'], 'a.png', { type: 'image/png' }))).toBeNull();
    expect(validateFile(new File(['x'], 'a.JPEG', { type: 'image/jpeg' }))).toBeNull();
    expect(validateFile(new File(['x'], 'a.gif', { type: 'image/gif' }))).toMatch(/not a JPG/);
    expect(validateFile(new File([''], 'a.png', { type: 'image/png' }))).toMatch(/empty/);
    const big = new File([new Uint8Array(21 * 1024 * 1024)], 'big.png', { type: 'image/png' });
    expect(validateFile(big)).toMatch(/limit is 20 MB/);
  });

  it('shows an empty state when no model has been trained', async () => {
    mockApi([...base, { path: '/models', body: { models: [] } }, { path: '/analyses', body: { analyses: [] } }, { path: '/dataset/status', body: { indexed: false } }]);
    renderApp('/analysis');
    expect(await screen.findByText('No trained model yet')).toBeInTheDocument();
  });

  it('uploads an image with the selected model and renders the segmentation result', async () => {
    const { calls } = mockApi([...base, { path: '/models', body: { models: [model, model2] } }, { path: '/analyses', body: { analyses: [] } },
      { path: '/dataset/status', body: { indexed: false } }, { method: 'POST', path: '/predict', body: analysis },
      { path: '/analyses/abc123', body: analysis }]);
    renderApp('/analysis');
    const select = await screen.findByRole('combobox', { name: 'Model' });
    // model selection: defaults to the best validation model, and can be changed
    await waitFor(() => expect(select).toHaveValue('deeplab-1'));
    await userEvent.selectOptions(select, 'unet-1');
    expect(select).toHaveValue('unet-1');
    const run = screen.getByTestId('run-analysis');
    expect(run).toBeDisabled();
    const file = new File([new Uint8Array([137, 80, 78, 71])], 'mars.png', { type: 'image/png' });
    await userEvent.upload(screen.getByTestId('file-input'), file);
    expect(screen.getByText('mars.png')).toBeInTheDocument();
    expect(run).toBeEnabled();
    await userEvent.click(run);
    // result rendering: viewer, legend, statistics computed by the backend
    expect(await screen.findByTestId('segmentation-frame')).toBeInTheDocument();
    expect(screen.getAllByText('50.00%').length).toBeGreaterThan(0);
    expect(screen.getByText(/dominated by soil/)).toBeInTheDocument();
    expect(screen.getAllByText('Big rock').length).toBeGreaterThan(0);
    const post = calls.find((c) => c.method === 'POST' && c.path === '/api/predict');
    expect(post?.body).toBeInstanceOf(FormData);
    expect((post!.body as FormData).get('model_id')).toBe('unet-1');
    // view modes switch the rendered image
    await userEvent.click(screen.getByRole('tab', { name: 'Mask' }));
    expect(screen.getByAltText('mask view').getAttribute('src')).toContain('/api/analyses/abc123/image/mask');
  });

  it('rejects an invalid file before uploading', async () => {
    const { calls } = mockApi([...base, { path: '/models', body: { models: [model] } }, { path: '/analyses', body: { analyses: [] } },
      { path: '/dataset/status', body: { indexed: false } }]);
    renderApp('/analysis');
    await screen.findByRole('combobox', { name: 'Model' });
    fireEvent.change(screen.getByTestId('file-input'), { target: { files: [new File(['x'], 'notes.txt', { type: 'text/plain' })] } });
    expect(await screen.findByText(/is not a JPG, JPEG or PNG/)).toBeInTheDocument();
    expect(screen.getByTestId('run-analysis')).toBeDisabled();
    expect(calls.some((c) => c.path === '/api/predict')).toBe(false);
  });

  it('labels test-fixture output as not a real prediction', async () => {
    mockApi([...base, { path: '/models', body: { models: [{ ...model, is_fixture: true, name: 'TEST FIXTURE' }] } }, { path: '/analyses', body: { analyses: [] } },
      { path: '/dataset/status', body: { indexed: false } }, { path: '/analyses/abc123', body: { ...analysis, is_fixture: true } }]);
    renderApp('/analysis?id=abc123');
    expect(await screen.findByText('Test fixture output')).toBeInTheDocument();
    expect(screen.getAllByText(/not a real prediction/i).length).toBeGreaterThan(0);
  });
});

describe('training controls', () => {
  const defaults = {
    path: '/training/defaults',
    body: {
      config: { arch: 'unet', model_config: {}, epochs: 8, batch_size: 8, learning_rate: 0.001, weight_decay: 0.0001, resolution: 192, device: 'auto',
        max_train_samples: 1000, max_val_samples: 200, dice_weight: 0.5, class_weighting: 'none', scheduler: 'cosine', early_stopping_patience: 5,
        augment: true, seed: 42, num_workers: 0 },
      reason: 'No GPU detected', cuda_available: false, train_available: 3570, val_available: 402,
      options: { architectures: [], resolutions: [128, 192, 256], devices: ['auto', 'cpu'] },
    },
  };
  const estimate = { device: 'cpu', parameters: 1812068, train_samples: 1000, val_samples: 200, train_samples_available: 3570, val_samples_available: 402,
    batches_per_epoch: 125, total_steps: 1000, pixels_per_epoch: 1, seconds_per_step: 0.5, estimated_seconds_per_epoch: 70, estimated_total_seconds: 560,
    large_job: false, fits_in_memory: true, error: null, note: 'Measured' };

  it('requires a measured estimate before starting, then starts with the chosen configuration', async () => {
    const { calls } = mockApi([...base, defaults, { method: 'POST', path: '/training/estimate', body: estimate },
      { method: 'POST', path: '/training/start', body: { experiment_id: 'deeplabv3-x', estimate } }]);
    renderApp('/training');
    const start = await screen.findByTestId('start-training');
    expect(start).toBeDisabled();
    await userEvent.click(screen.getByRole('button', { name: /DeepLabV3/ }));
    await userEvent.click(screen.getByRole('button', { name: /Estimate workload/ }));
    expect(await screen.findByText('9m 20s')).toBeInTheDocument();
    expect(start).toBeEnabled();
    await userEvent.click(start);
    await waitFor(() => expect(calls.some((c) => c.path === '/api/training/start')).toBe(true));
    const body = JSON.parse(String(calls.find((c) => c.path === '/api/training/start')!.body));
    expect(body.arch).toBe('deeplabv3');
    expect(body.model_options).toEqual({ backbone: 'mobilenet_v3_large', pretrained: 'coco' });
    expect(body.max_train_samples).toBe(1000);
  });

  it('asks for explicit confirmation of a large job', async () => {
    mockApi([...base, defaults, { method: 'POST', path: '/training/estimate', body: { ...estimate, large_job: true, estimated_total_seconds: 4 * 3600 } }]);
    renderApp('/training');
    await userEvent.click(await screen.findByRole('button', { name: /Estimate workload/ }));
    const start = screen.getByTestId('start-training');
    await screen.findByText(/I understand this run is estimated/);
    expect(start).toBeDisabled();
    await userEvent.click(screen.getByRole('checkbox', { name: /I understand/ }));
    expect(start).toBeEnabled();
  });

  it('shows live progress and sends pause and stop commands for an active run', async () => {
    const running = { active: true, experiment_id: 'unet-1', name: 'Live run', arch: 'unet', state: 'running', message: 'Epoch 2/5, batch 3/10',
      epoch: 2, epochs_total: 5, batch: 3, batches_per_epoch: 10, progress: 0.26, best_val_miou: 0.41, best_epoch: 1, elapsed_seconds: 100,
      eta_seconds: 280, device: 'cpu', parameters: 1812068, has_checkpoint: true,
      history: [{ epoch: 1, train_loss: 1.2, val_loss: 1.1, val_pixel_accuracy: 0.6, val_mean_iou: 0.41, val_mean_dice: 0.5,
        val_per_class_iou: { soil: 0.5, bedrock: 0.4, sand: 0.3, big_rock: 0 }, learning_rate: 0.001, epoch_seconds: 50 }] };
    const { calls } = mockApi([health, colors, systemInfo, defaults, { path: '/training/status', body: running },
      { method: 'POST', path: '/training/pause', body: { ok: true } }, { method: 'POST', path: '/training/stop', body: { ok: true } },
      { path: /\/api\/training\/logs/, body: { experiment_id: 'unet-1', lines: ['Epoch 1/5 train_loss 1.2 [best]'] } }]);
    renderApp('/training');
    expect(await screen.findByText('Epoch 2/5, batch 3/10')).toBeInTheDocument();
    expect(screen.getByText('41.00%')).toBeInTheDocument();
    expect(await screen.findByText(/Epoch 1\/5 train_loss/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /Pause/ }));
    await userEvent.click(screen.getByTestId('stop-training'));
    await waitFor(() => {
      expect(calls.some((c) => c.method === 'POST' && c.path === '/api/training/pause')).toBe(true);
      expect(calls.some((c) => c.method === 'POST' && c.path === '/api/training/stop')).toBe(true);
    });
    expect(screen.getByTestId('start-training')).toBeDisabled(); // no second run while one is active
  });
});
