import { defineConfig, devices } from '@playwright/test';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * End-to-end tests run against an ISOLATED stack: a backend on :8001 with a temporary data directory and the
 * clearly labelled untrained test fixture enabled, plus a Vite dev server on :5174 proxying to it.
 * Your real data/, checkpoints/ and experiments/ are never touched.
 */
const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..');
const tmp = path.join(os.tmpdir(), 'mti-e2e');
const python = process.platform === 'win32' ? path.join(repo, '.venv', 'Scripts', 'python.exe') : path.join(repo, '.venv', 'bin', 'python');

export default defineConfig({
  testDir: './e2e',
  timeout: 90_000,
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: 'http://127.0.0.1:5174',
    trace: 'off',
    ...devices['Desktop Chrome'],
    viewport: { width: 1440, height: 1000 },
    launchOptions: process.env.PW_CHROMIUM_PATH ? { executablePath: process.env.PW_CHROMIUM_PATH } : {},
  },
  webServer: [
    {
      command: `"${python}" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8001`,
      cwd: repo,
      url: 'http://127.0.0.1:8001/api/health',
      timeout: 120_000,
      reuseExistingServer: false,
      env: {
        MTI_DATA_DIR: path.join(tmp, 'data'),
        MTI_CHECKPOINT_DIR: path.join(tmp, 'checkpoints'),
        MTI_EXPERIMENTS_DIR: path.join(tmp, 'experiments'),
        MTI_ENABLE_TEST_FIXTURE: 'true',
        MTI_CORS_ORIGINS: 'http://127.0.0.1:5174',
      },
    },
    {
      command: 'npx vite --host 127.0.0.1 --port 5174 --strictPort',
      url: 'http://127.0.0.1:5174',
      timeout: 120_000,
      reuseExistingServer: false,
      env: { VITE_API_PROXY_TARGET: 'http://127.0.0.1:8001' },
    },
  ],
});
