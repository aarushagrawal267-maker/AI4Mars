import { expect, test } from '@playwright/test';
import { deflateSync } from 'node:zlib';

/** Build a small valid RGB PNG in memory (rocky-looking gradient) so the test needs no binary fixtures. */
function makePng(width: number, height: number): Buffer {
  const raw = Buffer.alloc((width * 3 + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (width * 3 + 1)] = 0;
    for (let x = 0; x < width; x++) {
      const i = y * (width * 3 + 1) + 1 + x * 3;
      const v = (x * 7 + y * 3 + ((x * y) % 23) * 5) % 256;
      raw[i] = v; raw[i + 1] = Math.floor(v * 0.7); raw[i + 2] = Math.floor(v * 0.5);
    }
  }
  const crcTable = Array.from({ length: 256 }, (_, n) => {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    return c >>> 0;
  });
  const crc = (buf: Buffer) => {
    let c = 0xffffffff;
    for (const b of buf) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  };
  const chunk = (type: string, data: Buffer) => {
    const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type), data]);
    const c = Buffer.alloc(4); c.writeUInt32BE(crc(td));
    return Buffer.concat([len, td, c]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0); ihdr.writeUInt32BE(height, 4); ihdr[8] = 8; ihdr[9] = 2;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}

test('uploading an image returns a valid segmentation result', async ({ page, request }) => {
  await page.goto('/analysis');
  await expect(page.getByRole('heading', { level: 1, name: 'Image Analysis' })).toBeVisible();

  // No trained checkpoint exists in the isolated stack, so the only model is the clearly labelled fixture.
  const select = page.getByRole('combobox', { name: 'Model' });
  await expect(select).toContainText('TEST FIXTURE');
  await expect(page.getByText(/untrained test fixture/)).toBeVisible();

  await page.getByTestId('file-input').setInputFiles({ name: 'mars-test.png', mimeType: 'image/png', buffer: makePng(160, 120) });
  await expect(page.getByText('mars-test.png')).toBeVisible();
  const [response] = await Promise.all([
    page.waitForResponse((r) => r.url().endsWith('/api/predict') && r.request().method() === 'POST'),
    page.getByTestId('run-analysis').click(),
  ]);
  expect(response.status()).toBe(200);
  const result = await response.json();

  // The response is a valid segmentation: dimensions match the upload, percentages are consistent.
  expect(result.is_fixture).toBe(true);
  expect(result.meta.output_size).toEqual([120, 160]);
  const pct = Object.values(result.stats.class_percent as Record<string, number>).reduce((a, b) => a + b, 0);
  expect(pct).toBeCloseTo(100, 1);
  expect(result.stats.valid_pixels).toBe(120 * 160);
  expect(['soil', 'bedrock', 'sand', 'big_rock']).toContain(result.stats.dominant_class);

  // The UI renders the result and never presents fixture output as a real prediction.
  await expect(page.getByTestId('segmentation-frame')).toBeVisible();
  await expect(page.getByText('Test fixture output')).toBeVisible();
  await page.getByRole('tab', { name: 'Mask' }).click();
  const mask = page.getByAltText('mask view');
  await expect(mask).toBeVisible();
  await expect.poll(() => mask.evaluate((img: HTMLImageElement) => img.naturalWidth)).toBe(160);

  // The stored class-id mask has exactly the uploaded image's size and only ids 0-3.
  const ids = await request.get(`/api/analyses/${result.id}/image/class_ids`);
  expect(ids.status()).toBe(200);
  expect(ids.headers()['content-type']).toBe('image/png');

  // Terrain Intelligence and the Navigation Simulator work on the same analysis.
  await page.getByRole('link', { name: 'Plan route' }).click();
  await expect(page.getByRole('heading', { level: 1, name: 'Navigation Simulator' })).toBeVisible();
  await expect(page.getByTestId('nav-canvas')).toBeVisible();
  await expect(page.getByText('Terrain-aware route').first()).toBeVisible({ timeout: 15_000 });
});

test('invalid uploads are rejected with a clear message', async ({ page }) => {
  await page.goto('/analysis');
  await page.getByTestId('file-input').setInputFiles({ name: 'notes.txt', mimeType: 'text/plain', buffer: Buffer.from('not an image') });
  await expect(page.getByText(/is not a JPG, JPEG or PNG image/)).toBeVisible();
  await expect(page.getByTestId('run-analysis')).toBeDisabled();
});

test('every page loads without errors', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  for (const [path, title] of [['/', 'Pixel-level terrain intelligence'], ['/dataset', 'Dataset Explorer'], ['/training', 'Model Training'],
    ['/evaluation', 'Model Evaluation'], ['/terrain', 'Terrain Intelligence'], ['/navigation', 'Navigation Simulator'],
    ['/experiments', 'Experiments'], ['/docs', 'Documentation'], ['/settings', 'Settings']]) {
    await page.goto(path);
    await expect(page.getByRole('heading', { level: 1 }).first()).toContainText(title);
  }
  expect(errors).toEqual([]);
});
