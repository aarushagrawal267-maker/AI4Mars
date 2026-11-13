# Testing

## Python tests (pytest)

Run from the repository root:

```powershell
.venv\Scripts\python -m pytest            # Windows
```

```bash
.venv/bin/python -m pytest                # macOS / Linux
```

Configuration is in `pytest.ini` (`testpaths = tests`, quiet output). The suite needs no internet
connection and no downloaded dataset: it builds a small **synthetic** dataset in a temporary folder
(`tests/conftest.py`), laid out exactly like the real AI4Mars v0.6 archive (`msl/ncam/images/edr`,
`labels/train`, `labels/test/masked-gold-min1|min3-100agree`, `label_keys.json`). Its tiny generated
images are never presented as real data. API tests run against an isolated app instance with a
temporary data, checkpoint and experiments directory. 68 tests currently pass.

The default application in `backend/app/main.py` is created lazily (only when uvicorn asks for
`backend.app.main:app`), so running the suite never opens your real `data/` folder or database.

### `tests/ml/`

| File | What it covers |
|---|---|
| `test_models.py` | Input/output shapes of U-Net and DeepLabV3 (4 classes, input resolution); U-Net configurability and skip connections; input sizes not divisible by the stride; rejection of unknown architectures and options; the loss ignores unlabelled pixels (value and gradient), an all-unlabelled batch gives a zero, differentiable loss, known loss values; inverse-sqrt class weights; checkpoint save/load round trip for both architectures; checkpoint safeguards (missing file, unrecognised format, non-checkpoint bytes); device resolution; inference pipeline outputs (class map, confidence, uncertainty) |
| `test_metrics_and_data.py` | Confusion matrix ignores 255 and matches scikit-learn; metrics on known values, perfect predictions and absent classes (null, not 0); image preprocessing; augmentation preserves label values and shapes; upload validation (empty, corrupt, truncated, too small, GIF); colour palette; exact terrain percentages and exclusion of unlabelled pixels; majority-vote navigation grid; MSL file-name parsing; layout detection; index pairing, grouped splits and statistics; detection of dataset problems (label without image, unexpected class value, unreadable label, size mismatch, fully unlabelled label, image without label); index save/load and the preprocessing cache; training configuration validation; the training loop writes history and checkpoints; a stop request leaves a consistent state; failures are recorded |
| `test_remote_zip.py` | Range-request ZIP reading against a local HTTP server (no internet): listing, subset planning (other missions not fetched, test set and documentation included), byte-exact CRC-checked extraction, resuming without re-downloading; zip-slip protection in `safe_join`; local archive import extracts only the MSL folders |

### `tests/backend/`

| File | What it covers |
|---|---|
| `test_api.py` | Health, system information and OpenAPI documentation of the core endpoints; empty state before any data is loaded; dataset load, validation report and statistics; clear error when no dataset exists; sample listing and rendering; import refuses paths and unknown archives; invalid uploads (not an image, empty, wrong type, truncated, unknown model, over the size limit) are rejected with the right status codes; prediction response, output sizes, class-id mask and every render kind; prediction from a dataset sample and from a JPEG; display colours change renderings but not class ids; navigation planning endpoint; configurable, persisted navigation costs; evaluation calculations on a held-out split and input validation; evaluation requires a dataset; training request validation and the measured estimate; an end-to-end training run with database persistence; runs left active by a previous server are marked interrupted on restart |
| `test_navigation.py` | A* on a uniform grid; paths are connected and avoid blocked cells; no corner cutting between diagonal obstacles; unreachable goals and blocked start/goal; optimality against an independent Dijkstra implementation; terrain costs change the route; impassable classes, user-blocked cells and unknown terrain; the alternative route and the uncertainty penalty |

Inference and evaluation tests in `test_api.py` use the **untrained fixture model** (see below). They
check plumbing and output validity, not model quality, and assert that its results are flagged
`is_fixture: true`.

## Frontend tests

Run inside `frontend/`:

```bash
npm test          # 15 unit/component tests with Vitest (jsdom, mocked API)
npm run e2e       # 3 end-to-end browser tests with Playwright
```

The end-to-end test needs the Playwright Chromium browser once:

```bash
npx playwright install chromium   # or run setup.ps1 -WithE2E / ./setup.sh --with-e2e
```

The Playwright configuration starts its own backend with `MTI_ENABLE_TEST_FIXTURE=true` on a
temporary data directory (it does not touch your `data/` folder) together with the frontend. The
test uploads an image through the web interface and checks that a valid segmentation is returned.
When no trained checkpoint exists, it uses the clearly labelled **untrained fixture model**
("TEST FIXTURE - untrained U-Net (not a real model)"). The fixture exists only when
`MTI_ENABLE_TEST_FIXTURE=true`, its output is meaningless by design, and it is flagged `is_fixture`
in every API response, so its output is never presented as a real prediction.

The Vitest suite (`src/test/app.test.tsx`) covers navigation between all ten pages, the not-found
page, dashboard empty states (and that no accuracy figure appears when none exists), real values from
the API, API error handling (FastAPI error details, unreachable backend banner, retry), client-side
upload validation, the upload-to-result flow with model selection and view modes, fixture labelling,
training controls (estimate required before start, large-job confirmation, pause/stop, live progress).

The Playwright suite (`e2e/segmentation.spec.ts`) uses its own ports — backend **8001**, frontend
**5174** — and a temporary data folder, so it runs safely next to an app started with `start.ps1`.
It checks: uploading a generated PNG returns a valid segmentation (dimensions, percentages summing to
100, class-id mask) that is labelled as fixture output; the mask renders; the route planner works on
the result; invalid uploads are rejected; every page loads without JavaScript errors. If Chromium is
installed elsewhere, set `PW_CHROMIUM_PATH` to its executable.
