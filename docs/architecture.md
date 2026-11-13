# Architecture

Mars Terrain Intelligence is a single-user application that runs entirely on the local machine. It
has three parts:

| Part | Folder | Technology | Responsibility |
|---|---|---|---|
| ML pipeline | `ml/` | PyTorch, torchvision, NumPy, OpenCV, Pillow | Dataset acquisition, indexing and validation, preprocessing cache, models, training loop, evaluation metrics, inference |
| Backend | `backend/` | FastAPI, SQLAlchemy (SQLite), Uvicorn | REST API, background jobs, training process management, persistence, route-planning simulator |
| Frontend | `frontend/` | React, Vite, TypeScript, Tailwind, Recharts | Web interface; talks only to the REST API |

The backend imports the `ml` package directly. The frontend never touches files or models; every
number it shows comes from an API response. This is an independent student research project. It is
not a NASA system and is not connected to any rover.

## Repository layout

```
ml/
  classes.py            class ids (0 soil, 1 bedrock, 2 sand, 3 big rock, 255 unlabelled), display colours
  datasets/             sources.py (verified Zenodo record), remote_zip.py (HTTP range reads),
                        acquire.py (subset / full / import), layout.py (finds the folders),
                        naming.py (MSL file names -> site/drive), index.py (pairing, validation, split,
                        statistics), cache.py (resized memory-mapped arrays), torch_dataset.py
  preprocessing/        transforms.py (normalisation, augmentation, upload validation)
  architectures/        unet.py, deeplab.py, factory.py (model building, checkpoint format)
  training/             config.py, losses.py, estimate.py, trainer.py, run.py (process entry point)
  evaluation/           metrics.py (confusion matrix based), evaluate.py
  inference/            predictor.py, terrain.py (statistics, navigation grid), visualize.py
backend/app/
  main.py               application factory, CORS, error mapping, router registration
  core/config.py        settings (MTI_* environment variables / .env)
  database/             SQLAlchemy engine and models
  api/                  one router per area (system, dataset, training, experiments, predict, evaluate, navigation)
  services/             dataset, training manager, job runner, model registry, inference, evaluation, settings
  navigation/           astar.py (A* and helpers), planner.py (route comparison)
scripts/                download_dataset.py, run_baseline_experiments.py
tests/                  pytest suite (ml/ and backend/)
data/ checkpoints/ experiments/   local, git-ignored runtime folders
```

## Data flow

```
                Zenodo record 15995036  /  local ZIP in data/raw/imports
                               |
            acquire.py (range-request subset | full download + MD5 | local extract)
                               v
   data/raw/<archive root>/msl/ncam/{images/edr, labels/train, labels/test/...}
                               |
          index.py: pair images and labels, validate every label, grouped train/val split
                               v
                 data/processed/index.json  (records, statistics, validation report, fingerprint)
                 /                         \
   cache.py: resize once to N px          evaluate.py / predict: read native-resolution files
   data/processed/cache_<N>_<fingerprint>/            |
                 |                                    v
   trainer.py (separate process)       evaluations table (metrics JSON) / analyses table + PNGs
                 |
   checkpoints/<id>/best.pt, last.pt  -->  model registry  -->  inference, evaluation
```

1. **Acquisition** writes the original archive layout under `data/raw/`.
2. **Indexing** (`POST /api/dataset/load`, also run automatically after a download or import)
   scans the folders, validates every label and writes `data/processed/index.json`. Everything the
   Dataset Explorer shows comes from this file.
3. **Training** reads the index, builds (or reuses) a resolution-specific cache, and writes
   checkpoints and per-epoch history.
4. **Evaluation** runs a checkpoint over the validation split or the expert test set at native label
   resolution and stores the metrics in SQLite.
5. **Inference** segments an uploaded image or a dataset sample, stores the outputs under
   `data/analyses/<id>/` and the statistics in SQLite.
6. **Navigation** downsamples a stored analysis to a grid and runs A* on it.

## Training as a separate process

A training run never runs inside the web server process. `TrainingManager.start`
(`backend/app/services/training_manager.py`) writes a job description and launches

```
python -m ml.training.run --run-dir experiments/<id>
```

with the repository root as working directory. The API stays responsive during training, and if the
run crashes or the operating system kills it for using too much memory, the server keeps running.
Only one training run can be active at a time.

The two processes communicate only through files:

| File | Written by | Content |
|---|---|---|
| `experiments/<id>/job.json` | API | Training configuration, data locations, the workload estimate |
| `experiments/<id>/control.json` | API | `{"command": "run" \| "pause" \| "stop"}` |
| `experiments/<id>/status.json` | trainer | State, epoch/batch progress, running loss, ETA, device, best validation mIoU; rewritten atomically (temporary file, then `os.replace`) |
| `experiments/<id>/history.jsonl` | trainer | One JSON line per completed epoch: losses, validation pixel accuracy / mIoU / mean Dice / per-class IoU, learning rate, epoch time |
| `experiments/<id>/train.log` | trainer | Human-readable log |
| `experiments/<id>/process.log` | API (stdout/stderr of the process) | Fallback log, e.g. for import errors |
| `checkpoints/<id>/last.pt` | trainer | Weights after every completed epoch |
| `checkpoints/<id>/best.pt` | trainer | Weights of the epoch with the best validation mIoU |

Checkpoints are written to `<name>.tmp` and then renamed with `os.replace`, so a stop, crash or
forced kill can never leave a half-written or deleted checkpoint from an epoch that already finished.

The trainer polls `control.json` between batches (at most every 0.4 s). `pause` blocks the loop
between batches; `stop` raises an internal exception that ends the run cleanly with state `stopped`.
If a run does not stop within 60 s of a stop request, the API terminates the process.

A watcher thread in the API copies `status.json` and `history.jsonl` into the `experiments` table
once per second, so experiment history survives restarts. If the process exits without writing a
final state (hard crash, OS out-of-memory kill), the watcher marks the run `failed` with an
explanation. On startup, runs still marked active from a previous server session are marked
`interrupted` (their completed-epoch checkpoints remain usable). When the server shuts down normally,
it asks an active run to stop and waits up to 20 s.

## Background job runner

Dataset scans, downloads, imports and evaluations run on daemon threads managed by
`backend/app/services/jobs.py`. Each job has a kind, status (`queued`, `running`, `completed`,
`failed`, `cancelled`), progress fraction, stage, message, error and result. Only one job of each
kind can be active at once. Cancellation sets an event that the long-running loops check (download
groups, extraction, evaluation images). Jobs are kept in memory only; their *results* (index file,
evaluation rows) are persisted. Training is not a job: it uses the separate process described above.

## Database

SQLite at `data/mti.sqlite3` by default (`MTI_DATABASE_URL` overrides it), opened in WAL mode with
foreign keys enabled. Tables are created automatically (`Base.metadata.create_all`).

| Table | Content |
|---|---|
| `experiments` | One row per training run: id, name, architecture, status, full configuration, per-epoch history, timestamps, epochs completed, best validation mIoU and epoch, device, parameter count, train/val sizes, elapsed time, `pretrained_loaded`, dataset fingerprint, message/error, user notes, `is_fixture` |
| `evaluations` | One row per evaluation: experiment id (cascade delete), split, label variant, number of images, the complete metrics JSON |
| `analyses` | One row per image analysis: id, model, source (`upload` or `dataset`), file name, sample id, `is_fixture`, terrain statistics, run metadata. The images themselves are PNG files in `data/analyses/<id>/` |
| `settings` | Key/value JSON: display colours (`colors`) and navigation costs (`nav_costs`) |

## REST API

All routes are under `/api`. Interactive documentation: `http://127.0.0.1:8000/docs` (Swagger UI)
and `/redoc`. Errors are JSON `{"detail": "..."}`; unknown ids give 404, invalid values 422,
conflicts (for example a second training run) 409.

### System

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Liveness check |
| GET | `/api/system/info` | Python/PyTorch versions, hardware (CUDA, GPUs, CPU cores, RAM), storage paths, free disk space, limits, record counts |
| GET | `/api/overview` | Aggregated state for the Overview page (dataset, models, latest experiment, training, recent analyses and evaluations, hardware) |

### Dataset

| Method | Path | Description |
|---|---|---|
| GET | `/api/dataset/status` | What is on disk, whether it is indexed, split counts, running job, archives in `data/raw/imports`, source record |
| POST | `/api/dataset/load` | Scan, validate and index the dataset folder (background job) |
| POST | `/api/dataset/download` | Download from Zenodo: `mode` `subset` (with `train_count`, 50-20000 or null for all) or `full` (requires `confirm_full`) |
| POST | `/api/dataset/import` | Extract a ZIP from `data/raw/imports`, by bare file name, then index it |
| POST | `/api/dataset/cancel` | Cancel the running dataset job |
| GET | `/api/dataset/statistics` | Per-split counts, class distribution, unlabelled percentage, split configuration, leakage check |
| GET | `/api/dataset/validation` | Validation report (missing pairs, unreadable files, size mismatches, unexpected label values) |
| GET | `/api/dataset/structure` | Folder tree of `data/raw` with file counts |
| GET | `/api/dataset/samples` | Page through indexed pairs (`split`, `offset`, `limit` up to 60, `contains` class) |
| GET | `/api/dataset/samples/{sample_id}` | Metadata for one sample |
| GET | `/api/dataset/samples/{sample_id}/{kind}` | PNG rendering: `image`, `label` or `overlay` (optional `variant`, `size`, `alpha`) |

### Training

| Method | Path | Description |
|---|---|---|
| GET | `/api/training/defaults` | Hardware-aware suggested configuration and allowed option values |
| POST | `/api/training/estimate` | Measured workload estimate for a configuration; nothing is started |
| POST | `/api/training/start` | Start a run in a separate process (`confirm_large` needed for runs estimated above 2 hours) |
| POST | `/api/training/stop` | Stop the active run after the current batch; completed checkpoints are kept |
| POST | `/api/training/pause` | Pause the active run between batches |
| POST | `/api/training/resume` | Resume a paused run |
| GET | `/api/training/status` | Progress, losses and validation metrics of the active (or most recent) run |
| GET | `/api/training/logs` | Tail of the training log (`experiment_id`, `tail` up to 2000 lines) |

### Experiments and models

| Method | Path | Description |
|---|---|---|
| GET | `/api/models` | Experiments that have a `best.pt` checkpoint, with their latest test evaluation |
| GET | `/api/experiments` | All experiments, newest first, including failed and stopped runs |
| GET | `/api/experiments/{exp_id}` | Configuration, per-epoch history and evaluations of one experiment |
| PATCH | `/api/experiments/{exp_id}` | Rename an experiment or edit its notes |
| DELETE | `/api/experiments/{exp_id}` | Delete an experiment with its checkpoints, run files and evaluations (refused while it is training) |
| GET | `/api/experiments/{exp_id}/report` | Download a JSON report, or `format=csv` for the epoch history |

### Inference

| Method | Path | Description |
|---|---|---|
| POST | `/api/predict` | Multipart form: `model_id` plus either `file` (JPG/PNG upload) or `sample_id` (dataset sample) |
| GET | `/api/analyses` | Recent analyses (`limit` up to 200) |
| GET | `/api/analyses/{analysis_id}` | Full statistics for one analysis |
| DELETE | `/api/analyses/{analysis_id}` | Delete an analysis and its stored images |
| GET | `/api/analyses/{analysis_id}/image/{kind}` | PNG: `original`, `mask`, `overlay`, `side_by_side`, `confidence`, `uncertainty`, `class_ids` (optional `alpha`, `download`) |
| GET | `/api/analyses/{analysis_id}/grid` | Prediction downsampled to a terrain grid by majority vote (`size` 8-160 cells on the long side) |

### Evaluation

| Method | Path | Description |
|---|---|---|
| POST | `/api/evaluate` | Evaluate a model on `test` (with `label_variant`, default `min3`) or `val`; optional `max_images` |
| GET | `/api/evaluate/status` | Progress of the running (or most recent) evaluation job |
| POST | `/api/evaluate/cancel` | Cancel the running evaluation |
| GET | `/api/evaluations` | Stored evaluations, newest first (optional `experiment_id`) |
| GET | `/api/evaluations/report` | Download evaluations `ids=1,2,...` as `format=json` or `csv` |
| GET | `/api/evaluations/{evaluation_id}` | One stored evaluation |
| DELETE | `/api/evaluations/{evaluation_id}` | Delete a stored evaluation |

### Navigation simulator and display settings

| Method | Path | Description |
|---|---|---|
| GET | `/api/navigation/costs` | Saved terrain movement costs, defaults and disclaimer |
| PUT | `/api/navigation/costs` | Save terrain costs (`cost` > 0 and <= 1000, `passable` per class) |
| POST | `/api/navigation/costs/reset` | Restore the default illustrative costs |
| POST | `/api/navigation/plan` | Plan routes with A* on the grid of a stored analysis |
| GET | `/api/settings/colors` | Display colour per class (class ids never change) |
| PUT | `/api/settings/colors` | Change display colours (`#RRGGBB`) |
| POST | `/api/settings/colors/reset` | Restore the default palette |

## Security choices

The application is designed for one user on their own machine. The start scripts bind both servers to
`127.0.0.1`, and there is no authentication, so it should not be exposed on a network as is. Within
that scope:

- **No user-supplied file paths.** Dataset import accepts only a bare file name (no `/`, `\`, `:`,
  `.` or `..`) that must also appear in the listing of ZIP files in `data/raw/imports`. Experiments,
  analyses and dataset samples are addressed by ids that are looked up in the database or the index
  before any path is built from them.
- **Zip-slip protection.** Every archive member, whether fetched remotely or extracted from a local
  ZIP, is joined onto the destination with `safe_join`, which refuses any path that resolves outside
  the destination folder.
- **Download integrity.** The remote archive's size is compared with the Zenodo record; every member
  fetched by range request is checked against the CRC-32 stored in the ZIP; the full download is
  checked against the record's MD5 before extraction.
- **Safe checkpoint loading.** Checkpoints are loaded with `torch.load(..., weights_only=True)`,
  which restricts unpickling to tensors and primitive containers, so a tampered file cannot execute
  code. The format version and class list are checked before the weights are used.
- **Upload validation.** Extension (`.jpg`, `.jpeg`, `.png`) and content type are checked, the size
  limit (`MTI_MAX_UPLOAD_MB`, default 20 MB) is enforced while reading, and the bytes must decode as
  a JPEG or PNG (Pillow `verify` detects truncated files). Images smaller than 32 x 32 or larger than
  40 megapixels are rejected. EXIF rotation is applied.
- **CORS** is limited to the origins in `MTI_CORS_ORIGINS`.
- **Test fixture off by default.** The untrained fixture model used by automated UI tests exists
  only when `MTI_ENABLE_TEST_FIXTURE=true`, and it is flagged `is_fixture` everywhere it appears.
- **Generic 500 responses.** Unexpected exceptions are logged on the server; the client receives a
  generic message without a stack trace.
