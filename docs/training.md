# Training

Code: `ml/training/` (`config.py`, `losses.py`, `estimate.py`, `trainer.py`, `run.py`), started and
monitored by `backend/app/services/training_manager.py`. Each run executes in its own process and
communicates through files (see [architecture.md](architecture.md#training-as-a-separate-process)).

## Configuration

One validated `TrainConfig` per experiment, stored with the experiment and inside every checkpoint.

| Field | Allowed values | Default |
|---|---|---|
| `arch` | `unet`, `deeplabv3` | `unet` |
| `model_config` | see [models.md](models.md) | architecture defaults |
| `epochs` | 1-500 | 10 |
| `batch_size` | 1-128 (>= 2 for DeepLabV3) | 8 |
| `learning_rate` | 1e-6 to 1 | 1e-3 |
| `weight_decay` | 0-1 | 1e-4 |
| `resolution` | 128, 160, 192, 224, 256, 320, 384, 448, 512 | 256 |
| `device` | `auto`, `cpu`, `cuda` | `auto` |
| `max_train_samples` / `max_val_samples` | >= 2, or null for the full split | null |
| `dice_weight` | 0-5 (0 disables Dice) | 0.5 |
| `class_weighting` | `none`, `inverse_sqrt` | `none` |
| `scheduler` | `cosine`, `plateau`, `none` | `cosine` |
| `early_stopping_patience` | 0-100 epochs (0 disables) | 5 |
| `augment` | true / false | true |
| `seed` | 0 to 2^31-1 | 42 |

### Hardware-aware suggestions

`GET /api/training/defaults` suggests a starting configuration; every value can be changed.

| | CPU (no CUDA GPU) | CUDA GPU |
|---|---|---|
| Training images | random subset of 1000 (or fewer if the split is smaller) | full training split |
| Validation images | subset of 200 | full validation split |
| Resolution | 192 px | 384 px |
| Batch size | 8 | 16 |
| Epochs | 8 | 30 |
| Learning rate | 1e-3 | 1e-3 |

The other fields keep the defaults above. The CPU suggestion is chosen so that a first run finishes
in minutes rather than hours; full-dataset training on a CPU takes many hours.

## Optimisation

- **Optimiser:** AdamW (Loshchilov and Hutter) with the configured learning rate and weight decay.
- **Learning-rate schedule** (stepped once per epoch):
  - `cosine`: cosine annealing over the planned number of epochs down to 1 % of the initial rate;
  - `plateau`: halve the rate when validation mIoU has not improved for 2 epochs;
  - `none`: constant rate.
- **Loss:** cross-entropy with `ignore_index=255`, so unlabelled pixels contribute nothing, plus
  `dice_weight` x soft Dice loss (Milletari et al., V-Net). The Dice term is computed over labelled
  pixels only and averaged over the classes present in the batch. A batch without any labelled pixel
  returns a zero loss that keeps the computation graph intact.
- **Optional class weighting** (`inverse_sqrt`): cross-entropy weights proportional to
  1 / sqrt(class frequency), computed from the pixel counts of the training images actually used,
  normalised to mean 1 and clipped to [0.25, 8]. It targets the imbalance described in
  [dataset.md](dataset.md#class-imbalance) (big rock is about 1 % of labelled pixels).
- **Mixed precision (AMP):** `torch.autocast` with a gradient scaler, enabled on CUDA only. CPU
  training runs in full float32.
- A non-finite loss (NaN/inf) stops the run with a message suggesting a lower learning rate.

## Data and augmentation

Training and validation images come from the preprocessing cache at the configured resolution
(see [dataset.md](dataset.md#preprocessing-cache)). When `max_train_samples` / `max_val_samples` are
set, a seeded random subset is used. The training loader shuffles with a seeded generator; the last
incomplete batch is dropped when there is more than one batch.

Augmentation (`augment_pair` in `ml/preprocessing/transforms.py`, training split only):

| Transform | Probability | Detail |
|---|---|---|
| Horizontal flip | 0.5 | image and label |
| Random scale-crop | 0.7 | crop 60-100 % of each side at a random position, resize back (image bilinear, label nearest-neighbour) |
| Brightness / contrast jitter | 0.8 | contrast x0.8-1.25 around mid-grey, brightness +-20 grey levels (image only) |
| Gaussian noise | 0.3 | sigma 2-6 grey levels (image only) |

Vertical flips are **not** used: in rover images the sky/horizon is at the top and near-field terrain
at the bottom, and that layout is meaningful. Label values are never interpolated, so 255 stays 255.

## Validation, model selection and early stopping

After every epoch the model is evaluated on the validation split (cached, at the training
resolution, without augmentation). Loss, pixel accuracy, mean IoU, mean Dice and per-class IoU are
appended to `history.jsonl`.

- `last.pt` is saved after every completed epoch.
- `best.pt` is saved whenever **validation mean IoU** improves. Inference and evaluation use
  `best.pt`.
- **Early stopping:** the run ends when validation mIoU has not improved for
  `early_stopping_patience` epochs (0 disables this).

Validation metrics during training are computed on resized labels at the training resolution.
The separate evaluation step (see [evaluation.md](evaluation.md)) works at native label resolution,
so its numbers can differ slightly from the training curves even on the same split.

## Reproducibility

`random`, NumPy and PyTorch (including all CUDA devices) are seeded with `seed`; the shuffling
generator, the augmentation random generator and the subset selection use the same seed, and data
loading runs in the main process. This is intended to make a CPU run repeatable on the same machine
and software versions (bit-for-bit equality has not been verified). cuDNN is not forced into
deterministic mode, so GPU runs can differ slightly from run to run.

## Workload estimate

Before a run starts, `POST /api/training/estimate` (also called by `start`) **measures** the cost
rather than guessing it (`ml/training/estimate.py`): it builds the requested model (without
downloading pretrained weights), runs one warm-up step and then times two real optimisation steps at
the requested batch size and resolution on the selected device, plus one validation forward pass.
The estimate is

```
seconds per epoch = training batches x seconds per step + validation batches x seconds per forward pass
total             = seconds per epoch x epochs
```

It reports parameters, batches per epoch, total steps and pixels per epoch. It excludes the one-off
dataset caching and data-loading overhead, and the run may end earlier through early stopping, so it
is labelled as an estimate. If the timing step runs out of memory, the configuration is reported as
not fitting and cannot be started.

**Large jobs:** a run estimated at more than 2 hours is refused unless the request sets
`confirm_large: true` (the UI asks for explicit confirmation).

## Pause, resume and stop

- **Pause** (`POST /api/training/pause`): the run stops between batches (also during validation)
  and its state becomes `paused` until resumed. Nothing is lost.
- **Resume** (`POST /api/training/resume`): continues from the same batch.
- **Stop** (`POST /api/training/stop`): the run ends after the current batch with state `stopped`.
  Checkpoints from completed epochs are kept; if no epoch had finished, there is no checkpoint and the
  message says so. A stop request is also honoured while the cache is being built. If the process has
  not exited 60 s after a stop request, the API terminates it.
- Closing the backend normally also stops an active run (it waits up to 20 s). A run that was still
  marked active when the server last stopped is shown as `interrupted`; its completed-epoch
  checkpoints remain usable.

Run states: `queued`, `preparing`, `running`, `paused`, `completed`, `stopped`, `failed`,
`interrupted`.

## Out-of-memory and failure handling

- Configurations that do not fit are usually caught by the estimate before starting.
- A CUDA or CPU out-of-memory error during training ends the run as `failed` with advice to reduce
  batch size or resolution, or to use a lighter model (U-Net `base_channels` 16, DeepLabV3
  MobileNetV3).
- Because training runs in a separate process, a crash or an operating-system out-of-memory kill does
  not take the web server down. If the process disappears without writing a final state, the API
  marks the run `failed` with the exit code and an explanation.
- Every failure is written to `status.json`, `train.log` and the `experiments` table.

## Fair comparison methodology

To compare U-Net and DeepLabV3 fairly, everything except the architecture is held fixed:

- **Same data:** the same dataset index, recorded by its fingerprint in every experiment and
  checkpoint, so the same train/validation/test split; the same training and validation subsets
  (same `max_*_samples` and seed).
- **Same input pipeline:** same resolution, grayscale replicated to 3 channels, ImageNet
  normalisation for both models, same augmentation.
- **Same optimisation:** batch size, number of epochs, optimiser (AdamW), learning rate, weight
  decay, schedule, loss (cross-entropy + Dice with the same weight, same class weighting), early
  stopping patience and seed.
- **Model selection on validation only:** the reported checkpoint is the epoch with the best
  validation mIoU. The test set is never used for training, model selection, early stopping or
  hyper-parameter choices; it is touched only for the final evaluation.
- **Report both test variants:** `min1` (denser, includes pixels labelled by one expert) and `min3`
  (sparsest, all three experts agree), together with per-class IoU, because mean values hide the
  big-rock class.
- **Report cost as well as accuracy:** parameter count, checkpoint size and measured inference time
  on the same device.

Caveats that remain:

- **Pretrained vs from scratch.** DeepLabV3 starts from COCO-trained weights while U-Net starts from
  random weights. This is part of the architecture comparison as practised (it is how DeepLabV3 is
  normally used) and is stated with the results, not a controlled variable. Check
  `pretrained_loaded`; if the download failed, the DeepLabV3 run was also from scratch.
- **Different capacity.** The models differ in size (for example U-Net base 16: 1.8 M parameters,
  DeepLabV3 MobileNetV3-Large: 11.0 M); see [models.md](models.md).
- **Limited training budget.** On a CPU, the number of epochs and the resolution are small; models
  may not have converged, and the ranking could change with longer training.
- **Single seed.** One run per architecture gives no estimate of run-to-run variance; small
  differences should not be over-interpreted.

## Baseline experiment

The baseline comparison is scripted in `scripts/run_baseline_experiments.py`. It drives the REST API
of a running backend (`http://localhost:8000`) with a loaded dataset and uses one shared
configuration:

| Setting | Value |
|---|---|
| Epochs | 10 (early stopping patience 4) |
| Batch size | 8 |
| Resolution | 192 px |
| Optimiser | AdamW, learning rate 1e-3, weight decay 1e-4, cosine schedule |
| Loss | cross-entropy (ignore 255) + 0.5 x Dice, no class weighting |
| Augmentation | on |
| Seed | 42 |
| Data | full training and validation splits of the loaded index (no `max_*_samples` limit) |
| Device | auto |

Runs:

1. U-Net, `base_channels` 16, `depth` 4 (bilinear up-sampling, no dropout);
2. DeepLabV3, MobileNetV3-Large backbone, COCO-pretrained initialisation.

After each run completes, the script evaluates `best.pt` on the test set with `min3` labels, on the
test set with `min1` labels, and on the validation split. It stops if a run does not complete.

```bash
# with the backend running and the dataset loaded
.venv/bin/python scripts/run_baseline_experiments.py      # Windows: .venv\Scripts\python scripts\run_baseline_experiments.py
```

### Results

See the README. Results are filled in from the stored evaluation reports
(`GET /api/evaluations/report?ids=...&format=csv|json`, or the Evaluation page) and are not
reproduced here.
