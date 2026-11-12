# Known limitations and future work

This is an independent student research project. It is not a NASA system, it is not connected to any
rover, and none of its outputs should be used for real operational decisions. The points below are
known limitations of the data, the models, the evaluation and the software.

## Data

- **One camera, one rover.** Models are trained only on Curiosity (MSL) NavCam images. Performance on
  MER or Perseverance images, MastCam images, colour photographs or non-Mars images has not been
  evaluated, and there is no out-of-distribution detection: any uploaded image receives a terrain
  segmentation, however unrelated it is.
- **Subset training.** On a laptop, experiments typically use a subset of the 16,064 MSL NavCam
  training labels (see [dataset.md](dataset.md) for the subset used here). Results on the full
  training set may differ.
- **Label quality.** Training labels are crowdsourced and merged with a 2/3 agreement rule; ambiguous
  pixels are unlabelled. The model is never supervised on the hardest regions.
- **No sky / rover / far-range class.** Labels mask terrain beyond about 30 m and the rover, and
  there is no class for sky. At inference every pixel still gets one of the four terrain classes.
  Observed with the baseline models: the rover's own hardware is often predicted as bedrock and
  distant ridges as big rock. Those pixels are meaningless and they inflate whole-image composition
  percentages. (Evaluation is unaffected, because those pixels are unlabelled in the ground truth.)
  Applying the dataset's rover masks (`images/mxy`) at inference is a possible improvement.
- **Class imbalance.** Big rock is about 1 % of labelled training pixels and well under 1 % of test
  pixels, so its scores are noisy and it is the class most likely to be missed.
- **Test-set overlap by position.** 31 of the 322 test images were taken at a rover position that
  also appears in the downloaded training subset. This is reported, not removed, to keep the official
  split; scores on the remaining images alone are not computed separately.

## Models and training

- **Limited compute.** CPU defaults use small resolutions (192 px), few epochs and a training subset.
  Models may be far from converged; architecture rankings could change with longer training or
  higher resolution.
- **Single seed.** Experiments are typically run once per configuration; there are no confidence
  intervals or significance tests, so small differences between models are not meaningful.
- **Pretrained vs from scratch.** DeepLabV3 uses COCO-pretrained weights by default while U-Net is
  trained from scratch; this difference is part of the comparison, not controlled for.
- **No hyper-parameter search.** Learning rate, loss weights and augmentation are fixed defaults,
  not tuned per architecture.
- **Training-time validation at reduced resolution.** Validation curves are computed on resized
  labels; final evaluation is at native resolution, so the two can differ slightly.
- **No resume from a checkpoint.** A stopped or interrupted run keeps its checkpoints for inference
  and evaluation, but cannot continue training from `last.pt` (the optimiser state is not saved); a
  new run starts from the beginning.
- **One training run at a time**, and only one model forward pass at a time for inference and
  evaluation.
- **GPU runs are not bit-for-bit reproducible** (cuDNN is not forced into deterministic mode).
- **Apple Silicon GPUs (MPS) are not used**; macOS runs on the CPU.

## Evaluation and uncertainty

- **Uncalibrated confidence.** Softmax confidence and entropy are shown, but calibration has not been
  measured or corrected, so they are not probabilities of being correct.
- **Single-model uncertainty only.** Uncertainty is the predictive entropy of one deterministic
  model; no ensembles, Monte Carlo dropout or other epistemic-uncertainty methods are used.
- **Timing depends on the machine.** Inference times are measured on the local device and are only
  comparable within one machine and session.

## Navigation simulator

- **Illustrative costs.** Terrain costs are assumptions, not values derived from rover data.
- **Terrain class is not traversability.** Slope, rock height, soil strength, rover state and
  mission rules are not modelled.
- **Image space, not ground space.** The grid follows image pixels in a perspective view; cell sizes
  on the ground vary strongly from the bottom to the top of the image, and no range or elevation data
  is used.
- **Built on predictions.** Segmentation errors propagate directly into the planned routes.

## Software

- **Single user, local only.** There is no authentication. The start scripts bind to `127.0.0.1`;
  exposing the servers on a network is not supported.
- **In-memory jobs.** Dataset and evaluation jobs are held in memory; if the backend restarts during
  one, it is lost (its partial files remain and a download resumes when restarted).
- **Cache growth.** Preprocessing caches (`data/processed/cache_<resolution>_<fingerprint>/`) are
  kept for every resolution and dataset version used and are not cleaned up automatically; delete old
  folders by hand to reclaim disk space.
- **Forced stop.** `stop.ps1` uses `taskkill /F`; a training run killed that way loses its current
  epoch (completed-epoch checkpoints are kept).
- **Docker route.** The Docker images use the CPU build of PyTorch only.

## Future work

- Repeat experiments with several seeds and report mean and spread.
- Report test scores separately for test images that do not share a rover position with training.
- Train on the full MSL NavCam training set at higher resolution on a GPU.
- Measure calibration (reliability diagrams, expected calibration error) and try temperature scaling.
- Add epistemic uncertainty (deep ensembles or Monte Carlo dropout) and compare it with errors.
- Add the MER data (Spirit and Opportunity) from the same archive to study cross-rover
  generalisation.
- Use the range and rover masks at inference time to grey out pixels the labels never covered.
- Project the navigation grid onto the ground using camera models and range data, so that path
  lengths and costs refer to metres rather than pixels.
- Allow resuming training from `last.pt` by saving optimiser and scheduler state.
