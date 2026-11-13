# Evaluation

Code: `ml/evaluation/metrics.py`, `ml/evaluation/evaluate.py`, run in the background by
`backend/app/services/evaluation_service.py`. Results are stored in the `evaluations` table and can
be downloaded as CSV or JSON.

## What is evaluated

`POST /api/evaluate` with a `model_id` (an experiment with `best.pt`) and a split:

- `test`: the official expert-labelled test set with one label variant, `min1`, `min2` or `min3`
  (default `min3`);
- `val`: the validation split with its merged crowdsourced labels.

Evaluating on the training split is refused. An optional `max_images` evaluates only the first N
images; such results are flagged `subset_of_split: true`.

## Procedure

For every image (`evaluate_model`):

1. Read the original image and label files (not the training cache).
2. Prepare the input as for inference: long side scaled to the model's training resolution, sides
   rounded to a multiple of 32, grayscale replicated to 3 channels, ImageNet normalisation.
3. Forward pass; softmax; the **probabilities are up-sampled bilinearly to the label's native
   resolution** (1024 x 1024 for MSL NavCam), and the class with the highest probability is taken.
4. Accumulate a 4 x 4 confusion matrix against the original, un-resized label.

Metrics are computed once, from the confusion matrix summed over all images (pixel-level micro
aggregation over the whole split, not an average of per-image scores).

## Metrics

Rows of the confusion matrix are ground truth, columns are predictions. **Pixels whose ground truth is
255 (unlabelled) are excluded** before anything is counted; predictions on those pixels do not
affect any metric.

With TP, FP, FN per class c taken from the confusion matrix:

| Metric | Definition |
|---|---|
| Pixel accuracy | sum of the diagonal / all labelled pixels |
| IoU (Jaccard) | TP / (TP + FP + FN) |
| Dice (F1) | 2 TP / (2 TP + FP + FN) |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| Mean IoU / Dice / precision / recall | unweighted mean over the classes **present in the ground truth** |

Also reported per class: support (labelled pixels of that class) and predicted pixels; overall: the
full confusion matrix, the number of evaluated pixels and the list of classes present.

**Undefined values are `null`, not 0.** When a denominator is zero (for example precision of a class
the model never predicts, or IoU of a class absent from both ground truth and prediction), the value
is reported as `null`. Means skip `null` values and classes that do not occur in the ground truth,
because a class the evaluation set does not contain cannot be scored. A class that occurs in the
ground truth but is never predicted gets IoU 0 and counts in the mean.

Because the mean weights classes equally, the rare *big rock* class has as much influence on mean IoU
as *soil*, although it covers well under 1 % of the test pixels. Read the per-class values alongside
the means.

## Timing and model size

- **Forward-pass time:** measured for each image around the model call only (with CUDA
  synchronisation on a GPU), at the model's input resolution, batch size 1. The **first image is
  excluded** as warm-up. Reported: mean, median and 95th percentile in milliseconds, images per second
  (1000 / mean), the number of timed images and the device.
- **End-to-end time:** mean time including preprocessing and up-sampling to native resolution.
- **Model size:** number of parameters (all tensors of the model) and checkpoint file size in bytes
  (and MB).

Timings depend on the machine, its load and its power settings. Compare models only with timings
measured on the same device in the same session.

## Reports

- `GET /api/evaluations` / `GET /api/evaluations/{id}`: the stored metrics as JSON.
- `GET /api/evaluations/report?ids=1,2&format=json`: a JSON document with an application header, a
  note that values were computed from held-out labelled data with unlabelled pixels excluded, and the
  selected evaluations (model information, dataset fingerprint, all metrics).
- `GET /api/evaluations/report?ids=1,2&format=csv`: long-format CSV with the columns
  `evaluation_id, experiment_id, experiment_name, arch, split, label_variant, images, section, class,
  metric, value`. Sections: `summary` (pixel accuracy, mean IoU/Dice/precision/recall, evaluated
  pixels), `per_class` (IoU, Dice, precision, recall, support and predicted pixels per class),
  `confusion_matrix` (one row per true/predicted pair), `inference` (device and timing values) and
  `model` (parameters, checkpoint MB, checkpoint epoch).
- `GET /api/experiments/{id}/report?format=csv` exports an experiment's per-epoch training history.

Each stored evaluation also records the model (experiment id, name, architecture, configuration,
checkpoint epoch, `is_fixture`), the dataset fingerprint and whether only a subset was evaluated.

## Confidence and uncertainty maps

For a single-image analysis, the application also shows:

- **Confidence:** the softmax probability of the predicted class at each pixel;
- **Uncertainty:** the predictive entropy of the softmax distribution divided by log(4), so it lies
  in [0, 1].

These are the model's own scores. **Softmax confidence is not calibrated**: modern neural networks are
often over-confident (Guo et al., 2017), and a confident prediction can be wrong. Calibration
(reliability diagrams, expected calibration error, temperature scaling) has **not** been tested in
this project, so confidence values must not be read as probabilities of being correct.
