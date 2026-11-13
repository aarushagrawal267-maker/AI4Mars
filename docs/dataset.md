# Dataset: AI4Mars (Curiosity NavCam)

## Source

| Field | Value |
|---|---|
| Title | AI4Mars: A Dataset for Terrain-Aware Autonomous Driving on Mars |
| Repository | Zenodo, record 15995036 - https://zenodo.org/records/15995036 |
| DOI | [10.5281/zenodo.15995036](https://doi.org/10.5281/zenodo.15995036) |
| Version | 0.6 (publication date 2022-10-22) |
| Creators | Swan, Robert; Ono, Masahiro; Atha, Deegan |
| Licence | CC-BY-4.0 |
| Archive used | `ai4mars-dataset-merged-0.6.zip`, 16,232,481,989 bytes, MD5 `daf80a86021253292e6c425f97baa5c6`, 146,179 entries |
| Not used | `ai4mars-labels-unmerged.zip` (about 1.57 GB, raw per-labeller annotations) |

These values were checked by inspecting the archive and the Zenodo record directly. They are also
stored in `ml/datasets/sources.py`, and every download re-reads the record from the Zenodo API so that
a changed record is detected rather than trusted. If the API cannot be reached, the stored values are
used and the download is marked as not verified online.

The labels were collected through the Zooniverse citizen-science project
(https://www.zooniverse.org/projects/hiro-ono/ai4mars), with expert labels from NASA JPL for the test
sets, as described in the dataset's `info.md` and in the AI4Mars paper (Swan et al., CVPR Workshops
2021; see [references.md](references.md)). Please cite both the paper and the dataset DOI when using
results from this application.

### A mirror on Kaggle

A copy is published on Kaggle at
https://www.kaggle.com/datasets/yash92328/ai4mars-terrainaware-autonomous-driving-on-mars. Kaggle
requires a login, so its contents could not be inspected for this project and no assumption is made
about its folder layout. It is supported through the local import route: download the ZIP yourself,
place it in `data/raw/imports/` and import it. The loader then searches for the folders instead of
expecting one fixed layout (see [Folder detection](#folder-detection)).

## What the archive contains

The merged archive covers three missions: MSL (Curiosity), MER (Spirit and Opportunity) and M2020
(Perseverance). This project uses only the **MSL NavCam** portion:

| Folder (inside the archive root) | Content |
|---|---|
| `msl/ncam/images/edr` | 18,127 JPG images, 1024 x 1024, grayscale |
| `msl/ncam/labels/train` | 16,064 PNG labels; every one has a matching image |
| `msl/ncam/labels/test/masked-gold-min1-100agree` | 322 expert labels |
| `msl/ncam/labels/test/masked-gold-min2-100agree` | 322 expert labels (same images) |
| `msl/ncam/labels/test/masked-gold-min3-100agree` | 322 expert labels (same images) |
| `msl/ncam/images/mxy` | Rover masks (binary) |
| `msl/ncam/images/rng-30m` | 30 m range masks (binary) |

The test images have zero file-name overlap with the training labels.

### Label encoding

From the dataset's own `info.md` and `label_keys.json` (`NAV` key), and verified on real label files:
labels are 8-bit single-channel grayscale PNGs with

| Value | Class |
|---|---|
| 0 | soil |
| 1 | bedrock |
| 2 | sand |
| 3 | big rock |
| 255 | NULL (no label) |

Value 255 is used as the ignore index everywhere in this project (loss, metrics, statistics). The ids
are fixed in `ml/classes.py`; only the display colours can be changed.

### How the labels were made

- **Train labels** are crowdsourced and merged: each image needed at least 3 labellers, and a pixel
  keeps a class only when at least 2/3 of them agree. Other pixels are NULL.
- **Test labels** were made by experts (rover planners and scientists, per the AI4Mars paper) and
  require 100 % agreement on each pixel. The three variants differ in how many of the three experts
  must have labelled the pixel: at least 1 (`min1`), 2 (`min2`) or all 3 (`min3`). `min3` is the
  sparsest and most conservative.
- According to `info.md`, unless otherwise noted, pixels further than 30 m and pixels showing the
  rover itself are masked (NULL) in the labels. The mask folders are therefore not needed for
  training and are not downloaded by the subset route.

### Why MSL NavCam

- The dataset changelog ranks data-cleaning quality as
  MSL NavCam > MER > MSL MastCam > M2020 NAV > M2020 Geology.
- MSL MastCam and M2020 have no expert test set; MSL NavCam has three.
- MSL NavCam has rover and range masks applied to its labels.
- One camera type keeps the input distribution consistent (grayscale, 1024 x 1024).

## Acquiring the data

All three routes end with the original archive layout under `data/raw/` and then index it.

### 1. Subset via HTTP range requests (recommended for laptops)

The full archive is about 16 GB, but Zenodo supports HTTP range requests.
`ml/datasets/remote_zip.py` reads the ZIP's central directory over HTTP (a few MB) and then fetches
only the byte ranges that hold the wanted files:

- `plan_subset` (`ml/datasets/acquire.py`) groups the training pairs by **rover position** (site +
  drive, parsed from the file name), shuffles the positions with a fixed seed and takes whole
  positions until the requested number of training pairs is reached (so the result can be slightly
  above the request). The complete expert test set (all three variants) and the small documentation
  files (`info.md`, `label_keys.json`, ...) are always included.
- Nearby members are grouped into one request (up to 48 MB per request, gaps up to 1 MB).
- Each member's data is decompressed and checked against the **CRC-32** stored in the archive; a
  mismatch aborts the download.
- Files are written as `.part` and renamed when complete. Files already present with the expected
  size are skipped, so an interrupted download resumes.
- Requests are spaced at least 0.55 s apart and HTTP 429 responses are honoured (`Retry-After`),
  to stay within Zenodo's rate limits. Failed requests are retried with back-off.
- Before anything is fetched, the remote archive size must equal the size in the Zenodo record.

From the web UI (Dataset Explorer) or `POST /api/dataset/download` with
`{"mode": "subset", "train_count": 3000}` (50-20000, or `null` for all training pairs).

### 2. Full download

`POST /api/dataset/download` with `{"mode": "full", "confirm_full": true}`. The archive is streamed
to `data/raw/ai4mars-dataset-merged-0.6.zip.part` (resumable with a range request), its **MD5** is
verified against the record, and then only the MSL image/label folders and documentation files are
extracted. The ZIP is deleted after extraction.

### 3. Local import

Place a ZIP in `data/raw/imports/` (for example a manual Zenodo or Kaggle download) and choose it in
the Dataset Explorer, or call `POST /api/dataset/import` with `{"archive": "<file name>.zip"}`. Only
bare file names of ZIPs that are actually in that folder are accepted. Extraction is restricted to
`msl/images/edr`, `msl/labels/train|test` and `msl/ncam/...` equivalents plus documentation files,
and every member path is checked against zip-slip.

### Command line

`scripts/download_dataset.py` calls the same functions as the web UI:

```bash
# Windows: .venv\Scripts\python scripts\download_dataset.py ...
.venv/bin/python scripts/download_dataset.py --train-count 3000   # subset (default 3000, seed 42)
.venv/bin/python scripts/download_dataset.py --train-count all    # all MSL NavCam training pairs
.venv/bin/python scripts/download_dataset.py --full               # full 16 GB archive, MD5-verified
# options: --dest DIR (default data/raw), --seed N (subset selection, default 42)
```

The script downloads and extracts but does **not** build the index. Afterwards click *Load dataset*
in the Dataset Explorer or call `POST /api/dataset/load`.

## Folder detection

`ml/datasets/layout.py` walks `data/raw` (up to 5 levels deep) and looks for a folder containing
both `images/edr` and `labels/train`. This finds the v0.6 layout (`msl/ncam/...`) and older layouts
(`msl/images/edr`, `msl/labels/...`) alike; NavCam folders are preferred when several candidates
exist. Test variants are read from the sub-folders of `labels/test` (`masked-gold-min3-100agree`
becomes `min3`). `images/mxy`, `images/rng-30m`, `info.md` and `label_keys.json` are recorded when
present. If nothing suitable is found, the API says so instead of guessing.

## Index and validation report

`ml/datasets/index.py` builds `data/processed/index.json`, the single description of what is on disk.
For every label it opens the PNG and records its size and per-class pixel histogram. It checks and
reports:

- images found, training labels found, test images with labels per variant;
- labels without a matching image, and images without any label;
- unreadable files (image or label cannot be opened, or label is not single-channel);
- size mismatches between an image and its label;
- unexpected label values (anything other than 0, 1, 2, 3, 255);
- fully unlabelled training labels (no class pixel at all), which are excluded because they
  provide no supervision;
- training labels whose image is also in the expert test set (excluded from training) and the
  train/test file-name overlap;
- the set of label values actually observed.

Problem lists are capped at 200 entries each; the counts are always complete. The index also stores
per-split statistics (class pixels, percentage of labelled pixels per class, number of images
containing each class, unlabelled percentage), the number of rover positions and a **fingerprint**
(SHA-256 of all record ids and splits, first 16 hex digits). Experiments store the fingerprint they
were trained on, and the preprocessing cache is keyed on it.

## Splits

| Split | Source | How it is formed |
|---|---|---|
| train | official `labels/train` | everything not assigned to validation |
| val | official `labels/train` | about 10 % (`MTI_VAL_FRACTION`), chosen by rover position |
| test | official expert test set | used unchanged, three label variants |

**Validation is grouped by rover position.** MSL file names encode a site and a drive index
(`ml/datasets/naming.py`, following the MSL camera SIS referenced by `info.md`); frames from one
position (stereo pairs, panorama tiles) look very similar. All frames of a position go to the same
split, so near-duplicate frames never straddle train and validation. Positions are shuffled with
`MTI_SPLIT_SEED` (default 42) and assigned to validation until about 10 % of the pairs are reached.

**The official test set is kept unchanged.** It is never used for training, model selection or early
stopping. Test images that share a rover position with training data are counted in the index
(`leakage_check`) and shown in the UI, not hidden or removed, so results stay comparable with other
work that uses the official split.

## Preprocessing cache

Decoding and resizing 1024 x 1024 JPGs every epoch would dominate CPU training time.
`ml/datasets/cache.py` resizes each train and validation pair once to the training resolution and
stores them as memory-mapped `uint8` arrays in `data/processed/cache_<resolution>_<fingerprint>/`
(`train_images.npy`, `train_labels.npy`, `train_ids.json`, and the same for `val`). Images use area
interpolation; labels use nearest-neighbour so class ids are never blended and 255 is preserved. A
cache is reused by later runs at the same resolution as long as the record ids match, and is written
to temporary files that are renamed only when complete. The test set is not cached: evaluation reads
the original files at native resolution.

## Statistics of the subset used in this environment

The subset downloaded and indexed here contains 4,015 training pairs (seed 42) plus the full expert
test set. After validation:

| | Value |
|---|---|
| Valid pairs | 4,294 = 3,570 train / 402 validation / 322 test |
| Fully unlabelled training labels (excluded) | 43 |
| Unreadable files / size mismatches / unexpected label values / labels without image | 0 / 0 / 0 / 0 |
| Observed label values | {0, 1, 2, 3, 255} |
| Rover positions (site + drive) | 480 |
| Train / validation positions shared | 0 |
| Test images sharing a rover position with the downloaded training subset | 31 of 322 |
| Image size | all 1024 x 1024 |

Unlabelled pixels and class distribution (percent of *labelled* pixels):

| Split | Unlabelled % | Soil | Bedrock | Sand | Big rock |
|---|---|---|---|---|---|
| train | 43.35 | 33.65 | 50.86 | 14.28 | 1.21 |
| validation | 39.26 | 42.94 | 39.56 | 17.22 | 0.28 |
| test min1 | 42.46 | 45.86 | 30.89 | 23.05 | 0.20 |
| test min2 | 51.60 | 49.10 | 27.36 | 23.46 | 0.08 |
| test min3 | 65.35 | 51.35 | 22.39 | 26.19 | 0.07 |

These numbers describe this particular subset; a different `train_count` or seed gives different
train/validation figures. The current values for any installation are shown in the Dataset Explorer
and returned by `GET /api/dataset/statistics`.

## Class imbalance

*Big rock* makes up about 1 % of the labelled training pixels and well under 1 % of the validation
and test pixels (0.07 % in `min3`). Consequences:

- A model can reach high pixel accuracy while almost never predicting big rock, so pixel accuracy
  alone is misleading; per-class IoU and mean IoU are reported alongside it.
- The big-rock IoU is computed from very few pixels and is therefore noisy, and it pulls the mean IoU
  strongly because the mean weights each class equally.
- Training offers optional inverse-square-root class weighting and a soft Dice term to counter the
  imbalance (see [training.md](training.md)); neither is a guarantee.

## Possible biases and caveats

- **Crowdsourced training labels.** Volunteers, not experts, labelled the training set. The 2/3
  agreement rule removes contested pixels, which likely leaves the harder, ambiguous regions
  (boundaries, mixed terrain) unlabelled, so the supervision is biased towards clear-cut areas.
- **Different labelling protocol for test.** Experts with 100 % agreement labelled the test set. The
  class distribution differs noticeably from training (more soil and sand, less bedrock), and the
  agreement variants differ again: `min3` keeps only about a third of the pixels. Scores on `min1`
  and `min3` are not interchangeable.
- **One rover, one region.** All MSL images come from Curiosity's traverse; terrain elsewhere on Mars
  (or seen by other rovers) may look different. The subset used here contains only left-eye NavCam
  frames (`NLA`/`NLB` in the file names).
- **Not a separate time period.** In this subset, test images span the same spacecraft-clock range as
  the training images, and 31 of them were taken at a rover position that also appears in training.
  The test set therefore measures generalisation to new frames, not necessarily to new terrain.
- **Range and rover masking.** Labels exclude terrain beyond about 30 m and the rover itself. Models
  are never supervised on distant terrain or on the sky (there is no sky class), yet they still
  assign one of the four classes to every pixel at inference time, including sky and rover pixels.
- **Grayscale only.** NavCam images are single-channel; colour information is not available, and
  colour uploads are converted to grayscale before inference.
- **Class definitions are coarse.** Four classes describe terrain appearance, not slope, rock height,
  soil strength or other properties that determine whether a rover can drive somewhere.
