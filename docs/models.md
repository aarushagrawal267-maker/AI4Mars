# Models

Two semantic-segmentation architectures are available. Both output per-pixel logits for the four
terrain classes (soil, bedrock, sand, big rock) at the input resolution. Neither has an "unlabelled"
or "sky" output: unlabelled pixels (255) are ignored during training and evaluation, so at inference
time every pixel receives one of the four classes.

## U-Net

Reference: Ronneberger, Fischer and Brox, "U-Net: Convolutional Networks for Biomedical Image
Segmentation", MICCAI 2015. Implemented from scratch in `ml/architectures/unet.py`.

- **Encoder:** an input block and `depth` down-sampling stages. Each stage is 2 x 2 max-pooling
  followed by a double convolution (3 x 3 convolution, batch normalisation, ReLU, twice). The
  channel count doubles at every stage, starting from `base_channels`.
- **Decoder:** mirrored stages. Each up-samples (bilinear up-sampling plus a 1 x 1 convolution, or a
  2 x 2 transposed convolution), **concatenates the encoder feature map of the same scale (skip
  connection)** and applies a double convolution. If an odd input size makes the shapes differ by a
  pixel, the up-sampled map is interpolated to the skip map's size.
- **Head:** a 1 x 1 convolution to 4 class logits.
- **Initialisation:** random (PyTorch defaults). U-Net is always trained from scratch here.

| Option | Values | Default | Meaning |
|---|---|---|---|
| `base_channels` | 16, 32, 64 in the UI (any value >= 4 in code) | 32 | Width of the first stage (64 matches the original paper) |
| `depth` | 3, 4, 5 in the UI (2-5 in code) | 4 | Number of down/up-sampling stages |
| `bilinear` | true / false | true | Bilinear up-sampling (lighter) instead of transposed convolutions |
| `dropout` | 0-1 | 0.0 | Spatial dropout at the bottleneck stage |

Parameter counts (4 output classes, bilinear, counted with `count_parameters`):

| Configuration | Parameters |
|---|---|
| base 16, depth 4 | 1,812,068 |
| base 32, depth 4 (default) | 7,240,900 |
| base 64, depth 4 | 28,948,868 |

## DeepLabV3

Reference: Chen, Papandreou, Schroff and Adam, "Rethinking Atrous Convolution for Semantic Image
Segmentation", arXiv:1706.05587, 2017. Implementation: torchvision's
`deeplabv3_mobilenet_v3_large` / `deeplabv3_resnet50`, wrapped in `ml/architectures/deeplab.py`.

DeepLabV3 uses atrous (dilated) convolutions in the backbone and an Atrous Spatial Pyramid Pooling
(ASPP) head to combine context at several scales. The wrapper returns the main output only (`"out"`),
which torchvision already up-samples to the input size.

| Option | Values | Default |
|---|---|---|
| `backbone` | `mobilenet_v3_large`, `resnet50` | `mobilenet_v3_large` |
| `pretrained` | `coco`, `none` | `coco` |

- **MobileNetV3-Large** (Howard et al., ICCV 2019) is the default because it is light enough for a
  laptop CPU. **ResNet-50** (He et al., CVPR 2016) is larger and slower.
- **`pretrained: coco`** loads torchvision's complete DeepLabV3 segmentation weights (backbone *and*
  ASPP head; torchvision trained them on a subset of COCO using the 20 Pascal VOC categories, 21
  outputs with background). The final 1 x 1 classification convolution is then replaced by a new,
  randomly initialised one with 4 outputs, and the auxiliary classifier is removed.
- **`pretrained: none`** builds the same network with random initialisation and no auxiliary head.
- **If the pretrained weights cannot be downloaded** (offline, blocked), the model falls back to
  random initialisation and a warning is logged. This is recorded, not hidden: the run's
  `pretrained_loaded` flag is `false` in `status.json`, in the `experiments` table, in the checkpoint
  metadata and in the UI. A run with `pretrained: coco` and `pretrained_loaded: false` is effectively
  a from-scratch run. Downloaded weights are cached by torchvision (by default under
  `~/.cache/torch/hub/checkpoints`).
- Batch size must be at least 2 for DeepLabV3, because its ASPP image-pooling branch uses batch
  normalisation on a 1 x 1 feature map.

Parameter counts (4 output classes, no auxiliary head):

| Backbone | Parameters |
|---|---|
| MobileNetV3-Large | 11,021,108 |
| ResNet-50 | 39,634,500 |

## Input handling (both models)

`ml/preprocessing/transforms.py` prepares identical input for both architectures, so the comparison is
like-for-like:

1. The image is read as 8-bit grayscale. Colour uploads are converted to luminance first, because the
   models have only ever seen grayscale NavCam data.
2. The grayscale channel is **replicated to 3 channels** and scaled to [0, 1].
3. **ImageNet normalisation** is applied (mean 0.485/0.456/0.406, std 0.229/0.224/0.225). The
   pretrained DeepLabV3 backbone expects it; U-Net uses it too.

Training uses square inputs at the configured resolution (128-512 px) from the preprocessing cache.
For inference and evaluation, the long side of the image is scaled to the training resolution with
the aspect ratio kept, and both sides are rounded to a multiple of 32 (needed by U-Net's
down-/up-sampling path). The softmax probabilities are then up-sampled bilinearly to the output size.

## Checkpoint format

`ml/architectures/factory.py` writes and reads checkpoints (`CHECKPOINT_FORMAT = 1`). A checkpoint is
a plain dictionary containing only tensors and primitive Python types:

| Key | Content |
|---|---|
| `format` | 1 |
| `arch` | `unet` or `deeplabv3` |
| `model_config` | The normalised architecture options (see tables above) |
| `state_dict` | Model weights, moved to CPU |
| `classes` | `["soil", "bedrock", "sand", "big_rock"]` |
| `ignore_index` | 255 |
| `resolution` | Training resolution in pixels |
| `normalization` | ImageNet mean/std and `"grayscale->3ch"` |
| `saved_at` | Timestamp |
| `extra` | Training configuration, train/val sizes, `pretrained_loaded`, dataset fingerprint, epoch and the validation metrics of that epoch (empty or fixture-specific for the test fixture) |

Writing is atomic: the file is saved as `<name>.tmp` and renamed with `os.replace`. Each experiment
has `checkpoints/<id>/last.pt` (latest completed epoch) and `checkpoints/<id>/best.pt` (best
validation mIoU). Inference and evaluation always use `best.pt`.

Loading uses `torch.load(path, map_location="cpu", weights_only=True)`, which refuses arbitrary
pickled objects, then checks the format number and the class list, rebuilds the architecture from
`arch` and `model_config` without downloading pretrained weights, loads the state dict, moves the
model to the device and switches it to evaluation mode. The parameter count and file size are added
to the returned metadata. The model registry keeps the two most recently used models in memory and
runs one forward pass at a time to bound memory use.

## Device selection

`resolve_device("auto")` uses CUDA when PyTorch can see a CUDA GPU and the CPU otherwise. Requesting
CUDA explicitly without a usable GPU is an error rather than a silent fallback. Apple MPS is not used.
