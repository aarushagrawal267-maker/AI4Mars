"""Image/label preprocessing shared by training, evaluation and inference.

AI4Mars NavCam frames are single-channel. Both architectures receive the same
input: the grayscale image replicated to three channels and normalised with
ImageNet statistics (required by the pretrained DeepLabV3 backbone, and used for
U-Net too so the comparison is like-for-like). Colour uploads are converted to
luminance first, because the models have only ever seen grayscale NavCam data.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def gray_to_tensor(gray: np.ndarray) -> torch.Tensor:
    """(H, W) uint8 -> normalised float tensor (3, H, W)."""
    t = torch.from_numpy(np.ascontiguousarray(gray)).float().div_(255.0)
    return (t.unsqueeze(0).expand(3, -1, -1) - _MEAN) / _STD


def label_to_tensor(label: np.ndarray) -> torch.Tensor:
    """(H, W) uint8 class ids (255 = ignore) -> int64 tensor."""
    return torch.from_numpy(np.ascontiguousarray(label)).long()


def augment_pair(img: np.ndarray, lab: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Light, label-preserving augmentation for rover imagery.

    * horizontal flip (Mars has no left/right preference; vertical flips are NOT
      used because sky/horizon/near-field layout is meaningful),
    * random scale-crop (60-100 % of the frame) resized back to the input size,
    * brightness / contrast jitter and mild sensor-like noise (image only).
    """
    h, w = img.shape
    if rng.random() < 0.5:
        img, lab = img[:, ::-1], lab[:, ::-1]
    if rng.random() < 0.7:
        scale = rng.uniform(0.6, 1.0)
        ch, cw = max(8, int(h * scale)), max(8, int(w * scale))
        y0, x0 = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
        img = cv2.resize(np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw]), (w, h), interpolation=cv2.INTER_LINEAR)
        lab = cv2.resize(np.ascontiguousarray(lab[y0:y0 + ch, x0:x0 + cw]), (w, h), interpolation=cv2.INTER_NEAREST)
    out = img.astype(np.float32)
    if rng.random() < 0.8:
        contrast, brightness = rng.uniform(0.8, 1.25), rng.uniform(-20, 20)
        out = (out - 127.5) * contrast + 127.5 + brightness
    if rng.random() < 0.3:
        out = out + rng.normal(0, rng.uniform(2, 6), size=out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8), np.ascontiguousarray(lab)


class ImageValidationError(ValueError):
    """Raised when an uploaded file is not an acceptable image."""


def decode_upload(data: bytes, max_pixels: int = 40_000_000) -> Image.Image:
    """Validate and decode uploaded bytes. Returns an RGB PIL image with EXIF rotation applied."""
    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    try:
        probe = Image.open(io.BytesIO(data))
        fmt = (probe.format or "").upper()
        probe.verify()  # detects truncated / corrupt files
    except Exception as exc:
        raise ImageValidationError("The file is not a readable image.") from exc
    if fmt not in {"JPEG", "PNG"}:
        raise ImageValidationError(f"Unsupported image format {fmt or 'unknown'}. Use JPG, JPEG or PNG.")
    img = Image.open(io.BytesIO(data))
    w, h = img.size
    if w < 32 or h < 32:
        raise ImageValidationError("Image is too small (minimum 32 x 32 pixels).")
    if w * h > max_pixels:
        raise ImageValidationError(f"Image is too large ({w} x {h}); maximum is {max_pixels // 1_000_000} megapixels.")
    try:
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except Exception as exc:
        raise ImageValidationError("The image data could not be decoded.") from exc


def inference_size(height: int, width: int, resolution: int, multiple: int = 32) -> tuple[int, int]:
    """Network input size for an arbitrary image: long side ~= ``resolution``, aspect ratio kept,
    both sides rounded to a multiple of ``multiple`` (needed by the U-Net down/up-sampling path)."""
    scale = resolution / max(height, width)
    nh = max(multiple, int(round(height * scale / multiple)) * multiple)
    nw = max(multiple, int(round(width * scale / multiple)) * multiple)
    return nh, nw


def prepare_inference_input(rgb: np.ndarray, resolution: int) -> tuple[torch.Tensor, np.ndarray]:
    """RGB uint8 (H, W, 3) -> (batch tensor (1, 3, h, w), grayscale uint8 (H, W))."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    nh, nw = inference_size(gray.shape[0], gray.shape[1], resolution)
    interp = cv2.INTER_AREA if nh < gray.shape[0] else cv2.INTER_LINEAR
    small = cv2.resize(gray, (nw, nh), interpolation=interp)
    return gray_to_tensor(small).unsqueeze(0), gray
