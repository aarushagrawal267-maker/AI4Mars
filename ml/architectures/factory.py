"""Model factory and checkpoint (de)serialisation shared by training, evaluation and inference."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..classes import CLASS_KEYS, IGNORE_INDEX, NUM_CLASSES
from ..preprocessing.transforms import IMAGENET_MEAN, IMAGENET_STD
from .deeplab import DeepLabV3
from .unet import UNet

ARCHITECTURES = ("unet", "deeplabv3")
CHECKPOINT_FORMAT = 1

DEFAULT_MODEL_CONFIG: dict[str, dict[str, Any]] = {
    "unet": {"base_channels": 32, "depth": 4, "bilinear": True, "dropout": 0.0},
    "deeplabv3": {"backbone": "mobilenet_v3_large", "pretrained": "coco"},
}


def normalise_model_config(arch: str, model_config: dict | None) -> dict:
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {arch!r}; choose from {ARCHITECTURES}")
    cfg = dict(DEFAULT_MODEL_CONFIG[arch])
    for k, v in (model_config or {}).items():
        if k not in cfg:
            raise ValueError(f"Unknown option {k!r} for architecture {arch}")
        cfg[k] = v
    return cfg


def build_model(arch: str, model_config: dict | None = None, for_inference: bool = False) -> nn.Module:
    """Create a model. ``for_inference`` skips pretrained downloads (weights come from a checkpoint)."""
    cfg = normalise_model_config(arch, model_config)
    if arch == "unet":
        return UNet(in_channels=3, num_classes=NUM_CLASSES, **cfg)
    pretrained = "none" if for_inference else cfg["pretrained"]
    return DeepLabV3(num_classes=NUM_CLASSES, backbone=cfg["backbone"], pretrained=pretrained)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def resolve_device(requested: str = "auto") -> torch.device:
    """``auto`` -> CUDA when available, else CPU. Asking for CUDA without a GPU is an error, not a silent fallback."""
    if requested in ("auto", None, ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA-capable GPU is available to PyTorch.")
    return torch.device(requested)


def save_checkpoint(path: Path, model: nn.Module, arch: str, model_config: dict, resolution: int,
                    extra: dict | None = None) -> None:
    """Atomically write a checkpoint containing only tensors and plain Python types."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "arch": arch,
        "model_config": dict(model_config),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "classes": list(CLASS_KEYS),
        "ignore_index": IGNORE_INDEX,
        "resolution": int(resolution),
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD), "input": "grayscale->3ch"},
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "extra": extra or {},
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[nn.Module, dict]:
    """Load a checkpoint written by ``save_checkpoint``.

    ``weights_only=True`` restricts unpickling to tensors and primitive containers,
    so a tampered checkpoint cannot execute code when loaded.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path.name}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unrecognised checkpoint format")
    if payload.get("classes") != list(CLASS_KEYS):
        raise ValueError("Checkpoint was trained for a different class set")
    model = build_model(payload["arch"], payload["model_config"], for_inference=True)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    meta = {k: v for k, v in payload.items() if k != "state_dict"}
    meta["parameters"] = count_parameters(model)
    meta["file_size_bytes"] = path.stat().st_size
    return model, meta
