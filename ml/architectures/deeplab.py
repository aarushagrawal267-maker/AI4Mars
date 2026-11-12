"""DeepLabV3 (Chen et al., 2017) from Torchvision, adapted to 4 terrain classes.

DeepLabV3 uses atrous (dilated) convolutions and an Atrous Spatial Pyramid Pooling
(ASPP) module to capture context at several scales without losing resolution.
We take Torchvision's implementation with a pretrained backbone and replace the
final 1x1 classification layer (21 COCO/VOC classes) with a new 4-class layer.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torchvision.models import segmentation as tvseg

log = logging.getLogger(__name__)

BACKBONES = {
    "mobilenet_v3_large": (tvseg.deeplabv3_mobilenet_v3_large, tvseg.DeepLabV3_MobileNet_V3_Large_Weights),
    "resnet50": (tvseg.deeplabv3_resnet50, tvseg.DeepLabV3_ResNet50_Weights),
}


class DeepLabV3(nn.Module):
    """Wrapper returning plain logits ``(N, num_classes, H, W)`` at input resolution.

    Args:
        backbone: ``mobilenet_v3_large`` (laptop friendly) or ``resnet50``.
        pretrained: ``"coco"`` loads Torchvision's full segmentation weights and swaps
            the classifier layer; ``"none"`` starts from random initialisation.
            If the weights cannot be downloaded the model falls back to random
            initialisation and ``pretrained_loaded`` is False so this is recorded.
    """

    def __init__(self, num_classes: int = 4, backbone: str = "mobilenet_v3_large", pretrained: str = "coco"):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown DeepLabV3 backbone {backbone!r}; choose from {sorted(BACKBONES)}")
        if pretrained not in ("coco", "none"):
            raise ValueError("pretrained must be 'coco' or 'none'")
        builder, weights_enum = BACKBONES[backbone]
        self.backbone_name = backbone
        self.pretrained_loaded = False
        net = None
        if pretrained == "coco":
            try:
                net = builder(weights=weights_enum.DEFAULT)
                net.aux_classifier = None
                in_ch = net.classifier[-1].in_channels
                net.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
                self.pretrained_loaded = True
            except Exception as exc:  # offline or blocked download
                log.warning("Could not load pretrained DeepLabV3 weights (%s); using random initialisation.", exc)
                net = None
        if net is None:
            net = builder(weights=None, weights_backbone=None, num_classes=num_classes, aux_loss=False)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)["out"]
