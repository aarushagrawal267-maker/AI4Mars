"""U-Net (Ronneberger, Fischer & Brox, 2015) for 4-class Mars terrain segmentation.

Encoder: ``depth`` stages of (Conv3x3-BatchNorm-ReLU) x 2, each followed by 2x2
max-pooling, doubling the channel count per stage. Decoder: mirrored stages that
upsample, concatenate the matching encoder feature map (skip connection) and
apply the same double convolution. A final 1x1 convolution produces per-pixel
class logits at the input resolution.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Upsample, concatenate the skip feature map, then DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, bilinear: bool):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                    nn.Conv2d(in_ch, in_ch // 2, 1))
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:  # odd input sizes: align to the skip map
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Configurable U-Net.

    Args:
        in_channels: input channels (3: grayscale replicated, see preprocessing).
        num_classes: output classes (4 terrain types).
        base_channels: channels of the first encoder stage (16 = light, 32 = default, 64 = original).
        depth: number of down/up-sampling stages (3-5).
        bilinear: bilinear upsampling (lighter) instead of transposed convolutions.
        dropout: spatial dropout at the bottleneck.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 4, base_channels: int = 32, depth: int = 4,
                 bilinear: bool = True, dropout: float = 0.0):
        super().__init__()
        if not 2 <= depth <= 5:
            raise ValueError("depth must be between 2 and 5")
        if base_channels < 4:
            raise ValueError("base_channels must be >= 4")
        chans = [base_channels * (2 ** i) for i in range(depth + 1)]
        self.depth = depth
        self.inc = DoubleConv(in_channels, chans[0])
        self.downs = nn.ModuleList(
            nn.Sequential(nn.MaxPool2d(2), DoubleConv(chans[i], chans[i + 1], dropout if i == depth - 1 else 0.0))
            for i in range(depth)
        )
        self.ups = nn.ModuleList(Up(chans[i + 1], chans[i], chans[i], bilinear) for i in reversed(range(depth)))
        self.head = nn.Conv2d(chans[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        y = skips.pop()
        for up in self.ups:
            y = up(y, skips.pop())
        return self.head(y)
