"""PyTorch dataset over the memory-mapped preprocessing cache."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..preprocessing.transforms import augment_pair, gray_to_tensor, label_to_tensor


class TerrainDataset(Dataset):
    """Yields (image tensor (3, H, W), label tensor (H, W) with 255 = ignore)."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: list[int] | None = None,
                 augment: bool = False, seed: int = 0):
        if len(images) != len(labels):
            raise ValueError("images and labels must have the same length")
        self.images, self.labels = images, labels
        self.indices = list(range(len(images))) if indices is None else list(indices)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        j = self.indices[i]
        img, lab = np.array(self.images[j]), np.array(self.labels[j])
        if self.augment:
            img, lab = augment_pair(img, lab, self.rng)
        return gray_to_tensor(img), label_to_tensor(lab)
