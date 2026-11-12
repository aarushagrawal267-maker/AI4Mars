"""Training configuration: one plain, serialisable, validated object per experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..architectures.factory import ARCHITECTURES, normalise_model_config

ALLOWED_RESOLUTIONS = (128, 160, 192, 224, 256, 320, 384, 448, 512)


@dataclass
class TrainConfig:
    arch: str = "unet"
    model_config: dict[str, Any] = field(default_factory=dict)
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    resolution: int = 256
    device: str = "auto"                      # auto | cpu | cuda
    max_train_samples: int | None = None      # None = full training split
    max_val_samples: int | None = None
    dice_weight: float = 0.5                  # 0 disables the Dice term
    class_weighting: str = "none"             # none | inverse_sqrt
    scheduler: str = "cosine"                 # cosine | plateau | none
    early_stopping_patience: int = 5          # epochs without val mIoU improvement; 0 disables
    augment: bool = True
    seed: int = 42
    num_workers: int = 0

    def validate(self) -> "TrainConfig":
        if self.arch not in ARCHITECTURES:
            raise ValueError(f"arch must be one of {ARCHITECTURES}")
        self.model_config = normalise_model_config(self.arch, self.model_config)
        if not 1 <= self.epochs <= 500:
            raise ValueError("epochs must be between 1 and 500")
        if not 1 <= self.batch_size <= 128:
            raise ValueError("batch_size must be between 1 and 128")
        if self.arch == "deeplabv3" and self.batch_size < 2:
            raise ValueError("DeepLabV3 needs batch_size >= 2 (its ASPP pooling branch uses batch normalisation)")
        if not 1e-6 <= self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be between 1e-6 and 1")
        if not 0 <= self.weight_decay <= 1:
            raise ValueError("weight_decay must be between 0 and 1")
        if self.resolution not in ALLOWED_RESOLUTIONS:
            raise ValueError(f"resolution must be one of {ALLOWED_RESOLUTIONS}")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        for name in ("max_train_samples", "max_val_samples"):
            v = getattr(self, name)
            if v is not None and v < 2:
                raise ValueError(f"{name} must be at least 2 (or omitted for the full split)")
        if not 0 <= self.dice_weight <= 5:
            raise ValueError("dice_weight must be between 0 and 5")
        if self.class_weighting not in ("none", "inverse_sqrt"):
            raise ValueError("class_weighting must be 'none' or 'inverse_sqrt'")
        if self.scheduler not in ("cosine", "plateau", "none"):
            raise ValueError("scheduler must be cosine, plateau or none")
        if not 0 <= self.early_stopping_patience <= 100:
            raise ValueError("early_stopping_patience must be between 0 and 100")
        if not 0 <= self.num_workers <= 16:
            raise ValueError("num_workers must be between 0 and 16")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown training options: {sorted(unknown)}")
        return cls(**data).validate()
