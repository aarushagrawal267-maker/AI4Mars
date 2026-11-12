"""Request/response schemas for the REST API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ClassKey = Literal["soil", "bedrock", "sand", "big_rock"]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    app: str
    version: str
    time: float


class JobResponse(BaseModel):
    id: str
    kind: str
    status: str
    progress: float
    stage: str
    message: str
    error: str | None
    params: dict[str, Any]
    created_at: float
    finished_at: float | None
    result: Any | None = None


class DatasetDownloadRequest(BaseModel):
    mode: Literal["subset", "full"] = "subset"
    train_count: int | None = Field(3000, ge=50, le=20000,
                                    description="Training pairs to fetch in subset mode; null fetches all")
    confirm_full: bool = Field(False, description="Must be true for the 16 GB full download")


class DatasetImportRequest(BaseModel):
    archive: str = Field(..., min_length=1, max_length=255,
                         description="File name of a .zip placed in data/raw/imports")

    @field_validator("archive")
    @classmethod
    def bare_name(cls, v: str) -> str:
        if "/" in v or "\\" in v or v in (".", "..") or ":" in v:
            raise ValueError("archive must be a bare file name, not a path")
        return v


class TrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    arch: Literal["unet", "deeplabv3"] = "unet"
    model_options: dict[str, Any] = Field(default_factory=dict,
                                          description="unet: base_channels, depth, bilinear, dropout; "
                                                      "deeplabv3: backbone, pretrained")
    epochs: int = Field(10, ge=1, le=500)
    batch_size: int = Field(8, ge=1, le=128)
    learning_rate: float = Field(1e-3, ge=1e-6, le=1.0)
    weight_decay: float = Field(1e-4, ge=0, le=1)
    resolution: int = 256
    device: Literal["auto", "cpu", "cuda"] = "auto"
    max_train_samples: int | None = Field(None, ge=2)
    max_val_samples: int | None = Field(None, ge=2)
    dice_weight: float = Field(0.5, ge=0, le=5)
    class_weighting: Literal["none", "inverse_sqrt"] = "none"
    scheduler: Literal["cosine", "plateau", "none"] = "cosine"
    early_stopping_patience: int = Field(5, ge=0, le=100)
    augment: bool = True
    seed: int = Field(42, ge=0, le=2**31 - 1)

    def to_config_dict(self) -> dict:
        d = self.model_dump(include=set(TrainRequest.model_fields))
        d["model_config"] = d.pop("model_options")
        return d


class TrainStartRequest(TrainRequest):
    name: str | None = Field(None, max_length=120)
    confirm_large: bool = False


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str = Field(..., min_length=1, max_length=64)
    split: Literal["test", "val"] = "test"
    label_variant: str | None = Field(None, max_length=32, description="Test label agreement variant, e.g. min3")
    max_images: int | None = Field(None, ge=1, le=100000)


class ColorsRequest(BaseModel):
    soil: str | None = None
    bedrock: str | None = None
    sand: str | None = None
    big_rock: str | None = None


class TerrainCost(BaseModel):
    cost: float = Field(..., gt=0, le=1000)
    passable: bool = True


class CostsRequest(BaseModel):
    soil: TerrainCost
    bedrock: TerrainCost
    sand: TerrainCost
    big_rock: TerrainCost


class NavigationPlanRequest(BaseModel):
    analysis_id: str = Field(..., min_length=1, max_length=32)
    grid_size: int = Field(64, ge=8, le=160, description="Number of grid cells along the image's long side")
    start: tuple[int, int]
    goal: tuple[int, int]
    costs: CostsRequest | None = Field(None, description="Omit to use the saved costs")
    blocked: list[tuple[int, int]] = Field(default_factory=list, max_length=20000)
    allow_diagonal: bool = True
    uncertainty_weight: float = Field(0.0, ge=0, le=10)
    alternatives: bool = True


class ExperimentPatch(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=120)
    notes: str | None = Field(None, max_length=5000)
