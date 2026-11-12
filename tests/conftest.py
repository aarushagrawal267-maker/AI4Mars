"""Shared pytest fixtures.

The dataset built here is SYNTHETIC: tiny generated images laid out exactly like
the real AI4Mars v0.6 archive, so the loader, validator, trainer and API can be
tested quickly and offline. It is never presented as real data, and the model
used for interface tests is the clearly labelled untrained fixture.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_ROOT = "ai4mars-dataset-merged-0.6"


def make_pair(rng: np.random.Generator, size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic image whose brightness bands match its label bands, plus an unlabelled strip."""
    label = np.full((size, size), 255, dtype=np.uint8)
    cuts = sorted(rng.choice(np.arange(8, size - 8), size=3, replace=False).tolist())
    bounds = [0, *cuts, size]
    image = np.zeros((size, size), dtype=np.uint8)
    for cls in range(4):
        label[:, bounds[cls]:bounds[cls + 1]] = cls
        image[:, bounds[cls]:bounds[cls + 1]] = 40 + cls * 60
    label[: size // 6] = 255            # "sky"/far range is unlabelled, as in the real data
    image = np.clip(image + rng.normal(0, 4, image.shape), 0, 255).astype(np.uint8)
    return image, label


def build_synthetic_dataset(raw_root: Path, positions: int = 8, per_position: int = 3, n_test: int = 4,
                            size: int = 64, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    base = raw_root / DATASET_ROOT / "msl" / "ncam"
    img_dir, train_dir = base / "images" / "edr", base / "labels" / "train"
    img_dir.mkdir(parents=True)
    train_dir.mkdir(parents=True)
    test_dirs = {v: base / "labels" / "test" / f"masked-gold-{v}-100agree" for v in ("min1", "min3")}
    for d in test_dirs.values():
        d.mkdir(parents=True)
    (raw_root / DATASET_ROOT / "label_keys.json").write_text(
        '{"NAV": {"0": "soil", "1": "bedrock", "2": "sand", "3": "big rock", "255": null}}')
    train_ids, test_ids = [], []
    sclk = 400000000
    for p in range(positions):
        for _ in range(per_position):
            sclk += 37
            stem = f"NLB_{sclk}EDR_F{p + 1:03d}{p * 10:04d}NCAM00353M1"
            image, label = make_pair(rng, size)
            Image.fromarray(image).save(img_dir / f"{stem}.JPG", quality=95)
            Image.fromarray(label).save(train_dir / f"{stem}.png")
            train_ids.append(stem)
    for t in range(n_test):
        sclk += 1000
        stem = f"NLB_{sclk}EDR_F{90 + t:03d}{t:04d}NCAM00353M1"
        image, label = make_pair(rng, size)
        Image.fromarray(image).save(img_dir / f"{stem}.JPG", quality=95)
        Image.fromarray(label).save(test_dirs["min1"] / f"{stem}_merged.png")
        strict = label.copy()
        strict[:, : size // 2] = 255     # the stricter variant keeps fewer labelled pixels
        Image.fromarray(strict).save(test_dirs["min3"] / f"{stem}_merged.png")
        test_ids.append(stem)
    return {"raw_root": raw_root, "base": base, "images": img_dir, "train_labels": train_dir,
            "test_dirs": test_dirs, "train_ids": train_ids, "test_ids": test_ids}


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> dict:
    return build_synthetic_dataset(tmp_path / "data" / "raw")


@pytest.fixture()
def settings(tmp_path: Path):
    from backend.app.core.config import Settings

    return Settings(data_dir=tmp_path / "data", checkpoint_dir=tmp_path / "checkpoints",
                    experiments_dir=tmp_path / "experiments", enable_test_fixture=True, _env_file=None)


@pytest.fixture()
def client(settings, synthetic_dataset):
    from fastapi.testclient import TestClient

    from backend.app.main import create_app

    with TestClient(create_app(settings)) as c:
        yield c


def wait_for(fn, timeout: float = 60.0, interval: float = 0.2):
    """Poll ``fn`` until it returns a truthy value."""
    end = time.time() + timeout
    while time.time() < end:
        value = fn()
        if value:
            return value
        time.sleep(interval)
    raise TimeoutError("condition not met in time")


@pytest.fixture()
def loaded_client(client):
    """Client whose synthetic dataset has already been scanned and indexed."""
    assert client.post("/api/dataset/load").status_code == 202
    wait_for(lambda: client.get("/api/dataset/status").json()["job"]["status"] != "running")
    status = client.get("/api/dataset/status").json()
    assert status["job"]["status"] == "completed", status["job"]
    return client


def png_bytes(size: tuple[int, int] = (96, 80), seed: int = 1) -> bytes:
    import io

    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()
