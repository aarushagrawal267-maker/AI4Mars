"""The demo-model seeder that makes a fresh clone usable without training."""
from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("seed_demo_models", ROOT / "scripts" / "seed_demo_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_bundle(root: Path) -> Path:
    b = root / "bundle"
    (b / "checkpoints" / "unet-x").mkdir(parents=True)
    (b / "checkpoints" / "unet-x" / "best.pt").write_bytes(b"weights")
    (b / "experiments" / "unet-x").mkdir(parents=True)
    (b / "experiments" / "unet-x" / "train.log").write_text("log")
    (b / "data" / "processed").mkdir(parents=True)
    (b / "data" / "processed" / "index.json").write_text("{}")
    with sqlite3.connect(b / "data" / "mti.sqlite3") as con:
        con.execute("CREATE TABLE experiments (id TEXT)")
        con.execute("INSERT INTO experiments VALUES ('unet-x')")
    return b


def test_seeds_empty_install_and_never_overwrites(tmp_path):
    mod = _load()
    bundle = _make_bundle(tmp_path)
    data, ck, ex = tmp_path / "data", tmp_path / "ck", tmp_path / "ex"
    # setup.sh creates an empty database first; the seeder must replace it
    data.mkdir()
    with sqlite3.connect(data / "mti.sqlite3") as con:
        con.execute("CREATE TABLE experiments (id TEXT)")

    assert mod.seed(data, ck, ex, bundle=bundle, dry_run=True)["status"] == "would-copy"
    assert not (ck / "unet-x" / "best.pt").exists()

    r = mod.seed(data, ck, ex, bundle=bundle)
    assert r["status"] == "copied" and r["database"] == "installed"
    assert (ck / "unet-x" / "best.pt").read_bytes() == b"weights"
    assert (ex / "unet-x" / "train.log").exists() and (data / "processed" / "index.json").exists()
    with sqlite3.connect(data / "mti.sqlite3") as con:
        assert con.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 1

    # second run: user now has models -> nothing changes
    (ck / "unet-x" / "best.pt").write_bytes(b"my own training")
    assert mod.seed(data, ck, ex, bundle=bundle)["status"] == "skipped"
    # --force adds missing files but keeps the user's files and database
    r = mod.seed(data, ck, ex, bundle=bundle, force=True)
    assert r["status"] == "copied" and r["database"] == "kept"
    assert (ck / "unet-x" / "best.pt").read_bytes() == b"my own training"


def test_missing_bundle_is_reported(tmp_path):
    assert _load().seed(tmp_path / "d", tmp_path / "c", tmp_path / "e", bundle=tmp_path / "none")["status"] \
        == "missing-bundle"


def test_real_bundle_gives_working_models_and_prediction(tmp_path):
    """Seeding the committed bundle yields trained models the API lists and can run on a bundled test image."""
    mod = _load()
    data, ck, ex = tmp_path / "data", tmp_path / "ck", tmp_path / "ex"
    assert mod.seed(data, ck, ex)["status"] == "copied"
    settings = Settings(data_dir=data, checkpoint_dir=ck, experiments_dir=ex, _env_file=None)
    with TestClient(create_app(settings)) as c:
        models = c.get("/api/models").json()
        models = models.get("models", models) if isinstance(models, dict) else models
        ids = [m["id"] for m in models if not m.get("is_fixture")]
        assert any(i.startswith("unet") for i in ids) and any(i.startswith("deeplabv3") for i in ids)
        evals = c.get("/api/evaluations").json()
        assert len(evals.get("evaluations", evals) if isinstance(evals, dict) else evals) > 0
        image = next((data / "raw").rglob("*.JPG"), None) or next((data / "raw").rglob("*.jpg"))
        with image.open("rb") as fh:
            r = c.post("/api/predict", data={"model_id": next(i for i in ids if i.startswith("unet"))},
                       files={"file": (image.name, fh, "image/jpeg")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("id") or body.get("analysis_id")
