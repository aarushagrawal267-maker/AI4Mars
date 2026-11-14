"""Public-mode guard, frontend serving and the demo-bundle fetcher used for hosted deployments."""
from __future__ import annotations

import http.server
import importlib.util
import threading
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app

ROOT = Path(__file__).resolve().parents[2]


def make_settings(tmp_path, **kw):
    return Settings(data_dir=tmp_path / "data", checkpoint_dir=tmp_path / "ck", experiments_dir=tmp_path / "ex",
                    _env_file=None, **kw)


def test_public_mode_blocks_state_changes_without_admin_token(tmp_path):
    with TestClient(create_app(make_settings(tmp_path, public_mode=True, admin_token="s3cret"))) as c:
        assert c.get("/api/health").status_code == 200
        assert c.get("/api/system/info").json()["public_mode"] is True
        r = c.post("/api/training/start", json={"arch": "unet"})
        assert r.status_code == 403 and "public demo" in r.json()["detail"]
        assert c.post("/api/dataset/load").status_code == 403
        assert c.put("/api/settings/colors", json={"soil": "#000000"}).status_code == 403
        assert c.delete("/api/experiments/x").status_code == 403
        assert c.post("/api/training/start", json={"arch": "unet"}, headers={"X-Admin-Token": "wrong"}).status_code == 403
        # the admin token passes the guard (the request then fails normally: no dataset)
        assert c.post("/api/dataset/load", headers={"X-Admin-Token": "s3cret"}).status_code == 404
        # visitors may still run inference and planning (here: validation errors, not 403)
        assert c.post("/api/predict", data={"model_id": "none"}).status_code != 403


def test_public_mode_without_token_configured_locks_everything(tmp_path):
    with TestClient(create_app(make_settings(tmp_path, public_mode=True))) as c:
        assert c.post("/api/dataset/load", headers={"X-Admin-Token": ""}).status_code == 403


def test_serves_built_frontend_with_spa_fallback(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<title>MTI</title>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    (tmp_path / "secret.txt").write_text("no")
    with TestClient(create_app(make_settings(tmp_path, frontend_dist=dist))) as c:
        assert "MTI" in c.get("/").text
        assert "MTI" in c.get("/navigation").text          # client-side route
        assert c.get("/assets/app.js").text == "console.log(1)"
        assert "no" != c.get("/../secret.txt").text        # never escapes the dist folder
        assert c.get("/api/does-not-exist").status_code == 404


def _load_fetcher():
    spec = importlib.util.spec_from_file_location("fetch_demo_bundle", ROOT / "scripts" / "fetch_demo_bundle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fetch_demo_bundle_installs_only_allowed_folders(tmp_path, monkeypatch):
    bundle = tmp_path / "srv" / "b.zip"
    bundle.parent.mkdir()
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("checkpoints/m1/best.pt", b"weights")
        zf.writestr("data/processed/index.json", b"{}")
        zf.writestr("backend/app/main.py", b"evil")       # outside the allowed folders: ignored
        zf.writestr("data/../../escape.txt", b"evil")     # path traversal: rejected
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(bundle.parent), **k)  # noqa: E731
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    app_root = tmp_path / "app"
    (app_root / "checkpoints").mkdir(parents=True)
    mod = _load_fetcher()
    monkeypatch.setattr(mod, "ROOT", app_root)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("MTI_DEMO_BUNDLE_URL", f"http://127.0.0.1:{server.server_address[1]}/b.zip")
    try:
        with pytest.raises(ValueError):
            mod.main()
        # the traversal entry aborted extraction; a clean bundle installs fine
        with zipfile.ZipFile(bundle, "w") as zf:
            zf.writestr("checkpoints/m1/best.pt", b"weights")
            zf.writestr("backend/app/main.py", b"evil")
        assert mod.main() == 0
        assert (app_root / "checkpoints" / "m1" / "best.pt").read_bytes() == b"weights"
        assert not (app_root / "backend").exists() and not (tmp_path / "escape.txt").exists()
        monkeypatch.setenv("MTI_DEMO_BUNDLE_URL", "http://127.0.0.1:1/never-fetched.zip")
        assert mod.main() == 0                            # models present: download skipped
        monkeypatch.setenv("MTI_DEMO_BUNDLE_SHA256", "0" * 64)
        (app_root / "checkpoints" / "m1" / "best.pt").unlink()
        monkeypatch.setenv("MTI_DEMO_BUNDLE_URL", f"http://127.0.0.1:{server.server_address[1]}/b.zip")
        assert mod.main() == 1                            # checksum mismatch refused
    finally:
        server.shutdown()
