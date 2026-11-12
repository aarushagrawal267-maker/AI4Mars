"""REST API tests against an isolated app instance (temporary data directory, synthetic dataset).

Inference/evaluation tests use the UNTRAINED fixture model, which the API flags with
``is_fixture: true``. They verify plumbing and output validity, not model quality.
"""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from backend.app.services.model_registry import FIXTURE_ID
from tests.conftest import png_bytes, wait_for


def upload(client, data: bytes, name: str = "mars.png", model: str = FIXTURE_ID, ctype: str = "image/png"):
    return client.post("/api/predict", data={"model_id": model}, files={"file": (name, data, ctype)})


# ----------------------------------------------------------------- system
def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["app"] == "Mars Terrain Intelligence" and body["version"]


def test_system_info_reports_real_hardware(client):
    info = client.get("/api/system/info").json()
    assert info["hardware"]["selected_device"] in ("cpu", "cuda")
    assert info["hardware"]["cpu_cores_logical"] >= 1 and info["hardware"]["ram_total_gb"] > 0
    assert info["torch"] and info["paths"]["dataset_dir"].endswith("raw")
    assert info["test_fixture_enabled"] is True


def test_openapi_documents_core_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/api/health", "/api/system/info", "/api/dataset/status", "/api/dataset/load",
              "/api/dataset/statistics", "/api/models", "/api/training/start", "/api/training/stop",
              "/api/training/status", "/api/training/logs", "/api/predict", "/api/evaluate", "/api/experiments",
              "/api/experiments/{exp_id}", "/api/navigation/plan", "/api/navigation/costs"):
        assert p in paths, p


def test_empty_state_before_anything_is_loaded(client):
    ov = client.get("/api/overview").json()
    assert ov["dataset"]["indexed"] is False and ov["trained_model_count"] == 0
    assert ov["latest_experiment"] is None and ov["latest_analysis"] is None and ov["evaluations"] == []
    assert client.get("/api/dataset/statistics").status_code == 404
    assert client.get("/api/training/status").json()["state"] == "idle"
    assert client.post("/api/training/start", json={"arch": "unet", "epochs": 1}).status_code == 409


# ---------------------------------------------------------------- dataset
def test_dataset_load_validation_and_statistics(loaded_client):
    c = loaded_client
    st = c.get("/api/dataset/status").json()
    assert st["dataset_found"] and st["indexed"] and st["total_images"] == 28
    assert st["split_counts"]["test"] == 4 and st["split_counts"]["val"] >= 1
    assert st["layout"]["label_keys_nav"]["0"] == "soil"
    stats = c.get("/api/dataset/statistics").json()
    assert stats["classes"] == ["soil", "bedrock", "sand", "big_rock"] and stats["ignore_index"] == 255
    train = stats["statistics"]["splits"]["train"]
    assert train["images"] == st["split_counts"]["train"]
    assert sum(train["class_percent_of_labeled"].values()) == pytest.approx(100, abs=0.01)
    val = c.get("/api/dataset/validation").json()
    assert val["validation"]["valid_pairs"] == 28 and val["leakage_check"]["train_val_shared_positions"] == 0
    tree = c.get("/api/dataset/structure").json()["tree"]
    assert any(n["path"].endswith("images/edr") and n["files"] == 28 for n in tree)


def test_dataset_load_without_data_is_a_clear_error(settings):
    from fastapi.testclient import TestClient
    from backend.app.main import create_app
    with TestClient(create_app(settings)) as c:
        st = c.get("/api/dataset/status").json()
        assert st["dataset_found"] is False and st["indexed"] is False
        r = c.post("/api/dataset/load")
        assert r.status_code == 404 and "Download or import" in r.json()["detail"]


def test_dataset_samples_and_rendering(loaded_client):
    c = loaded_client
    page = c.get("/api/dataset/samples", params={"split": "test", "limit": 3}).json()
    assert page["total"] == 4 and len(page["items"]) == 3
    item = page["items"][0]
    assert item["variants"] == ["min1", "min3"] and item["width"] == 64
    for kind in ("image", "label", "overlay"):
        r = c.get(f"/api/dataset/samples/{item['id']}/{kind}", params={"variant": "min1"})
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        assert Image.open(io.BytesIO(r.content)).size == (64, 64)
    assert c.get("/api/dataset/samples/does-not-exist/image").status_code == 404
    assert c.get(f"/api/dataset/samples/{item['id']}/label", params={"variant": "nope"}).status_code == 404
    assert c.get("/api/dataset/samples", params={"split": "bogus"}).status_code == 422
    assert c.get("/api/dataset/samples", params={"contains": "big_rock"}).json()["total"] > 0


def test_dataset_import_rejects_paths_and_unknown_archives(client):
    for bad in ("../secret.zip", "C:\\Windows\\x.zip", "/etc/passwd", "sub/dir.zip"):
        assert client.post("/api/dataset/import", json={"archive": bad}).status_code == 422
    assert client.post("/api/dataset/import", json={"archive": "missing.zip"}).status_code == 404
    assert client.post("/api/dataset/download", json={"mode": "full"}).status_code == 400


# -------------------------------------------------------------- inference
def test_invalid_image_uploads_are_rejected(client):
    assert upload(client, b"this is not an image", "fake.png").status_code == 400
    assert upload(client, b"", "empty.png").status_code == 400
    assert upload(client, png_bytes(), "notes.txt", ctype="text/plain").status_code == 415
    assert upload(client, png_bytes(), "anim.gif", ctype="image/gif").status_code == 415
    truncated = png_bytes()[:60]
    assert upload(client, truncated, "cut.png").status_code == 400
    assert upload(client, png_bytes(), model="unknown-model").status_code == 404
    assert client.post("/api/predict", data={"model_id": FIXTURE_ID}).status_code == 400
    big = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (3000, 3000, 3), dtype=np.uint8)).save(big, format="PNG")
    assert len(big.getvalue()) > 20 * 1024 * 1024
    r = upload(client, big.getvalue(), "huge.png")
    assert r.status_code == 413 and "20 MB" in r.json()["detail"]


def test_prediction_response_and_output_dimensions(client):
    r = upload(client, png_bytes((96, 80)))
    assert r.status_code == 200, r.text
    a = r.json()
    assert a["is_fixture"] is True                         # never presented as a real prediction
    assert a["source"] == "upload" and a["filename"] == "mars.png"
    assert a["meta"]["output_size"] == [80, 96] and a["meta"]["original_size"] == [80, 96]
    s = a["stats"]
    assert s["valid_pixels"] == 80 * 96
    assert sum(s["class_pixels"].values()) == 80 * 96
    assert sum(s["class_percent"].values()) == pytest.approx(100.0, abs=0.01)
    assert s["dominant_class"] in ("soil", "bedrock", "sand", "big_rock")
    assert 0.25 <= s["confidence"]["mean"] <= 1.0 and s["confidence"]["calibrated"] is False
    assert sum(b["pixels"] for b in s["confidence"]["histogram"]) == 80 * 96
    assert 0.0 <= s["uncertainty"]["mean_entropy"] <= 1.0
    assert len(s["band_profile"]) == 12 and set(s["zone_composition"]) >= {"upper_left", "lower_right"}
    # stored class-id mask: single channel, same size as the image, only ids 0-3
    ids = Image.open(io.BytesIO(client.get(f"/api/analyses/{a['id']}/image/class_ids").content))
    assert ids.mode == "L" and ids.size == (96, 80) and set(np.unique(np.asarray(ids)).tolist()) <= {0, 1, 2, 3}
    counts = np.bincount(np.asarray(ids).ravel(), minlength=4)
    assert counts.tolist() == [s["class_pixels"][k] for k in ("soil", "bedrock", "sand", "big_rock")]
    for kind, size in (("mask", (96, 80)), ("overlay", (96, 80)), ("confidence", (96, 80)),
                       ("uncertainty", (96, 80)), ("original", (96, 80)), ("side_by_side", (96 * 3 + 16, 80))):
        img = client.get(f"/api/analyses/{a['id']}/image/{kind}")
        assert img.status_code == 200 and Image.open(io.BytesIO(img.content)).size == size, kind
    assert client.get(f"/api/analyses/{a['id']}/image/nonsense").status_code == 422
    dl = client.get(f"/api/analyses/{a['id']}/image/mask", params={"download": True})
    assert "attachment" in dl.headers["content-disposition"]
    assert client.get("/api/analyses").json()["analyses"][0]["id"] == a["id"]
    assert client.delete(f"/api/analyses/{a['id']}").status_code == 200
    assert client.get(f"/api/analyses/{a['id']}").status_code == 404


def test_predict_from_dataset_sample_and_jpeg(loaded_client):
    c = loaded_client
    sid = c.get("/api/dataset/samples", params={"split": "test"}).json()["items"][0]["id"]
    r = c.post("/api/predict", data={"model_id": FIXTURE_ID, "sample_id": sid})
    assert r.status_code == 200 and r.json()["source"] == "dataset" and r.json()["sample_id"] == sid
    assert c.post("/api/predict", data={"model_id": FIXTURE_ID, "sample_id": "nope"}).status_code == 404
    buf = io.BytesIO()
    Image.new("RGB", (120, 90), (150, 90, 60)).save(buf, format="JPEG")
    j = upload(c, buf.getvalue(), "colour.JPG", ctype="image/jpeg")
    assert j.status_code == 200 and j.json()["meta"]["color_input"] is True


def test_display_colours_change_rendering_but_not_class_ids(client):
    a = upload(client, png_bytes()).json()
    ids_before = client.get(f"/api/analyses/{a['id']}/image/class_ids").content
    mask_before = client.get(f"/api/analyses/{a['id']}/image/mask").content
    r = client.put("/api/settings/colors", json={"soil": "#00FF00", "bedrock": "#FF0000", "sand": "#0000FF",
                                                  "big_rock": "#FFFFFF"})
    assert r.status_code == 200 and r.json()["colors"]["soil"] == "#00FF00"
    assert [c["id"] for c in r.json()["classes"]] == [0, 1, 2, 3]
    assert client.get(f"/api/analyses/{a['id']}/image/mask").content != mask_before
    assert client.get(f"/api/analyses/{a['id']}/image/class_ids").content == ids_before
    assert client.put("/api/settings/colors", json={"soil": "green"}).status_code == 422
    assert client.post("/api/settings/colors/reset").json()["colors"]["soil"] != "#00FF00"


# ------------------------------------------------------------- navigation
def test_navigation_plan_endpoint(client):
    a = upload(client, png_bytes((128, 96))).json()
    grid = client.get(f"/api/analyses/{a['id']}/grid", params={"size": 32}).json()
    assert (grid["rows"], grid["cols"]) == (24, 32) and len(grid["cells"]) == 24
    body = {"analysis_id": a["id"], "grid_size": 32, "start": [22, 1], "goal": [2, 30]}
    r = client.post("/api/navigation/plan", json=body)
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["found"] and res["routes"][0]["kind"] == "terrain_aware"
    route = res["routes"][0]
    assert route["path"][0] == [22, 1] and route["path"][-1] == [2, 30]
    assert route["length"] > 0 and route["cost"] > 0 and route["steps"] == len(route["path"]) - 1
    assert sum(route["composition"].values()) == pytest.approx(100, abs=0.1)
    assert "not connected" in res["disclaimer"]
    wall = [[r_, 15] for r_ in range(24)]
    assert client.post("/api/navigation/plan", json={**body, "blocked": wall}).json()["found"] is False
    assert client.post("/api/navigation/plan", json={**body, "goal": [99, 99]}).status_code == 422
    assert client.post("/api/navigation/plan", json={**body, "goal": [22, 1]}).status_code == 422
    assert client.post("/api/navigation/plan", json={**body, "analysis_id": "missing"}).status_code == 404
    bad_cost = {k: {"cost": 1, "passable": True} for k in ("soil", "bedrock", "sand")}
    bad_cost["big_rock"] = {"cost": -3, "passable": True}
    assert client.post("/api/navigation/plan", json={**body, "costs": bad_cost}).status_code == 422


def test_navigation_costs_are_configurable_and_persist(client):
    d = client.get("/api/navigation/costs").json()
    assert all(v["passable"] for v in d["costs"].values())         # nothing is assumed impassable by default
    assert "not validated" in d["disclaimer"]
    new = {"soil": {"cost": 1.5, "passable": True}, "bedrock": {"cost": 3, "passable": True},
           "sand": {"cost": 9, "passable": False}, "big_rock": {"cost": 50, "passable": True}}
    assert client.put("/api/navigation/costs", json=new).status_code == 200
    saved = client.get("/api/navigation/costs").json()["costs"]
    assert saved["sand"] == {"cost": 9.0, "passable": False} and saved["soil"]["cost"] == 1.5
    assert client.put("/api/navigation/costs", json={**new, "soil": {"cost": 0, "passable": True}}).status_code == 422
    assert client.post("/api/navigation/costs/reset").json()["costs"]["sand"]["passable"] is True


# ------------------------------------------------------------- evaluation
def run_evaluation(c, **body) -> dict:
    r = c.post("/api/evaluate", json={"model_id": FIXTURE_ID, **body})
    assert r.status_code == 202, r.text
    job = wait_for(lambda: (lambda j: j if j["status"] not in ("queued", "running") else None)(
        c.get("/api/evaluate/status").json()["job"]))
    assert job["status"] == "completed", job
    return c.get(f"/api/evaluations/{job['result']['evaluation_id']}").json()


def test_evaluation_calculations_on_held_out_split(loaded_client):
    c = loaded_client
    ev = run_evaluation(c, split="test", label_variant="min1")
    m = ev["metrics"]
    assert ev["split"] == "test" and ev["label_variant"] == "min1" and ev["images"] == 4 and ev["is_fixture"] is True
    cm = np.array(m["confusion_matrix"])
    assert cm.shape == (4, 4) and cm.sum() == m["evaluated_pixels"] > 0
    # every reported number is re-derivable from the confusion matrix
    assert m["pixel_accuracy"] == pytest.approx(np.trace(cm) / cm.sum())
    for i, k in enumerate(("soil", "bedrock", "sand", "big_rock")):
        tp, gt, pd = cm[i, i], cm[i].sum(), cm[:, i].sum()
        pc = m["per_class"][k]
        assert pc["support_pixels"] == gt
        assert pc["iou"] == pytest.approx(tp / (gt + pd - tp)) if (gt + pd - tp) else pc["iou"] is None
        assert pc["recall"] == pytest.approx(tp / gt)
        if pd:
            assert pc["precision"] == pytest.approx(tp / pd) and pc["dice"] == pytest.approx(2 * tp / (gt + pd))
    ious = [m["per_class"][k]["iou"] for k in m["classes_present"]]
    assert m["mean_iou"] == pytest.approx(np.mean(ious))
    assert m["inference"]["forward_ms_mean"] > 0 and m["model"]["parameters"] > 0 and m["model"]["checkpoint_mb"] > 0
    # the unlabelled strip of every synthetic test label is excluded
    assert m["evaluated_pixels"] < 4 * 64 * 64
    strict = run_evaluation(c, split="test", label_variant="min3")
    assert strict["metrics"]["evaluated_pixels"] < m["evaluated_pixels"]
    val = run_evaluation(c, split="val")
    assert val["split"] == "val" and val["label_variant"] == "train-merged"

    listing = c.get("/api/evaluations").json()["evaluations"]
    assert len(listing) == 3
    ids = f"{ev['id']},{strict['id']}"
    csv_r = c.get("/api/evaluations/report", params={"ids": ids, "format": "csv"})
    assert csv_r.status_code == 200 and csv_r.headers["content-type"].startswith("text/csv")
    lines = csv_r.text.strip().splitlines()
    assert lines[0].startswith("evaluation_id,experiment_id") and any(",mean_iou," in l for l in lines)
    assert sum(1 for l in lines if ",confusion_matrix," in l) == 2 * 16
    js = json.loads(c.get("/api/evaluations/report", params={"ids": ids, "format": "json"}).text)
    assert len(js["evaluations"]) == 2 and js["evaluations"][0]["metrics"]["mean_iou"] == m["mean_iou"]
    assert c.get("/api/evaluations/report", params={"ids": "1;drop"}).status_code == 422


def test_evaluation_input_validation(loaded_client):
    c = loaded_client
    assert c.post("/api/evaluate", json={"model_id": "nope"}).status_code == 404
    assert c.post("/api/evaluate", json={"model_id": FIXTURE_ID, "split": "train"}).status_code == 422
    assert c.post("/api/evaluate", json={"model_id": FIXTURE_ID, "label_variant": "min9"}).status_code == 422


def test_evaluation_requires_dataset(client):
    r = client.post("/api/evaluate", json={"model_id": FIXTURE_ID})
    assert r.status_code == 404 and "No dataset" in r.json()["detail"]


# --------------------------------------------------- training + persistence
def test_training_request_validation(loaded_client):
    c = loaded_client
    assert c.post("/api/training/start", json={"arch": "resnet"}).status_code == 422
    assert c.post("/api/training/start", json={"arch": "unet", "resolution": 111}).status_code == 422
    assert c.post("/api/training/start", json={"arch": "unet", "epochs": 0}).status_code == 422
    assert c.post("/api/training/start", json={"arch": "deeplabv3", "batch_size": 1}).status_code == 422
    assert c.post("/api/training/estimate", json={"arch": "unet", "model_options": {"bogus": 1}}).status_code == 422
    for action in ("stop", "pause", "resume"):
        assert c.post(f"/api/training/{action}").status_code == 409
    d = c.get("/api/training/defaults").json()
    assert d["train_available"] > 0 and 128 in d["options"]["resolutions"]
    est = c.post("/api/training/estimate", json={"arch": "unet", "epochs": 1, "batch_size": 4, "resolution": 128,
                                                  "model_options": {"base_channels": 8, "depth": 3}}).json()
    assert est["seconds_per_step"] > 0 and est["estimated_total_seconds"] > 0 and est["parameters"] > 0


def test_training_run_end_to_end_and_database_persistence(loaded_client, settings):
    """Start a real (tiny) training run in a subprocess, then use the trained checkpoint."""
    c = loaded_client
    cfg = {"arch": "unet", "model_options": {"base_channels": 8, "depth": 3}, "epochs": 2, "batch_size": 4,
           "resolution": 128, "device": "cpu", "name": "pytest tiny run"}
    r = c.post("/api/training/start", json=cfg)
    assert r.status_code == 202, r.text
    exp_id = r.json()["experiment_id"]
    assert c.post("/api/training/start", json=cfg).status_code == 409           # one run at a time
    final = wait_for(lambda: (lambda s: s if not s["active"] else None)(c.get("/api/training/status").json()),
                     timeout=240, interval=1.0)
    assert final["state"] == "completed", (final, c.get("/api/training/logs").json())
    assert len(final["history"]) == 2 and final["has_checkpoint"]
    logs = c.get("/api/training/logs", params={"experiment_id": exp_id}).json()["lines"]
    assert any("Epoch 2/2" in line for line in logs)

    detail = wait_for(lambda: (lambda d: d if d["status"] == "completed" else None)(
        c.get(f"/api/experiments/{exp_id}").json()), timeout=15)
    assert detail["epochs_completed"] == 2 and detail["config"]["resolution"] == 128
    assert detail["best_val_miou"] is not None and detail["n_train"] > 0 and not detail["is_fixture"]
    models = c.get("/api/models").json()["models"]
    assert any(m["id"] == exp_id and not m["is_fixture"] for m in models)

    pred = upload(c, png_bytes(), model=exp_id)
    assert pred.status_code == 200 and pred.json()["is_fixture"] is False
    ev = c.post("/api/evaluate", json={"model_id": exp_id, "split": "test", "label_variant": "min1"})
    assert ev.status_code == 202
    wait_for(lambda: c.get("/api/evaluate/status").json()["job"]["status"] not in ("queued", "running"))
    assert c.get(f"/api/experiments/{exp_id}").json()["evaluations"][0]["metrics"]["mean_iou"] is not None
    assert c.patch(f"/api/experiments/{exp_id}", json={"notes": "kept"}).json()["notes"] == "kept"
    csv_report = c.get(f"/api/experiments/{exp_id}/report", params={"format": "csv"}).text.strip().splitlines()
    assert len(csv_report) == 3 and csv_report[0].startswith("experiment_id,arch,epoch")
    assert json.loads(c.get(f"/api/experiments/{exp_id}/report").text)["experiment"]["id"] == exp_id

    # a brand-new application instance on the same directories sees the same history (SQLite persistence)
    from fastapi.testclient import TestClient
    from backend.app.main import create_app
    with TestClient(create_app(settings)) as c2:
        again = c2.get(f"/api/experiments/{exp_id}").json()
        assert again["notes"] == "kept" and len(again["history"]) == 2 and len(again["evaluations"]) == 1
        assert c2.get("/api/analyses").json()["analyses"][0]["experiment_id"] == exp_id
        assert c2.get("/api/overview").json()["trained_model_count"] == 1
        assert c2.delete(f"/api/experiments/{exp_id}").status_code == 200
        assert c2.get(f"/api/experiments/{exp_id}").status_code == 404
        assert not any(m["id"] == exp_id for m in c2.get("/api/models").json()["models"])


def test_interrupted_runs_are_marked_on_restart(settings, synthetic_dataset):
    from fastapi.testclient import TestClient
    from backend.app.database.db import Database
    from backend.app.database.models import Experiment
    from backend.app.main import create_app
    settings.ensure_dirs()
    db = Database(settings.db_url)
    with db.session() as s:
        s.add(Experiment(id="unet-orphan", name="orphan", arch="unet", status="running", config={"epochs": 5}))
    db.dispose()
    with TestClient(create_app(settings)) as c:
        exp = c.get("/api/experiments/unet-orphan").json()
        assert exp["status"] == "interrupted" and "server stopped" in exp["message"].lower()
