"""Metrics, preprocessing, dataset index/validation, cache and the training loop."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from ml.classes import DEFAULT_COLORS, IGNORE_INDEX, hex_to_rgb, palette_array
from ml.datasets.cache import ensure_cache
from ml.datasets.index import build_index, load_index, save_index
from ml.datasets.layout import find_layout
from ml.datasets.naming import parse_msl_name
from ml.datasets.torch_dataset import TerrainDataset
from ml.evaluation.evaluate import evaluate_model
from ml.evaluation.metrics import confusion_matrix, metrics_from_confusion
from ml.inference.terrain import class_grid, terrain_statistics
from ml.inference.visualize import colorize, overlay
from ml.preprocessing.transforms import (ImageValidationError, augment_pair, decode_upload, gray_to_tensor,
                                         inference_size, prepare_inference_input)
from ml.training.config import TrainConfig
from ml.training.trainer import read_history, run_training, select_indices


# ------------------------------------------------------------------ metrics
def test_confusion_matrix_ignores_unlabelled():
    target = np.array([0, 0, 1, 1, 2, 3, 255, 255])
    pred = np.array([0, 1, 1, 1, 2, 0, 3, 3])
    cm = confusion_matrix(pred, target)
    assert cm.sum() == 6                                   # the two 255 pixels are not counted
    assert cm.tolist() == [[1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]


def test_metrics_known_values():
    cm = np.array([[1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]])
    m = metrics_from_confusion(cm)
    assert m["pixel_accuracy"] == pytest.approx(4 / 6)
    pc = m["per_class"]
    assert pc["soil"]["iou"] == pytest.approx(1 / 3)       # tp 1, gt 2, pred 2 -> 1 / (2 + 2 - 1)
    assert pc["bedrock"]["iou"] == pytest.approx(2 / 3)
    assert pc["sand"]["iou"] == pytest.approx(1.0)
    assert pc["big_rock"]["iou"] == pytest.approx(0.0)
    assert m["mean_iou"] == pytest.approx((1 / 3 + 2 / 3 + 1 + 0) / 4)
    assert pc["soil"]["dice"] == pytest.approx(0.5)
    assert pc["soil"]["precision"] == pytest.approx(0.5) and pc["soil"]["recall"] == pytest.approx(0.5)
    assert pc["bedrock"]["precision"] == pytest.approx(2 / 3) and pc["bedrock"]["recall"] == pytest.approx(1.0)
    assert pc["big_rock"]["precision"] is None             # never predicted: undefined, not zero
    assert m["mean_dice"] == pytest.approx((0.5 + 0.8 + 1.0 + 0.0) / 4)


def test_metrics_perfect_and_absent_class():
    m = metrics_from_confusion(np.diag([5, 7, 0, 3]))
    assert m["pixel_accuracy"] == 1.0 and m["mean_iou"] == 1.0
    assert m["per_class"]["sand"]["iou"] is None           # absent from ground truth and predictions
    assert m["classes_present"] == ["soil", "bedrock", "big_rock"]
    empty = metrics_from_confusion(np.zeros((4, 4), dtype=int))
    assert empty["pixel_accuracy"] is None and empty["mean_iou"] is None


def test_confusion_matrix_matches_sklearn():
    from sklearn.metrics import confusion_matrix as sk_cm, jaccard_score
    rng = np.random.default_rng(0)
    target, pred = rng.integers(0, 4, 5000), rng.integers(0, 4, 5000)
    cm = confusion_matrix(pred, target)
    assert (cm == sk_cm(target, pred, labels=[0, 1, 2, 3])).all()
    ours = [metrics_from_confusion(cm)["per_class"][k]["iou"] for k in ("soil", "bedrock", "sand", "big_rock")]
    assert np.allclose(ours, jaccard_score(target, pred, average=None, labels=[0, 1, 2, 3]))


# ------------------------------------------------------------ preprocessing
def test_image_preprocessing():
    gray = np.full((32, 48), 128, dtype=np.uint8)
    t = gray_to_tensor(gray)
    assert t.shape == (3, 32, 48) and t.dtype == torch.float32
    assert t[0, 0, 0] == pytest.approx((128 / 255 - 0.485) / 0.229, abs=1e-4)
    assert inference_size(1024, 1024, 256) == (256, 256)
    assert inference_size(480, 640, 256) == (192, 256)
    assert inference_size(10, 2000, 128) == (32, 128)
    batch, g = prepare_inference_input(np.zeros((480, 640, 3), dtype=np.uint8), 256)
    assert batch.shape == (1, 3, 192, 256) and g.shape == (480, 640)


def test_augmentation_preserves_label_values_and_shape():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (64, 64), dtype=np.uint8)
    lab = rng.choice(np.array([0, 1, 2, 3, 255], dtype=np.uint8), size=(64, 64))
    for _ in range(25):
        a, b = augment_pair(img, lab, rng)
        assert a.shape == (64, 64) and b.shape == (64, 64) and a.dtype == np.uint8
        assert set(np.unique(b).tolist()) <= {0, 1, 2, 3, 255}   # nearest-neighbour: no blended class ids


def test_decode_upload_validation():
    import io
    buf = io.BytesIO()
    Image.new("L", (50, 40), 100).save(buf, format="PNG")
    assert decode_upload(buf.getvalue()).size == (50, 40)
    for bad in (b"", b"not an image at all", buf.getvalue()[:40]):
        with pytest.raises(ImageValidationError):
            decode_upload(bad)
    tiny = io.BytesIO()
    Image.new("L", (8, 8)).save(tiny, format="PNG")
    with pytest.raises(ImageValidationError):
        decode_upload(tiny.getvalue())
    gif = io.BytesIO()
    Image.new("L", (64, 64)).save(gif, format="GIF")
    with pytest.raises(ImageValidationError):
        decode_upload(gif.getvalue())


def test_palette_and_colours():
    assert hex_to_rgb("#FF8000") == (255, 128, 0)
    with pytest.raises(ValueError):
        hex_to_rgb("orange")
    lut = palette_array({"soil": "#010203"})
    assert tuple(lut[0]) == (1, 2, 3) and tuple(lut[1]) == hex_to_rgb(DEFAULT_COLORS["bedrock"])
    cmap = np.array([[0, 1], [3, 255]], dtype=np.uint8)
    assert colorize(cmap).shape == (2, 2, 3)
    base = np.full((2, 2), 200, dtype=np.uint8)
    ov = overlay(base, cmap, alpha=0.5)
    assert tuple(ov[1, 1]) == (200, 200, 200)             # unlabelled pixels keep the original image


# ------------------------------------------------------- terrain statistics
def test_terrain_percentages_are_exact():
    cmap = np.zeros((10, 10), dtype=np.uint8)
    cmap[:, 5:8] = 1
    cmap[:, 8:] = 2
    conf = np.full((10, 10), 0.8, dtype=np.float32)
    stats = terrain_statistics(cmap, conf, np.zeros((10, 10), dtype=np.float32))
    assert stats["class_percent"] == {"soil": 50.0, "bedrock": 30.0, "sand": 20.0, "big_rock": 0.0}
    assert sum(stats["class_percent"].values()) == pytest.approx(100.0)
    assert stats["dominant_class"] == "soil" and stats["valid_pixels"] == 100
    assert stats["confidence"]["mean"] == pytest.approx(0.8, abs=1e-5) and stats["confidence"]["calibrated"] is False
    assert stats["spatial"]["soil"]["regions"] == 1 and stats["spatial"]["big_rock"]["centroid"] is None
    assert "soil" in stats["summary"]


def test_terrain_statistics_excludes_unlabelled():
    cmap = np.full((4, 4), IGNORE_INDEX, dtype=np.uint8)
    cmap[0, :2] = 3
    stats = terrain_statistics(cmap)
    assert stats["valid_pixels"] == 2 and stats["unlabeled_pixels"] == 14
    assert stats["class_percent"]["big_rock"] == 100.0


def test_class_grid_majority_vote():
    cmap = np.zeros((8, 8), dtype=np.uint8)
    cmap[:4, 4:] = 2
    cmap[4:, :4] = IGNORE_INDEX
    cmap[4:6, 4:] = 1
    cmap[6:, 4:] = 3
    grid = class_grid(cmap, 2, 2)
    assert grid["cells"] == [[0, 2], [-1, 1]] or grid["cells"] == [[0, 2], [-1, 3]]


# -------------------------------------------------------------- dataset index
def test_msl_filename_parsing():
    n = parse_msl_name("NLB_451023553EDR_F0310634NCAM00353M1")
    assert (n.camera, n.sclk, n.site, n.drive, n.parsed) == ("NLB", 451023553, "031", "0634", True)
    assert parse_msl_name("weird_name").parsed is False


def test_layout_detection(synthetic_dataset, tmp_path):
    layout = find_layout(synthetic_dataset["raw_root"])
    assert layout is not None and layout.images_dir.name == "edr"
    assert set(layout.test_label_dirs) == {"min1", "min3"}
    assert layout.label_keys["NAV"]["3"] == "big rock"
    assert find_layout(tmp_path / "nothing-here") is None


def test_index_pairs_splits_and_statistics(synthetic_dataset):
    idx = build_index(synthetic_dataset["raw_root"], val_fraction=0.25, seed=1)
    splits = {s: [r for r in idx["records"] if r["split"] == s] for s in ("train", "val", "test")}
    assert len(splits["test"]) == 4 and len(splits["train"]) + len(splits["val"]) == 24
    assert len(splits["val"]) >= 3
    # no rover position is shared between train and validation
    assert not ({r["group"] for r in splits["train"]} & {r["group"] for r in splits["val"]})
    assert idx["leakage_check"]["train_val_shared_positions"] == 0
    v = idx["validation"]
    assert v["valid_pairs"] == 28 and v["unreadable_count"] == 0 and v["unexpected_values_count"] == 0
    assert v["observed_label_values"] == [0, 1, 2, 3, 255]
    st = idx["statistics"]["splits"]["train"]
    assert st["images"] == len(splits["train"]) and 0 < st["unlabeled_percent"] < 100
    assert sum(st["class_percent_of_labeled"].values()) == pytest.approx(100.0, abs=0.01)
    # statistics are real pixel counts
    total = sum(sum(r["labels"]["train"]["counts"]) + r["labels"]["train"]["unlabeled"] for r in splits["train"])
    assert st["pixels_total"] == total == len(splits["train"]) * 64 * 64
    assert set(idx["statistics"]["splits"]["test"]) == {"min1", "min3"}
    assert (idx["statistics"]["splits"]["test"]["min3"]["unlabeled_percent"]
            > idx["statistics"]["splits"]["test"]["min1"]["unlabeled_percent"])


def test_index_detects_dataset_problems(synthetic_dataset):
    ds = synthetic_dataset
    ids = ds["train_ids"]
    (ds["images"] / f"{ids[0]}.JPG").unlink()                                   # label without image
    Image.fromarray(np.full((64, 64), 7, dtype=np.uint8)).save(ds["train_labels"] / f"{ids[1]}.png")  # bad class id
    (ds["train_labels"] / f"{ids[2]}.png").write_bytes(b"corrupt")               # unreadable label
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(ds["train_labels"] / f"{ids[3]}.png")    # wrong size
    Image.fromarray(np.full((64, 64), 255, dtype=np.uint8)).save(ds["train_labels"] / f"{ids[4]}.png")  # fully unlabelled
    Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(ds["images"] / "NLB_499999999EDR_F0990000NCAM00353M1.JPG")
    v = build_index(ds["raw_root"])["validation"]
    assert v["labels_without_image_count"] == 1 and v["labels_without_image"] == [f"{ids[0]}.png"]
    assert v["unexpected_values_count"] == 1 and v["unexpected_values"][0]["values"] == [7]
    assert v["unreadable_count"] == 1 and v["size_mismatch_count"] == 1
    assert v["empty_labels_count"] == 1 and v["images_without_label_count"] == 1
    assert v["valid_pairs"] == 28 - 5


def test_index_requires_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_index(tmp_path)


def test_index_save_load_and_cache(synthetic_dataset, tmp_path):
    idx = build_index(synthetic_dataset["raw_root"])
    path = tmp_path / "processed" / "index.json"
    save_index(idx, path)
    assert load_index(path)["fingerprint"] == idx["fingerprint"]
    assert load_index(tmp_path / "none.json") is None
    images, labels, ids = ensure_cache(idx, synthetic_dataset["raw_root"], tmp_path / "processed", 128, "train")
    n_train = sum(1 for r in idx["records"] if r["split"] == "train")
    assert images.shape == (n_train, 128, 128) and labels.shape == images.shape and len(ids) == n_train
    assert set(np.unique(labels).tolist()) <= {0, 1, 2, 3, 255}
    again, _, _ = ensure_cache(idx, synthetic_dataset["raw_root"], tmp_path / "processed", 128, "train")
    assert np.array_equal(images, again)                   # second call reuses the cache
    ds = TerrainDataset(images, labels, augment=True, seed=0)
    x, y = ds[0]
    assert x.shape == (3, 128, 128) and y.shape == (128, 128) and y.dtype == torch.int64
    with pytest.raises(ValueError):
        ensure_cache(idx, synthetic_dataset["raw_root"], tmp_path / "processed", 128, "test")


# ------------------------------------------------------------------ training
def test_train_config_validation():
    assert TrainConfig.from_dict({"arch": "unet", "epochs": 2}).model_config["base_channels"] == 32
    for bad in ({"arch": "vgg"}, {"resolution": 100}, {"epochs": 0}, {"batch_size": 0}, {"learning_rate": 5},
                {"arch": "deeplabv3", "batch_size": 1}, {"scheduler": "magic"}, {"surprise": True}):
        with pytest.raises(ValueError):
            TrainConfig.from_dict(bad)
    assert select_indices(10, None, 0) == list(range(10))
    assert select_indices(10, 4, 0) == select_indices(10, 4, 0) and len(select_indices(10, 4, 0)) == 4


def test_training_loop_writes_history_and_checkpoints(synthetic_dataset, tmp_path):
    idx = build_index(synthetic_dataset["raw_root"], val_fraction=0.25, seed=1)
    cfg = TrainConfig(arch="unet", model_config={"base_channels": 8, "depth": 3}, epochs=2, batch_size=4,
                      resolution=128, device="cpu", learning_rate=3e-3, early_stopping_patience=0)
    run_dir, ckpt_dir = tmp_path / "run", tmp_path / "ckpt"
    status = run_training(cfg, run_dir, ckpt_dir, idx, synthetic_dataset["raw_root"], tmp_path / "processed")
    assert status["state"] == "completed", status
    assert (ckpt_dir / "best.pt").is_file() and (ckpt_dir / "last.pt").is_file()
    hist = read_history(run_dir)
    assert [h["epoch"] for h in hist] == [1, 2]
    for h in hist:
        assert np.isfinite(h["train_loss"]) and np.isfinite(h["val_loss"])
        assert 0.0 <= h["val_mean_iou"] <= 1.0 and 0.0 <= h["val_pixel_accuracy"] <= 1.0
        assert set(h["val_per_class_iou"]) == {"soil", "bedrock", "sand", "big_rock"}
    assert json.loads((run_dir / "status.json").read_text())["has_checkpoint"] is True
    # the saved model can be evaluated on the held-out synthetic test split at native label resolution
    from ml.architectures.factory import load_checkpoint
    model, meta = load_checkpoint(ckpt_dir / "best.pt")
    m = evaluate_model(model, meta["resolution"], idx, synthetic_dataset["raw_root"], "test", "min1")
    assert m["images"] == 4 and m["evaluated_pixels"] > 0 and m["label_variant"] == "min1"
    assert np.array(m["confusion_matrix"]).shape == (4, 4)
    assert np.array(m["confusion_matrix"]).sum() == m["evaluated_pixels"]
    strict = evaluate_model(model, meta["resolution"], idx, synthetic_dataset["raw_root"], "test", "min3")
    assert strict["evaluated_pixels"] < m["evaluated_pixels"]      # stricter labels -> fewer scored pixels
    assert m["inference"]["forward_ms_mean"] > 0
    with pytest.raises(ValueError):
        evaluate_model(model, 128, idx, synthetic_dataset["raw_root"], "train")


def test_training_stop_request_keeps_state_consistent(synthetic_dataset, tmp_path):
    idx = build_index(synthetic_dataset["raw_root"], val_fraction=0.25, seed=1)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "control.json").write_text(json.dumps({"command": "stop"}))
    cfg = TrainConfig(arch="unet", model_config={"base_channels": 8, "depth": 3}, epochs=3, batch_size=4,
                      resolution=128, device="cpu")
    status = run_training(cfg, run_dir, tmp_path / "ckpt", idx, synthetic_dataset["raw_root"], tmp_path / "processed")
    assert status["state"] == "stopped" and status["has_checkpoint"] is False


def test_training_failure_is_recorded(synthetic_dataset, tmp_path):
    idx = build_index(synthetic_dataset["raw_root"])
    idx["records"] = [r for r in idx["records"] if r["split"] == "test"]      # no training data at all
    cfg = TrainConfig(arch="unet", model_config={"base_channels": 8, "depth": 3}, epochs=1, batch_size=2,
                      resolution=128, device="cpu")
    status = run_training(cfg, tmp_path / "run", tmp_path / "ckpt", idx, synthetic_dataset["raw_root"],
                          tmp_path / "processed")
    assert status["state"] == "failed" and status["error"]
