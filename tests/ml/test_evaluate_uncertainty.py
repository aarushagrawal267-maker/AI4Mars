"""Statistics helpers of scripts/evaluate_uncertainty.py (bootstrap intervals and the majority-class baseline)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("evaluate_uncertainty", ROOT / "scripts" / "evaluate_uncertainty.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_majority_baseline_predicts_the_most_frequent_class():
    mod = _load()
    cm = np.array([[5, 1, 0, 0], [1, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])  # support 6 / 3 / 1 / 0
    base = mod.majority_baseline(cm)
    assert base["class"] == "soil"
    assert base["pixel_accuracy"] == pytest.approx(0.6)
    # soil IoU 6/10, bedrock and sand 0; big rock is absent from the ground truth and not averaged
    assert base["mean_iou"] == pytest.approx(0.2)


def test_bootstrap_intervals_and_paired_difference():
    mod = _load()
    rng = np.random.default_rng(1)
    n = 40
    perfect = np.zeros((n, 4, 4), dtype=np.int64)
    noisy = np.zeros((n, 4, 4), dtype=np.int64)
    for i in range(n):
        counts = rng.integers(10, 100, 3)
        for c in range(3):
            perfect[i, c, c] = counts[c]
            wrong = int(rng.integers(1, counts[c]))
            noisy[i, c, c] = counts[c] - wrong
            noisy[i, c, (c + 1) % 3] = wrong
    out = mod.bootstrap({"a": perfect, "b": noisy}, n_boot=200, seed=3)
    assert out["models"]["a"]["mean_iou"]["low"] == pytest.approx(1.0)
    assert out["models"]["a"]["mean_iou"]["high"] == pytest.approx(1.0)
    pooled_b = mod.metrics_from_confusion(noisy.sum(axis=0))["mean_iou"]
    ci_b = out["models"]["b"]["mean_iou"]
    assert ci_b["low"] < pooled_b < ci_b["high"]
    # big rock never occurs or is predicted: its IoU is undefined in every replicate
    assert out["models"]["b"]["per_class_iou"]["big_rock"]["defined_in"] == 0
    diff = out["differences"]["a - b"]
    assert diff["low"] > 0 and diff["share_not_positive"] == 0.0
    # same seed, same result
    assert mod.bootstrap({"a": perfect, "b": noisy}, n_boot=200, seed=3) == out


def test_bootstrap_requires_the_same_images_for_every_model():
    mod = _load()
    with pytest.raises(ValueError):
        mod.bootstrap({"a": np.zeros((3, 4, 4), dtype=np.int64), "b": np.zeros((2, 4, 4), dtype=np.int64)}, n_boot=5)
