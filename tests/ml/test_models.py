"""Model architecture, loss, checkpoint and inference tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.architectures.factory import build_model, count_parameters, load_checkpoint, resolve_device, save_checkpoint
from ml.classes import IGNORE_INDEX, NUM_CLASSES
from ml.inference.predictor import predict
from ml.training.losses import SegmentationLoss, class_weights_from_counts, soft_dice_loss

UNET_SMALL = {"base_channels": 8, "depth": 3}
DEEPLAB_OFFLINE = {"backbone": "mobilenet_v3_large", "pretrained": "none"}


@pytest.mark.parametrize("arch,cfg", [("unet", UNET_SMALL), ("deeplabv3", DEEPLAB_OFFLINE)])
def test_model_input_and_output_shape(arch, cfg):
    model = build_model(arch, cfg).eval()
    x = torch.randn(2, 3, 128, 160)                      # input: batch of 3-channel images
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, NUM_CLASSES, 128, 160)          # logits: four classes at input resolution
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()


def test_four_class_output():
    model = build_model("unet", UNET_SMALL).eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 64, 64))
    assert logits.shape[1] == 4
    probs = logits.softmax(1)
    assert torch.allclose(probs.sum(1), torch.ones(1, 64, 64), atol=1e-5)
    assert set(torch.unique(logits.argmax(1)).tolist()) <= {0, 1, 2, 3}


def test_unet_is_configurable_and_has_skip_connections():
    small, large = build_model("unet", {"base_channels": 8, "depth": 3}), build_model("unet", {"base_channels": 16, "depth": 4})
    assert count_parameters(large) > count_parameters(small)
    # each decoder stage consumes upsampled features concatenated with the encoder skip map
    up = small.ups[0]
    assert up.conv.block[0].in_channels == up.up[1].out_channels + small.downs[-2][1].block[0].out_channels
    with pytest.raises(ValueError):
        build_model("unet", {"depth": 9})


def test_unet_handles_sizes_not_divisible_by_stride():
    model = build_model("unet", UNET_SMALL).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 70, 90)).shape == (1, 4, 70, 90)


def test_unknown_architecture_and_option_rejected():
    with pytest.raises(ValueError):
        build_model("segformer")
    with pytest.raises(ValueError):
        build_model("unet", {"not_an_option": 1})


def test_loss_ignores_unlabelled_pixels():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 16, 16)
    target = torch.randint(0, 4, (2, 16, 16))
    target[:, :8] = IGNORE_INDEX
    crit = SegmentationLoss(dice_weight=0.5)
    base = crit(logits, target)
    changed = logits.clone()
    changed[:, :, :8] += 50.0                     # only touch logits under ignored pixels
    assert torch.allclose(base, crit(changed, target), atol=1e-6)
    assert not torch.allclose(base, crit(logits + torch.randn_like(logits), target))


def test_loss_gradient_is_zero_on_ignored_pixels():
    logits = torch.randn(1, 4, 8, 8, requires_grad=True)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :, 4:] = IGNORE_INDEX
    SegmentationLoss(dice_weight=1.0)(logits, target).backward()
    assert logits.grad[:, :, :, 4:].abs().sum() == 0
    assert logits.grad[:, :, :, :4].abs().sum() > 0


def test_loss_all_ignored_is_zero_and_differentiable():
    logits = torch.randn(1, 4, 8, 8, requires_grad=True)
    target = torch.full((1, 8, 8), IGNORE_INDEX, dtype=torch.long)
    loss = SegmentationLoss(dice_weight=0.5)(logits, target)
    assert float(loss) == 0.0
    loss.backward()


def test_loss_values():
    target = torch.randint(0, 4, (2, 12, 12))
    perfect = torch.nn.functional.one_hot(target, 4).permute(0, 3, 1, 2).float() * 40.0
    assert float(SegmentationLoss(0.0)(perfect, target)) < 1e-4
    assert float(soft_dice_loss(perfect, target)) < 1e-3
    uniform = torch.zeros(2, 4, 12, 12)
    assert float(SegmentationLoss(0.0)(uniform, target)) == pytest.approx(np.log(4), rel=1e-4)


def test_class_weights():
    assert class_weights_from_counts([10, 10, 10, 10], "none") is None
    w = class_weights_from_counts([1000, 1000, 1000, 10], "inverse_sqrt")
    assert w.shape == (4,) and w[3] > w[0] and float(w.max()) <= 8.0
    with pytest.raises(ValueError):
        class_weights_from_counts([1, 1, 1, 1], "bogus")


@pytest.mark.parametrize("arch,cfg", [("unet", UNET_SMALL), ("deeplabv3", DEEPLAB_OFFLINE)])
def test_checkpoint_save_and_load_roundtrip(tmp_path, arch, cfg):
    torch.manual_seed(1)
    model = build_model(arch, cfg).eval()
    path = tmp_path / "ckpt" / "best.pt"
    from ml.architectures.factory import normalise_model_config
    save_checkpoint(path, model, arch, normalise_model_config(arch, cfg), 128, {"epoch": 3})
    assert path.is_file() and not path.with_name("best.pt.tmp").exists()
    loaded, meta = load_checkpoint(path)
    assert meta["arch"] == arch and meta["resolution"] == 128 and meta["extra"]["epoch"] == 3
    assert meta["classes"] == ["soil", "bedrock", "sand", "big_rock"] and meta["ignore_index"] == 255
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-6)


def test_checkpoint_loading_safeguards(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")
    bad = tmp_path / "bad.pt"
    torch.save({"something": "else"}, bad)
    with pytest.raises(ValueError):
        load_checkpoint(bad)
    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"this is not a checkpoint")
    with pytest.raises(Exception):
        load_checkpoint(garbage)


def test_device_resolution():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            resolve_device("cuda")


def test_inference_pipeline_outputs():
    torch.manual_seed(2)
    model = build_model("unet", UNET_SMALL).eval()
    rgb = np.random.default_rng(0).integers(0, 255, (90, 150, 3), dtype=np.uint8)
    out = predict(model, 128, rgb, "cpu")
    assert out["class_map"].shape == (90, 150) and out["class_map"].dtype == np.uint8
    assert set(np.unique(out["class_map"]).tolist()) <= {0, 1, 2, 3}
    assert out["confidence"].shape == (90, 150)
    assert 0.25 - 1e-4 <= float(out["confidence"].min()) and float(out["confidence"].max()) <= 1.0 + 1e-6
    assert 0.0 <= float(out["uncertainty"].min()) and float(out["uncertainty"].max()) <= 1.0
    assert sum(out["mean_probs"].values()) == pytest.approx(1.0, abs=1e-3)
    assert out["input_size"][0] % 32 == 0 and out["input_size"][1] % 32 == 0
    assert max(out["input_size"]) == 128
    # large images are capped for output
    big = predict(model, 128, np.zeros((400, 800), dtype=np.uint8), "cpu", max_output_side=200)
    assert big["class_map"].shape == (100, 200)
