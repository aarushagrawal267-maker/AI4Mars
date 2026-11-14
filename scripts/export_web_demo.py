"""Export the data for the browser edition (static build that runs the trained models in the browser).

    python scripts/export_web_demo.py [--samples 120] [--out frontend/public-demo]

Writes, under ``--out`` (served next to the static build):
  demo/data.json         models, experiments (with per-epoch history and log tail), evaluations, dataset statistics,
                         validation report and folder tree — all read from the local database and dataset index
  demo/samples.json      the exported test samples and their ground-truth label composition
  demo/samples/*         384-px test images, colour-coded ground truth and overlays (min1 expert labels)
  demo/models/*.txt      the trained models as ONNX, Base64-encoded in parts of <= 14 MB (the page host serves
                         text but not arbitrary binary files); models.json lists the parts
  ort/                   onnxruntime-web 1.17.3 (script + WebAssembly), copied from its npm package
Nothing is computed for display here except image resizing: every number comes from the stored results.
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.architectures.factory import load_checkpoint  # noqa: E402
from ml.classes import DEFAULT_COLORS  # noqa: E402
from ml.datasets.cache import read_gray, read_label  # noqa: E402
from ml.datasets.layout import directory_tree  # noqa: E402
from ml.inference.visualize import colorize, overlay  # noqa: E402

ORT_VERSION = "1.17.3"
PART_BYTES = 14 * 1024 * 1024  # Base64 text per part, under the host's 16 MB text-file limit
SIZE = 384


def export_models(out: Path, experiments: list[dict]) -> list[dict]:
    mdir = out / "demo" / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    listing = []
    for exp in experiments:
        ck = ROOT / "checkpoints" / exp["id"] / "best.pt"
        if exp["is_fixture"] or not ck.is_file() or exp["status"] != "completed":
            continue
        model, meta = load_checkpoint(ck)
        res = int(meta["resolution"])
        with tempfile.TemporaryDirectory() as td:
            onnx_path = Path(td) / "m.onnx"
            torch.onnx.export(model, torch.zeros(1, 3, res, res), onnx_path, input_names=["input"], output_names=["logits"],
                              opset_version=13, dynamic_axes={"input": {2: "h", 3: "w"}, "logits": {2: "h", 3: "w"}},
                              dynamo=False)
            b64 = base64.b64encode(onnx_path.read_bytes()).decode()
            raw_bytes = onnx_path.stat().st_size
        parts = []
        for i in range(0, len(b64), PART_BYTES):
            name = f"{exp['id']}.part{len(parts)}.txt"
            (mdir / name).write_text(b64[i:i + PART_BYTES])
            parts.append(f"demo/models/{name}")
        listing.append({"id": exp["id"], "arch": exp["arch"], "resolution": res, "parts": parts, "onnx_bytes": raw_bytes,
                        "parameters": meta["parameters"]})
        print(f"model {exp['id']}: {raw_bytes / 1e6:.1f} MB onnx, {len(parts)} part(s)")
    (mdir / "models.json").write_text(json.dumps(listing, indent=1))
    return listing


def export_samples(out: Path, index: dict, n: int) -> list[dict]:
    raw = ROOT / "data" / "raw"
    sdir = out / "demo" / "samples"
    sdir.mkdir(parents=True, exist_ok=True)
    tests = [r for r in index["records"] if r["split"] == "test"]
    rng = random.Random(7)
    # always keep the test images that contain big rocks (rare), fill the rest at random
    with_rock = [r for r in tests if r["labels"].get("min1", {}).get("counts", [0, 0, 0, 0])[3] > 0]
    rest = [r for r in tests if r not in with_rock]
    chosen = (with_rock + rng.sample(rest, max(0, n - len(with_rock))))[:n]
    chosen.sort(key=lambda r: r["sclk"])
    lut_colors = dict(DEFAULT_COLORS)
    items = []
    for r in chosen:
        img = read_gray(raw / r["image"])
        small = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(sdir / f"{r['id']}_image.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        labels = {}
        for variant in ("min1",):  # one label variant keeps the page under the host's 511-file limit
            if variant not in r["labels"]:
                continue
            lab = cv2.resize(read_label(raw / r["labels"][variant]["path"]), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(sdir / f"{r['id']}_label_{variant}.png"), cv2.cvtColor(colorize(lab, lut_colors), cv2.COLOR_RGB2BGR))
            ov = overlay(small, lab, lut_colors, 0.5)
            cv2.imwrite(str(sdir / f"{r['id']}_overlay_{variant}.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
            c = r["labels"][variant]["counts"]
            labeled = sum(c)
            total = labeled + r["labels"][variant]["unlabeled"]
            labels[variant] = {
                "class_percent": {k: (round(100 * c[i] / labeled, 2) if labeled else 0.0) for i, k in enumerate(DEFAULT_COLORS)},
                "unlabeled_percent": round(100 * r["labels"][variant]["unlabeled"] / total, 2) if total else 0.0,
            }
        items.append({"id": r["id"], "split": "test", "width": r["width"], "height": r["height"], "camera": r["camera"],
                      "site": r["site"], "drive": r["drive"], "sclk": r["sclk"], "variants": sorted(labels), "labels": labels})
    (out / "demo" / "samples.json").write_text(json.dumps(items))
    print(f"samples: {len(items)} of {len(tests)} test images ({len(with_rock)} contain big rocks)")
    return items


def db_rows(query: str) -> list[dict]:
    with sqlite3.connect(ROOT / "data" / "mti.sqlite3") as db:
        db.row_factory = sqlite3.Row
        return [dict(r) for r in db.execute(query)]


def export_data(out: Path, index: dict, models: list[dict], n_samples: int) -> None:
    exps = db_rows("SELECT * FROM experiments WHERE is_fixture = 0 ORDER BY created_at DESC")
    evals = db_rows("SELECT * FROM evaluations ORDER BY created_at DESC")
    for e in exps:
        for k in ("config", "history"):
            e[k] = json.loads(e[k]) if isinstance(e[k], str) else e[k]
        e["is_fixture"] = bool(e["is_fixture"])
        e["pretrained_loaded"] = None if e["pretrained_loaded"] is None else bool(e["pretrained_loaded"])
        log = ROOT / "experiments" / e["id"] / "train.log"
        e["log"] = log.read_text(encoding="utf-8", errors="replace").splitlines()[-400:] if log.is_file() else []
        e["has_checkpoint"] = (ROOT / "checkpoints" / e["id"] / "best.pt").is_file()
        e["browser_model"] = any(m["id"] == e["id"] for m in models)
    names = {e["id"]: e for e in exps}
    out_evals = []
    for v in evals:
        exp = names.get(v["experiment_id"])
        if exp is None:
            continue
        out_evals.append({"id": v["id"], "experiment_id": v["experiment_id"], "experiment_name": exp["name"], "arch": exp["arch"],
                          "is_fixture": False, "created_at": v["created_at"], "split": v["split"],
                          "label_variant": v["label_variant"], "images": v["images"], "metrics": json.loads(v["metrics"])})
    data = {
        "exported_note": "Exported from the local Mars Terrain Intelligence database and dataset index.",
        "experiments": exps,
        "evaluations": out_evals,
        "dataset": {
            "created": index["created"], "fingerprint": index["fingerprint"], "classes": index["classes"],
            "ignore_index": index["ignore_index"], "split_config": index["split_config"], "statistics": index["statistics"],
            "leakage_check": index["leakage_check"], "layout": index["layout"], "validation": index["validation"],
            "scan_seconds": index["scan_seconds"],
            "split_counts": {s: sum(1 for r in index["records"] if r["split"] == s) for s in ("train", "val", "test")},
            "samples_included": n_samples,
        },
        "tree": directory_tree(ROOT / "data" / "raw"),
        "colors": DEFAULT_COLORS,
    }
    (out / "demo" / "data.json").write_text(json.dumps(data))
    print(f"data: {len(exps)} experiments, {len(out_evals)} evaluations")


def export_ort(out: Path) -> None:
    odir = out / "ort"
    odir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["npm", "pack", f"onnxruntime-web@{ORT_VERSION}", "--silent"], cwd=td, check=True, capture_output=True)
        with tarfile.open(next(Path(td).glob("*.tgz"))) as tf:
            for name in ("ort.wasm.min.js", "ort-wasm-simd.wasm", "ort-wasm.wasm"):
                member = tf.getmember(f"package/dist/{name}")
                with tf.extractfile(member) as src, open(odir / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    print(f"onnxruntime-web {ORT_VERSION} copied")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--out", default=str(ROOT / "frontend" / "public-demo"))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    index = json.loads((ROOT / "data" / "processed" / "index.json").read_text(encoding="utf-8"))
    exps = db_rows("SELECT id, arch, status, is_fixture FROM experiments")
    models = export_models(out, exps)
    samples = export_samples(out, index, a.samples)
    export_data(out, index, models, len(samples))
    export_ort(out)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"wrote {out} ({size / 1e6:.0f} MB)")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
