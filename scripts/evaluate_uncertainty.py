"""Uncertainty and leakage check for the test-set results: bootstrap intervals, a trivial baseline and a clean subset.

The app reports one mean IoU per model and split. This script re-evaluates trained checkpoints on the official
expert test set exactly as the app does (``ml.evaluation.evaluate``: prediction at the training resolution,
probabilities up-sampled to the native label resolution, unlabelled pixels ignored), but keeps one confusion matrix
per image so it can also report:

* the pooled metrics (identical to the stored evaluations; checked with ``--compare-db``),
* 95 % image-level bootstrap intervals for mean IoU and per-class IoU (images are resampled, not pixels,
  because neighbouring pixels are strongly correlated),
* a paired bootstrap interval for the mean-IoU difference between two models,
* a majority-class baseline (every labelled pixel predicted as the most frequent class),
* the scores on test images whose rover position (site + drive) does *not* occur in the training data, and
  on the ones that do.

The rover positions of the training data come from the local dataset index. The demo bundle installed by
``seed_demo_models.py`` only contains the test split, so for that case ``--zenodo-subset N`` rebuilds the list of
training positions that ``Download subset`` with ``train_count=N`` (seed ``--subset-seed``) picks, by reading only
the archive's file directory from Zenodo (a few MB, no images).

    python scripts/evaluate_uncertainty.py --compare-db                       # local index has training records
    python scripts/evaluate_uncertainty.py --zenodo-subset 4000 --compare-db   # demo bundle (baseline subset)
    python scripts/evaluate_uncertainty.py --zenodo-subset all                 # positions of the full training set

CPU time: about 3.5 minutes for DeepLabV3 and 7 minutes for U-Net on 4 threads (both label variants).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.classes import CLASS_KEYS, NUM_CLASSES  # noqa: E402
from ml.evaluation.metrics import metrics_from_confusion  # noqa: E402


# --------------------------------------------------------------------------------------------- statistics

def majority_baseline(cm: np.ndarray) -> dict:
    """Metrics of a model that predicts the most frequent ground-truth class for every pixel."""
    cm = np.asarray(cm, dtype=np.int64)
    support = cm.sum(axis=1)
    base = np.zeros_like(cm)
    base[:, int(np.argmax(support))] = support
    m = metrics_from_confusion(base)
    return {"class": CLASS_KEYS[int(np.argmax(support))], "pixel_accuracy": m["pixel_accuracy"],
            "mean_iou": m["mean_iou"], "mean_dice": m["mean_dice"]}


def _interval(values: list[float | None]) -> dict:
    vals = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if vals.size == 0:
        return {"low": None, "high": None, "defined_in": 0}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"low": float(lo), "high": float(hi), "defined_in": int(vals.size)}


def bootstrap(per_image: dict[str, np.ndarray], n_boot: int = 2000, seed: int = 0) -> dict:
    """Image-level percentile bootstrap (95 %) of mean IoU and per-class IoU, plus paired model differences.

    ``per_image`` maps a model name to an (images, C, C) array of confusion matrices for the *same* images in the
    same order. Each replicate resamples images with replacement and re-pools their confusion matrices, so the
    statistic is computed exactly like the reported one. A replicate in which a class does not occur averages over
    the classes that do (the same convention as the reported mean).
    """
    names = list(per_image)
    n = len(next(iter(per_image.values())))
    if any(len(v) != n for v in per_image.values()):
        raise ValueError("All models must be evaluated on the same images")
    rng = np.random.default_rng(seed)
    miou = {k: [] for k in names}
    cls = {k: {c: [] for c in CLASS_KEYS} for k in names}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for k in names:
            m = metrics_from_confusion(per_image[k][idx].sum(axis=0))
            miou[k].append(m["mean_iou"])
            for c in CLASS_KEYS:
                cls[k][c].append(m["per_class"][c]["iou"])
    out: dict = {"replicates": n_boot, "images": n, "seed": seed, "models": {}, "differences": {}}
    for k in names:
        out["models"][k] = {"mean_iou": _interval(miou[k]),
                            "per_class_iou": {c: _interval(cls[k][c]) for c in CLASS_KEYS}}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = [x - y for x, y in zip(miou[a], miou[b]) if x is not None and y is not None]
            out["differences"][f"{a} - {b}"] = {**_interval(d), "share_not_positive": float(np.mean(np.asarray(d) <= 0))}
    return out


# --------------------------------------------------------------------------------------------- data

def training_groups(index: dict, zenodo_subset: str | None, subset_seed: int) -> tuple[set[str], str]:
    """Rover positions (site-drive) of the training + validation data the models were built from."""
    local = {r["group"] for r in index["records"] if r["split"] in ("train", "val")}
    if local:
        return local, "local dataset index"
    if zenodo_subset is None:
        raise SystemExit("The local index has no training records (demo bundle?). Pass --zenodo-subset N with the "
                         "train_count used for the download (the baseline used 4000), or 'all'.")
    train_count = None if zenodo_subset == "all" else int(zenodo_subset)
    from ml.datasets.acquire import plan_subset
    from ml.datasets.naming import parse_msl_name
    from ml.datasets.remote_zip import RemoteZip
    from ml.datasets.sources import resolve_source

    source = resolve_source()
    entries, summary = plan_subset(RemoteZip(source.url).entries, train_count, subset_seed)
    groups = {parse_msl_name(Path(e.name).stem).group for e in entries if "/msl/ncam/labels/train/" in e.name}
    return groups, f"Zenodo directory, subset train_count={zenodo_subset} seed={subset_seed} ({summary['train_pairs']} pairs)"


def per_image_confusion(checkpoint: Path, index: dict, raw_root: Path, variants: list[str]) -> dict[str, np.ndarray]:
    """One confusion matrix per test image and label variant, computed as in ``ml.evaluation.evaluate``."""
    import torch
    import torch.nn.functional as F

    from ml.architectures.factory import load_checkpoint
    from ml.datasets.cache import read_gray, read_label
    from ml.evaluation.metrics import confusion_matrix
    from ml.preprocessing.transforms import prepare_inference_input

    model, meta = load_checkpoint(checkpoint)
    recs = [r for r in index["records"] if r["split"] == "test"]
    out = {v: np.zeros((len(recs), NUM_CLASSES, NUM_CLASSES), dtype=np.int64) for v in variants}
    with torch.no_grad():
        for i, r in enumerate(recs):
            tensor, _ = prepare_inference_input(read_gray(raw_root / r["image"]), meta["resolution"])
            probs = F.softmax(model(tensor).float(), dim=1)
            for v in variants:
                label = read_label(raw_root / r["labels"][v]["path"])
                pred = F.interpolate(probs, size=label.shape, mode="bilinear", align_corners=False).argmax(1)[0]
                out[v][i] = confusion_matrix(pred.numpy(), label)
    return out


# --------------------------------------------------------------------------------------------- main

def main() -> None:
    from backend.app.core.config import Settings

    settings = Settings()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--experiments", nargs="*", help="experiment ids (default: every checkpoints/*/best.pt that has "
                                                     "a stored test evaluation, else all)")
    ap.add_argument("--variants", nargs="*", default=["min3", "min1"])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zenodo-subset", default=None, metavar="N|all",
                    help="rebuild the training positions from the Zenodo file directory (demo bundle)")
    ap.add_argument("--subset-seed", type=int, default=42)
    ap.add_argument("--compare-db", action="store_true", help="check the pooled confusion matrices against the "
                                                              "evaluations stored in the database")
    ap.add_argument("--out", type=Path, default=None, help="write the full result as JSON")
    args = ap.parse_args()

    index = json.loads(settings.index_path.read_text(encoding="utf-8"))
    db_path = settings.data_dir / "mti.sqlite3"
    stored: dict = {}
    if db_path.is_file():
        with sqlite3.connect(db_path) as con:
            for exp, variant, metrics in con.execute(
                    "SELECT experiment_id, label_variant, metrics FROM evaluations WHERE split = 'test'"):
                stored[(exp, variant)] = json.loads(metrics)
    exps = args.experiments or sorted({e for e, _ in stored if (settings.checkpoint_dir / e / "best.pt").is_file()}) \
        or sorted(p.parent.name for p in settings.checkpoint_dir.glob("*/best.pt"))
    if not exps:
        raise SystemExit("No trained checkpoints found.")

    groups, groups_source = training_groups(index, args.zenodo_subset, args.subset_seed)
    test = [r for r in index["records"] if r["split"] == "test"]
    clean = np.array([r["group"] not in groups for r in test])
    print(f"Training positions from {groups_source}: {len(groups)}. Test images sharing a position with them: "
          f"{int((~clean).sum())} of {len(test)}.\n")

    cms = {e: per_image_confusion(settings.checkpoint_dir / e / "best.pt", index, settings.raw_dir, args.variants)
           for e in exps}
    pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}"  # noqa: E731
    result: dict = {"training_positions_source": groups_source, "test_images": len(test),
                    "test_images_sharing_position": int((~clean).sum()), "variants": {}}
    for v in args.variants:
        pooled = {e: metrics_from_confusion(cms[e][v].sum(axis=0)) for e in exps}
        support = next(iter(cms.values()))[v].sum(axis=0)
        labelled_images = int((next(iter(cms.values()))[v].sum(axis=(1, 2)) > 0).sum())
        boot = bootstrap({e: cms[e][v] for e in exps}, args.bootstrap, args.seed)
        base = majority_baseline(support)
        res = {"labelled_images": labelled_images,
               "images_with_class": {c: int((next(iter(cms.values()))[v][:, i, :].sum(axis=1) > 0).sum())
                                     for i, c in enumerate(CLASS_KEYS)},
               "majority_baseline": base, "bootstrap": boot, "models": {}}
        print(f"== test · {v}: {len(test)} images, {labelled_images} with labelled pixels; images containing each "
              f"class: {res['images_with_class']}")
        print(f"   majority-class baseline ({base['class']} everywhere): pixel acc. {pct(base['pixel_accuracy'])}, "
              f"mIoU {pct(base['mean_iou'])}")
        for e in exps:
            m = pooled[e]
            b = boot["models"][e]
            sub_clean = metrics_from_confusion(cms[e][v][clean].sum(axis=0))
            sub_shared = metrics_from_confusion(cms[e][v][~clean].sum(axis=0)) if (~clean).any() else None
            match = None
            if args.compare_db and (e, v) in stored:
                match = stored[(e, v)]["confusion_matrix"] == m["confusion_matrix"]
            res["models"][e] = {"pooled": {k: m[k] for k in ("pixel_accuracy", "mean_iou", "mean_dice")},
                                "per_class_iou": {c: m["per_class"][c]["iou"] for c in CLASS_KEYS},
                                "clean_subset": {"images": int(clean.sum()), "mean_iou": sub_clean["mean_iou"],
                                                 "pixel_accuracy": sub_clean["pixel_accuracy"]},
                                "shared_position_subset": None if sub_shared is None else {
                                    "images": int((~clean).sum()), "mean_iou": sub_shared["mean_iou"],
                                    "pixel_accuracy": sub_shared["pixel_accuracy"],
                                    "classes_present": sub_shared["classes_present"]},
                                "matches_stored_evaluation": match}
            ci = b["mean_iou"]
            br = b["per_class_iou"]["big_rock"]
            print(f"   {e}: mIoU {pct(m['mean_iou'])} [95% CI {pct(ci['low'])}-{pct(ci['high'])}], big-rock IoU "
                  f"{pct(m['per_class']['big_rock']['iou'])} [{pct(br['low'])}-{pct(br['high'])}]; "
                  f"without shared positions ({int(clean.sum())} img) mIoU {pct(sub_clean['mean_iou'])}"
                  + (f"; shared ({int((~clean).sum())} img) {pct(sub_shared['mean_iou'])}" if sub_shared else "")
                  + ("" if match is None else f"; matches stored evaluation: {match}"))
        for name, d in boot["differences"].items():
            print(f"   paired difference {name}: 95% CI {pct(d['low'])}-{pct(d['high'])} mIoU points "
                  f"(share of replicates <= 0: {d['share_not_positive']:.4f})")
        print()
        result["variants"][v] = res
    if args.out:
        args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
