"""Print a Markdown results table built from the evaluations stored by the running backend.

    python scripts/results_table.py                 # all non-fixture evaluations
    python scripts/results_table.py --update-readme # also replace the table between TABLE markers in README.md

Numbers are read from the API (i.e. the SQLite database); nothing is typed by hand.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:8000/api"
ROOT = Path(__file__).resolve().parents[1]
CLASSES = [("soil", "Soil"), ("bedrock", "Bedrock"), ("sand", "Sand"), ("big_rock", "Big rock")]
ARCH = {"unet": "U-Net", "deeplabv3": "DeepLabV3"}


def get(path: str) -> dict:
    with urllib.request.urlopen(API + path, timeout=30) as r:
        return json.loads(r.read())


def pct(v: float | None, d: int = 1) -> str:
    return "n/a" if v is None else f"{v * 100:.{d}f}"


def build() -> str:
    evals = [e for e in get("/evaluations")["evaluations"] if not e["is_fixture"]]
    exps = {e["id"]: e for e in get("/experiments")["experiments"]}
    if not evals:
        return "*No evaluations have been run yet.*"
    evals.sort(key=lambda e: ({"test": 0, "val": 1}[e["split"]], e["label_variant"], e["arch"], e["id"]))
    lines = [
        "| Model | Split · labels | Images | Pixel acc. | mIoU | Mean Dice | " + " | ".join(f"IoU {n}" for _, n in CLASSES)
        + " | Forward pass (median) | Params | Checkpoint |",
        "|---|---|---:|---:|---:|---:|" + "---:|" * len(CLASSES) + "---:|---:|---:|",
    ]
    for e in evals:
        m = e["metrics"]
        split = f"{e['split']} · {e['label_variant']}" if e["split"] == "test" else "validation"
        lines.append(
            f"| {ARCH.get(e['arch'], e['arch'])} | {split} | {e['images']} | {pct(m['pixel_accuracy'])} | **{pct(m['mean_iou'])}** | "
            f"{pct(m['mean_dice'])} | " + " | ".join(pct(m["per_class"][k]["iou"]) for k, _ in CLASSES)
            + f" | {m['inference']['forward_ms_median']:.1f} ms ({m['inference']['device']}) | "
            f"{m['model']['parameters'] / 1e6:.2f} M | {m['model']['checkpoint_mb']:.1f} MB |"
        )
    used = {e["experiment_id"] for e in evals}
    lines += ["", "| Experiment | Architecture | Epochs run (best) | Best val mIoU | Train / val images | Resolution | Wall time |",
              "|---|---|---|---:|---|---|---:|"]
    for eid in sorted(used):
        x = exps.get(eid)
        if not x:
            continue
        c = x["config"]
        mins = (x["elapsed_seconds"] or 0) / 60
        lines.append(f"| {x['name']} | {ARCH.get(x['arch'], x['arch'])} {json.dumps(c.get('model_config', {}))} | "
                     f"{x['epochs_completed']}/{c['epochs']} ({x['best_epoch']}) | {pct(x['best_val_miou'])} | "
                     f"{x['n_train']} / {x['n_val']} | {c['resolution']} px | {mins:.0f} min ({x['device']}) |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-readme", action="store_true")
    args = ap.parse_args()
    table = build()
    print(table)
    if args.update_readme:
        readme = ROOT / "README.md"
        text = readme.read_text(encoding="utf-8")
        new = re.sub(r"<!-- TABLE:START -->.*?<!-- TABLE:END -->",
                     lambda _: f"<!-- TABLE:START -->\n{table}\n<!-- TABLE:END -->", text, flags=re.S)
        readme.write_text(new, encoding="utf-8")


if __name__ == "__main__":
    main()
