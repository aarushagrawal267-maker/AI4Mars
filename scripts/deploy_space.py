"""Package the app with its trained models and deploy it as a public Hugging Face Space.

    python scripts/deploy_space.py --stage                       # build deploy/stage (no upload)
    HF_TOKEN=hf_xxx python scripts/deploy_space.py --push --space <user>/mars-terrain-intelligence

What goes into the public bundle (nothing else from your machine is uploaded):
  * application code (backend/, ml/) and the pre-built website (frontend/dist)
  * best.pt checkpoints of every experiment, their logs/history, and the SQLite database
    (experiments, evaluations, analyses, settings)
  * the official expert-labelled TEST images and labels only (plus the dataset docs) and a dataset
    index trimmed to those records. Training images stay local: training is disabled in public mode.

The Space runs with MTI_PUBLIC_MODE=true. An admin token is stored as a Space secret
(MTI_ADMIN_TOKEN); entering it in Settings unlocks training, evaluation and deletion.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "deploy" / "stage"
SPACE_README = """---
title: Mars Terrain Intelligence
emoji: 🪐
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: AI4Mars terrain segmentation with U-Net and DeepLabV3
---

# Mars Terrain Intelligence — public demo

Pixel-level Mars terrain segmentation (soil, bedrock, sand, big rock) trained on NASA JPL's AI4Mars
dataset (Zenodo 10.5281/zenodo.15995036, CC-BY-4.0). Independent student research project; not a NASA system.

This demo ships the trained models, their evaluations and the 322 expert-labelled test images.
Training, dataset changes, evaluation and deletion are disabled for visitors.
Source code and full instructions: https://github.com/aarushagrawal267-maker/AI4Mars
"""


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def stage() -> Path:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    # code
    for name in ("backend", "ml", "requirements.txt"):
        copy(ROOT / name, STAGE / name)
    copy(ROOT / "deploy" / "Dockerfile", STAGE / "Dockerfile")
    (STAGE / "README.md").write_text(SPACE_README, encoding="utf-8")
    (STAGE / ".gitattributes").write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n*.JPG filter=lfs diff=lfs merge=lfs -text\n"
                                          "*.png filter=lfs diff=lfs merge=lfs -text\n*.sqlite3 filter=lfs diff=lfs merge=lfs -text\n")
    # website
    print("building frontend…")
    npm = "npm.cmd" if os.name == "nt" else "npm"
    subprocess.run([npm, "run", "build"], cwd=ROOT / "frontend", check=True)
    copy(ROOT / "frontend" / "dist", STAGE / "frontend_dist")

    data = ROOT / "data"
    # dataset: test split only
    index = json.loads((data / "processed" / "index.json").read_text(encoding="utf-8"))
    tests = [r for r in index["records"] if r["split"] == "test"]
    raw = data / "raw"
    for r in tests:
        copy(raw / r["image"], STAGE / "data" / "raw" / r["image"])
        for lab in r["labels"].values():
            copy(raw / lab["path"], STAGE / "data" / "raw" / lab["path"])
    base = Path(index["layout"]["base"])
    (STAGE / "data" / "raw" / base / "labels" / "train").mkdir(parents=True, exist_ok=True)  # layout marker
    for doc in ("info.md", "label_keys.json", "changelog.md"):
        top = raw / base.parts[0] / doc
        if top.is_file():
            copy(top, STAGE / "data" / "raw" / base.parts[0] / doc)
    index["records"] = tests
    index["demo_note"] = "Public demo bundle: only the official test split is included; statistics describe the full local index."
    (STAGE / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (STAGE / "data" / "processed" / "index.json").write_text(json.dumps(index), encoding="utf-8")
    # database (consistent copy) and analyses
    dst_db = STAGE / "data" / "mti.sqlite3"
    with sqlite3.connect(data / "mti.sqlite3") as src, sqlite3.connect(dst_db) as dst:
        src.backup(dst)
    with sqlite3.connect(dst_db) as db:
        db.execute("PRAGMA journal_mode=DELETE")
    if (data / "analyses").is_dir():
        copy(data / "analyses", STAGE / "data" / "analyses")
    # models and run records
    for ck in (ROOT / "checkpoints").glob("*/best.pt"):
        if ck.parent.name.startswith("fixture"):
            continue
        copy(ck, STAGE / "checkpoints" / ck.parent.name / "best.pt")
    for run in (ROOT / "experiments").iterdir():
        if run.is_dir():
            for f in ("train.log", "history.jsonl", "status.json"):
                if (run / f).is_file():
                    copy(run / f, STAGE / "experiments" / run.name / f)
    size = sum(p.stat().st_size for p in STAGE.rglob("*") if p.is_file())
    print(f"staged {STAGE} ({size / 1e6:.0f} MB, {len(tests)} test images)")
    return STAGE


def bundle_zip() -> Path:
    """Zip only data/, checkpoints/ and experiments/ of the staged bundle (for Render's MTI_DEMO_BUNDLE_URL)."""
    import hashlib
    import zipfile
    out = ROOT / "deploy" / "mti-demo-bundle.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
        for top in ("data", "checkpoints", "experiments"):
            for p in sorted((STAGE / top).rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(STAGE).as_posix())
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (out.with_suffix(".sha256")).write_text(digest + "\n")
    print(f"bundle {out} ({out.stat().st_size / 1e6:.0f} MB) sha256 {digest}")
    return out


def push(space: str) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN to a Hugging Face access token with write permission.")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("pip install huggingface_hub")
    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    admin_file = ROOT / "deploy" / "admin_token.txt"
    admin = admin_file.read_text().strip() if admin_file.is_file() else secrets.token_urlsafe(18)
    admin_file.write_text(admin)
    api.add_space_secret(space, "MTI_ADMIN_TOKEN", admin)
    api.upload_folder(repo_id=space, repo_type="space", folder_path=str(STAGE), commit_message="Deploy Mars Terrain Intelligence")
    owner, name = space.split("/")
    print(f"Space: https://huggingface.co/spaces/{space}")
    print(f"App URL (after the build finishes): https://{owner}-{name}.hf.space".lower().replace("_", "-"))
    print(f"Admin token saved to {admin_file}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--space", help="<hf-username>/<space-name>")
    ap.add_argument("--bundle", action="store_true", help="also write deploy/mti-demo-bundle.zip (for Render)")
    a = ap.parse_args()
    if not (a.stage or a.push or a.bundle):
        ap.error("use --stage, --bundle and/or --push")
    if a.stage or not STAGE.exists():
        stage()
    if a.bundle:
        bundle_zip()
    if a.push:
        if not a.space:
            ap.error("--push needs --space <user>/<name>")
        push(a.space)


if __name__ == "__main__":
    main()
