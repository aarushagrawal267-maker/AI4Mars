"""Publish the browser edition (frontend/dist-demo) as a free static Hugging Face Space.

    python scripts/export_web_demo.py
    (cd frontend && VITE_DEMO=1 npx vite build)
    python scripts/deploy_hf_static.py --space <hf-username>/mars-terrain-intelligence-web

Static Spaces are free. The browser edition runs the trained models in the visitor's browser (WebAssembly), shows the
real results, experiments and dataset statistics, and includes the navigation simulator; training and evaluation need
the full app. Credentials: HF_TOKEN, or a huggingface.co network secret that adds the Authorization header.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "frontend" / "dist-demo"
README = """---
title: Mars Terrain Intelligence
emoji: 🪐
colorFrom: red
colorTo: gray
sdk: static
app_file: index.html
pinned: false
short_description: AI4Mars terrain segmentation running in your browser
---

# Mars Terrain Intelligence — browser edition

Pixel-level Mars terrain segmentation (soil, bedrock, sand, big rock). The U-Net and DeepLabV3 models trained on NASA
JPL's AI4Mars dataset (Zenodo doi:10.5281/zenodo.15995036, CC-BY-4.0) run in your browser with onnxruntime-web; nothing
is uploaded. Real evaluation results, experiments and dataset statistics; navigation simulator. Independent student
research project, not a NASA system. Full app (training, evaluation): https://github.com/aarushagrawal267-maker/AI4Mars
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True)
    a = ap.parse_args()
    index = DIST / "index.html"
    if not index.is_file() or not index.read_text(encoding="utf-8").lstrip().lower().startswith("<!doctype"):
        raise SystemExit("Build the browser edition first: cd frontend && VITE_DEMO=1 npx vite build "
                         "(do not run finalize_web_demo.py for Hugging Face; it needs a full HTML document)")
    stage = ROOT / "deploy" / "hf-static"
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(DIST, stage)
    (stage / "README.md").write_text(README, encoding="utf-8")
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    api.create_repo(a.space, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(repo_id=a.space, repo_type="space", folder_path=str(stage), commit_message="Publish browser edition")
    owner, name = a.space.split("/", 1)
    print(f"Space page: https://huggingface.co/spaces/{a.space}")
    print(f"App URL:    https://{owner.lower()}-{name.lower().replace('_', '-').replace('.', '-')}.static.hf.space")


if __name__ == "__main__":
    main()
