"""Deploy the full app (public demo mode) to a free Hugging Face Space, from the repository source.

    HF_TOKEN=hf_xxx python scripts/deploy_hf_space.py \\
        --space <hf-username>/mars-terrain-intelligence \\
        --bundle-url https://huggingface.co/datasets/<hf-username>/<dataset>/resolve/main/mti-demo-bundle.zip \\
        [--bundle-sha256 <hex>] [--wait]

The Space builds deploy/render.Dockerfile (website + API in one container). At start-up the container downloads
the trained models, results database and test images from --bundle-url (see scripts/fetch_demo_bundle.py).
Visitors can browse and analyse images; training, evaluation and deletion need the admin token, which is
stored as a Space secret and printed once (also saved to deploy/admin_token.txt, which Git ignores).
Use --dry-run to build the upload folder without contacting Hugging Face.
"""
from __future__ import annotations

import argparse
import os
import secrets
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "deploy" / "hf-space"
README = """---
title: Mars Terrain Intelligence
emoji: 🪐
colorFrom: red
colorTo: gray
sdk: docker
app_port: 10000
pinned: false
short_description: AI4Mars terrain segmentation with U-Net and DeepLabV3
---

# Mars Terrain Intelligence

Pixel-level Mars terrain segmentation (soil, bedrock, sand, big rock) trained on NASA JPL's AI4Mars dataset
(Zenodo doi:10.5281/zenodo.15995036, CC-BY-4.0). Independent student research project; not a NASA system.
Public demo mode: browse results, analyse images and plan routes; training and evaluation need the admin token.
Source: https://github.com/aarushagrawal267-maker/AI4Mars
"""
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "node_modules", "dist", "dist-demo", "public-demo",
                              "test-results", "playwright-report", ".vite")


def stage() -> Path:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    for name in ("backend", "ml", "scripts", "frontend"):
        shutil.copytree(ROOT / name, STAGE / name, ignore=SKIP)
    shutil.copy2(ROOT / "requirements.txt", STAGE / "requirements.txt")
    shutil.copy2(ROOT / "deploy" / "render.Dockerfile", STAGE / "Dockerfile")
    (STAGE / "README.md").write_text(README, encoding="utf-8")
    (STAGE / ".dockerignore").write_text("**/node_modules\n**/__pycache__\n", encoding="utf-8")
    n = sum(1 for p in STAGE.rglob("*") if p.is_file())
    print(f"staged {STAGE} ({n} files)")
    return STAGE


def space_url(space: str) -> str:
    owner, name = space.split("/", 1)
    slug = f"{owner}-{name}".lower().replace("_", "-").replace(".", "-")
    return f"https://{slug}.hf.space"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True, help="<hf-username>/<space-name>")
    ap.add_argument("--bundle-url", required=True, help="direct, public download link to mti-demo-bundle.zip")
    ap.add_argument("--bundle-sha256", default="")
    ap.add_argument("--wait", action="store_true", help="wait until the Space is running (or fails)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if "/" not in a.space:
        ap.error("--space must look like <hf-username>/<space-name>")
    folder = stage()
    if a.dry_run:
        print("dry run: nothing uploaded")
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN to a Hugging Face access token with Write permission.")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print("signed in as", api.whoami().get("name"))
    api.create_repo(a.space, repo_type="space", space_sdk="docker", exist_ok=True)
    variables = {"MTI_PUBLIC_MODE": "true", "MTI_DEMO_BUNDLE_URL": a.bundle_url, "MTI_CORS_ORIGINS": ""}
    if a.bundle_sha256:
        variables["MTI_DEMO_BUNDLE_SHA256"] = a.bundle_sha256
    for k, v in variables.items():
        api.add_space_variable(a.space, k, v)
    token_file = ROOT / "deploy" / "admin_token.txt"
    admin = token_file.read_text().strip() if token_file.is_file() else secrets.token_urlsafe(18)
    token_file.write_text(admin + "\n")
    api.add_space_secret(a.space, "MTI_ADMIN_TOKEN", admin)
    api.upload_folder(repo_id=a.space, repo_type="space", folder_path=str(folder), commit_message="Deploy Mars Terrain Intelligence")
    print(f"Space page: https://huggingface.co/spaces/{a.space}")
    print(f"App URL:    {space_url(a.space)}")
    print(f"Admin token (enter it in the app's Settings to unlock training): saved to {token_file}")
    if a.wait:
        last = ""
        for _ in range(180):  # up to ~30 minutes: the first build installs PyTorch
            stage_now = api.get_space_runtime(a.space).stage
            if stage_now != last:
                print("space status:", stage_now, flush=True)
                last = stage_now
            if stage_now in ("RUNNING",) or "ERROR" in stage_now:
                break
            time.sleep(10)


if __name__ == "__main__":
    main()
