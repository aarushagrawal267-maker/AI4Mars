"""Download and unpack the demo bundle (trained checkpoints, results database, AI4Mars test split)
at container start, for hosts where large files are not part of the Git repository (e.g. Render).

Reads MTI_DEMO_BUNDLE_URL (a direct download link to the zip made by `scripts/deploy_space.py --bundle`)
and optional MTI_DEMO_BUNDLE_SHA256. Does nothing when no URL is set or when models are already present.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.datasets.remote_zip import safe_join  # noqa: E402

ALLOWED_TOP = {"data", "checkpoints", "experiments"}


def main() -> int:
    url = os.environ.get("MTI_DEMO_BUNDLE_URL", "").strip()
    if not url:
        print("[bundle] MTI_DEMO_BUNDLE_URL not set; starting without a demo bundle")
        return 0
    if any((ROOT / "checkpoints").glob("*/best.pt")):
        print("[bundle] checkpoints already present; skipping download")
        return 0
    print(f"[bundle] downloading {url}")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        req = urllib.request.Request(url, headers={"User-Agent": "mars-terrain-intelligence"})
        with urllib.request.urlopen(req, timeout=600) as r:
            shutil.copyfileobj(r, tmp, length=4 * 1024 * 1024)
        path = Path(tmp.name)
    expected = os.environ.get("MTI_DEMO_BUNDLE_SHA256", "").strip().lower()
    if expected:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != expected:
            print("[bundle] SHA-256 mismatch; refusing to use the bundle", file=sys.stderr)
            return 1
    with zipfile.ZipFile(path) as zf:
        for m in zf.infolist():
            if m.is_dir():
                continue
            top = m.filename.split("/", 1)[0]
            if top not in ALLOWED_TOP:
                continue  # only data/, checkpoints/ and experiments/ are accepted
            target = safe_join(ROOT, m.filename)  # rejects paths escaping the app folder
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    path.unlink(missing_ok=True)
    print("[bundle] installed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
