"""Command-line dataset acquisition (the web UI calls the same functions).

Examples
--------
Subset (recommended for laptops): ~3000 training pairs + the full expert test set
    python scripts/download_dataset.py --train-count 3000

All Curiosity NavCam training pairs via range requests (~6 GB)
    python scripts/download_dataset.py --train-count all

Full 16 GB archive, MD5-verified, then extract MSL NavCam
    python scripts/download_dataset.py --full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.datasets.acquire import download_full, download_subset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=str(ROOT / "data" / "raw"), help="Destination directory (default: data/raw)")
    ap.add_argument("--train-count", default="3000", help="Number of training pairs, or 'all'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full", action="store_true", help="Download the whole archive instead of a subset")
    args = ap.parse_args()

    last = {"msg": ""}

    def progress(stage: str, fraction: float, message: str) -> None:
        line = f"[{stage}] {fraction * 100:5.1f}%  {message}"
        if line != last["msg"]:
            print(line, flush=True)
            last["msg"] = line

    dest = Path(args.dest)
    if args.full:
        summary = download_full(dest, progress=progress)
    else:
        count = None if args.train_count == "all" else int(args.train_count)
        summary = download_subset(dest, count, seed=args.seed, progress=progress)
    print("Done:", summary)


if __name__ == "__main__":
    main()
