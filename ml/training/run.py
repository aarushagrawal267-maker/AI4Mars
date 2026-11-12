"""Entry point for a training run executed as its own process.

    python -m ml.training.run --run-dir experiments/<id>

``<run-dir>/job.json`` (written by the API or by hand) holds the training
configuration and the data locations:

    {"config": {...TrainConfig...}, "raw_root": "...", "processed_root": "...",
     "index_path": "...", "checkpoint_dir": "..."}
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ..datasets.index import load_index
from .config import TrainConfig
from .trainer import RunIO, run_training


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one Mars Terrain Intelligence training experiment")
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.addHandler(logging.StreamHandler(sys.stdout))
    try:
        job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
        cfg = TrainConfig.from_dict(job["config"])
        index = load_index(Path(job["index_path"]))
        if index is None:
            raise RuntimeError("Dataset index not found or outdated. Load the dataset before training.")
    except Exception as exc:  # configuration problems must still surface in status.json
        logging.exception("Could not start training")
        RunIO(run_dir).write_status(state="failed", message=f"{type(exc).__name__}: {exc}",
                                    error=f"{type(exc).__name__}: {exc}", has_checkpoint=False)
        return 2
    status = run_training(cfg, run_dir, Path(job["checkpoint_dir"]), index, Path(job["raw_root"]),
                          Path(job["processed_root"]))
    return 0 if status.get("state") in ("completed", "stopped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
