"""Install the pre-trained models and real results from ``deploy/bundle`` into your local folders.

A fresh clone has no trained model, so the Image Analysis, Evaluation and Terrain pages have nothing to run until
you download the dataset and train (about 40 minutes on a laptop CPU). This script copies what the repository
already ships in ``deploy/bundle``:

* the trained U-Net and DeepLabV3 checkpoints (``checkpoints/*/best.pt``),
* their training logs and curves (``experiments/``),
* the results database (experiments, evaluations, example analyses),
* the official AI4Mars expert-labelled test split (322 images + labels) and its dataset index.

so the app is usable immediately. It never overwrites your own work: if a trained checkpoint or a database
with experiments already exists it does nothing (use ``--force`` to add the bundled models anyway: existing files
are kept, and the bundled experiments are added to your database next to your own).

Usage (from the repository root, inside the virtual environment)::

    python scripts/seed_demo_models.py            # copy only if you have no trained model yet
    python scripts/seed_demo_models.py --force    # add the bundled models even if you already have your own
    python scripts/seed_demo_models.py --check    # report what would happen, change nothing
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "deploy" / "bundle"
DB_NAME = "mti.sqlite3"


def _has_trained_checkpoint(checkpoint_dir: Path) -> bool:
    return checkpoint_dir.is_dir() and any(checkpoint_dir.glob("*/best.pt"))


def _db_has_work(db_path: Path) -> bool:
    """True if the database already holds experiments or evaluations (i.e. it is not the empty one from setup)."""
    if not db_path.is_file():
        return False
    try:
        with sqlite3.connect(db_path) as con:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table in ("experiments", "evaluations"):
                if table in tables and con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] > 0:
                    return True
    except sqlite3.DatabaseError:
        return True  # unreadable database: do not touch it
    return False


def _merge_bundled_rows(db_path: Path, bundle_db: Path) -> str:
    """Add the bundle's experiments (and their evaluations and example analyses) to an existing database.

    Only experiments whose id is not in the database yet are added, so the user's rows are never changed and running
    it again adds nothing twice. Evaluation ids are left to the database (they are autoincrement integers that would
    collide with the user's). Returns a short description for the summary.
    """
    con = None
    try:
        con = sqlite3.connect(db_path)
        with con:  # one transaction: all rows are added, or none
            con.execute("ATTACH DATABASE ? AS bundle", (str(bundle_db),))

            def columns(table: str) -> list[str]:
                mine = [r[1] for r in con.execute(f"PRAGMA main.table_info({table})")]
                theirs = {r[1] for r in con.execute(f"PRAGMA bundle.table_info({table})")}
                return [c for c in mine if c in theirs]

            new_ids = [r[0] for r in con.execute(
                "SELECT id FROM bundle.experiments WHERE id NOT IN (SELECT id FROM main.experiments)")]
            if not new_ids:
                return "kept (it already has the bundled models)"
            marks = ",".join("?" * len(new_ids))
            for table, key, skip in (("experiments", "id", ()), ("evaluations", "experiment_id", ("id",)),
                                     ("analyses", "experiment_id", ())):
                cols = ",".join(f'"{c}"' for c in columns(table) if c not in skip)
                if not cols:  # table missing in either database
                    continue
                con.execute(f"INSERT OR IGNORE INTO main.{table} ({cols}) SELECT {cols} FROM bundle.{table} "
                            f"WHERE {key} IN ({marks})", new_ids)
        return f"kept, {len(new_ids)} bundled model(s) added to it"
    except sqlite3.DatabaseError as exc:
        return f"kept (the bundled models could not be added to it: {exc})"
    finally:
        if con is not None:
            con.close()


def _copy_tree(src: Path, dst: Path) -> int:
    """Copy files from src to dst without overwriting anything that already exists. Returns files copied."""
    copied = 0
    for path in sorted(src.rglob("*")):
        if not path.is_file() or path.name.endswith(("-shm", "-wal")):
            continue
        target = dst / path.relative_to(src)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def seed(data_dir: Path, checkpoint_dir: Path, experiments_dir: Path, database_url: str = "",
         bundle: Path = BUNDLE, force: bool = False, dry_run: bool = False) -> dict:
    """Copy the demo bundle into the given folders. Returns a summary dict (``status`` and file counts)."""
    if not (bundle / "checkpoints").is_dir():
        return {"status": "missing-bundle", "message": f"No demo bundle found at {bundle}."}

    db_path = data_dir / DB_NAME
    custom_db = bool(database_url) and database_url != f"sqlite:///{db_path.as_posix()}"
    has_models = _has_trained_checkpoint(checkpoint_dir)
    has_db_work = (not custom_db) and _db_has_work(db_path)

    if (has_models or has_db_work) and not force:
        return {"status": "skipped",
                "message": "You already have trained models or results; nothing was changed (use --force to add "
                           "the bundled ones alongside them)."}
    if dry_run:
        return {"status": "would-copy", "message": f"Would copy the demo bundle into {data_dir}, {checkpoint_dir} "
                                                    f"and {experiments_dir}."}

    summary = {"status": "copied", "checkpoints": 0, "experiments": 0, "data": 0, "database": "kept"}
    summary["checkpoints"] = _copy_tree(bundle / "checkpoints", checkpoint_dir)
    summary["experiments"] = _copy_tree(bundle / "experiments", experiments_dir)

    # The database: replace the EMPTY one setup creates; never replace one that holds results. With --force the
    # bundled models are added to it instead, because the app only lists models that have a database row.
    if custom_db:
        summary["database"] = "skipped (MTI_DATABASE_URL points to another database)"
    elif not _db_has_work(db_path):
        for suffix in ("", "-shm", "-wal"):
            Path(str(db_path) + suffix).unlink(missing_ok=True)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle / "data" / DB_NAME, db_path)
        summary["database"] = "installed"
    else:
        summary["database"] = _merge_bundled_rows(db_path, bundle / "data" / DB_NAME)

    # Test images, labels, dataset index and the example analyses (never overwrites existing files).
    for path in sorted((bundle / "data").iterdir()):
        if path.name.startswith(DB_NAME):
            continue
        if path.is_dir():
            summary["data"] += _copy_tree(path, data_dir / path.name)
        elif not (data_dir / path.name).exists():
            shutil.copy2(path, data_dir / path.name)
            summary["data"] += 1
    summary["message"] = (f"Installed {summary['checkpoints']} checkpoint file(s), {summary['experiments']} experiment "
                          f"file(s) and {summary['data']} data file(s); database {summary['database']}.")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="copy even if trained models already exist")
    parser.add_argument("--check", action="store_true", help="only report what would happen")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT))
    from backend.app.core.config import get_settings  # noqa: E402  (needs the repo on sys.path)

    s = get_settings()
    result = seed(s.data_dir, s.checkpoint_dir, s.experiments_dir, s.database_url or "",
                  force=args.force, dry_run=args.check)
    print(result["message"])
    return 1 if result["status"] == "missing-bundle" else 0


if __name__ == "__main__":
    raise SystemExit(main())
