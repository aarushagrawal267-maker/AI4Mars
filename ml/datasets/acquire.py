"""Dataset acquisition: subset download, full download, and local archive import.

Three supported routes, all ending in the same on-disk layout under ``data/raw``:

1. ``download_subset`` - fetch only the Curiosity (MSL) NavCam files needed for an
   experiment directly out of the Zenodo archive (range requests, CRC-checked).
2. ``download_full``   - download the whole 16 GB archive (resumable, MD5-checked)
   and extract the MSL NavCam portion.
3. ``extract_archive`` - extract a ZIP the user already has (e.g. from Zenodo or
   the Kaggle mirror) that was placed in ``data/raw/imports``.
"""
from __future__ import annotations

import hashlib
import logging
import os
import random
import re
import zipfile
from pathlib import Path
from typing import Callable

import requests

from .naming import parse_msl_name
from .remote_zip import DownloadCancelled, RemoteZip, ZipEntry, safe_join
from .sources import USER_AGENT, DatasetSource, resolve_source

log = logging.getLogger(__name__)

Progress = Callable[[str, float, str], None]  # (stage, fraction 0..1, message)
_MSL_RE = re.compile(r"(^|/)msl/(ncam/)?(images/edr|labels/train|labels/test)/", re.IGNORECASE)
_DOC_EXT = {".md", ".json", ".txt", ".csv"}


def _noop(stage: str, fraction: float, message: str) -> None:  # pragma: no cover - default callback
    pass


def plan_subset(entries: list[ZipEntry], train_count: int | None, seed: int = 42) -> tuple[list[ZipEntry], dict]:
    """Choose which archive members to fetch for an MSL NavCam subset.

    Training images are chosen as whole *rover positions* (site + drive groups), in a
    seeded random order, until ``train_count`` images are reached (``None`` = all).
    Picking whole positions keeps frames from one location together, which the
    train/validation split relies on, and keeps byte ranges contiguous.
    The complete expert test set (all agreement variants) is always included.
    """
    by_dir: dict[str, list[ZipEntry]] = {}
    for e in entries:
        by_dir.setdefault(os.path.dirname(e.name), []).append(e)

    def find(suffix: str) -> list[ZipEntry]:
        hits = [v for k, v in by_dir.items() if k.endswith(suffix)]
        return hits[0] if hits else []

    images = {Path(e.name).stem: e for e in find("msl/ncam/images/edr")}
    train = {Path(e.name).stem: e for e in find("msl/ncam/labels/train")}
    if not images or not train:
        raise RuntimeError("Archive does not contain the expected msl/ncam layout; refusing to guess.")

    groups: dict[tuple[str, str], list[str]] = {}
    for stem in sorted(train):
        if stem not in images:
            continue
        info = parse_msl_name(stem)
        groups.setdefault((info.site, info.drive), []).append(stem)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    chosen: list[str] = []
    for key in keys:
        if train_count is not None and len(chosen) >= train_count:
            break
        chosen.extend(groups[key])

    selected: list[ZipEntry] = []
    for stem in chosen:
        selected += [images[stem], train[stem]]
    test_stems: set[str] = set()
    n_test_labels = 0
    for d, items in by_dir.items():
        if "/msl/ncam/labels/test/" in d + "/":
            for e in items:
                selected.append(e)
                if e.name.lower().endswith(".png"):
                    n_test_labels += 1
                    test_stems.add(Path(e.name).stem.replace("_merged", ""))
    for stem in sorted(test_stems):
        if stem in images:
            selected.append(images[stem])
    # Dataset documentation files (tiny): info.md, label_keys.json, notes.
    for e in entries:
        if Path(e.name).suffix.lower() in _DOC_EXT and ("/msl/" in e.name or e.name.count("/") == 1):
            selected.append(e)
    unique = {e.name: e for e in selected}
    summary = {
        "train_pairs": len(chosen),
        "train_positions": len({(parse_msl_name(s).site, parse_msl_name(s).drive) for s in chosen}),
        "train_pairs_available": sum(len(v) for v in groups.values()),
        "test_images": len(test_stems),
        "test_label_files": n_test_labels,
        "files": len(unique),
        "bytes": sum(e.compressed_size for e in unique.values()),
        "seed": seed,
    }
    return list(unique.values()), summary


def download_subset(dest: Path, train_count: int | None, seed: int = 42, progress: Progress = _noop,
                    should_stop: Callable[[], bool] | None = None, source: DatasetSource | None = None) -> dict:
    """Fetch an MSL NavCam subset straight from Zenodo into ``dest``."""
    source = source or resolve_source()
    progress("directory", 0.0, "Reading archive directory from Zenodo")
    rz = RemoteZip(source.url)
    if rz.size != source.size:
        raise RuntimeError(f"Remote archive size {rz.size} differs from the Zenodo record ({source.size}).")
    entries, summary = plan_subset(rz.entries, train_count, seed)
    mb = summary["bytes"] / 1e6

    def cb(fd: int, ft: int, bd: int, bt: int) -> None:
        progress("download", bd / max(bt, 1), f"{fd}/{ft} files, {bd / 1e6:.0f}/{bt / 1e6:.0f} MB")

    progress("download", 0.0, f"Fetching {summary['files']} files ({mb:.0f} MB)")
    rz.fetch(entries, dest, progress=cb, should_stop=should_stop)
    summary.update({"source_url": source.url, "source_version": source.version, "mode": "subset"})
    return summary


def download_full(dest: Path, progress: Progress = _noop, should_stop: Callable[[], bool] | None = None,
                  source: DatasetSource | None = None, keep_zip: bool = False) -> dict:
    """Download the full archive (resumable), verify its MD5, then extract MSL NavCam."""
    source = source or resolve_source()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / source.name
    part = zip_path.with_name(zip_path.name + ".part")
    if not zip_path.exists():
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        with requests.get(source.url, headers=headers, stream=True, timeout=120) as r:
            if have and r.status_code != 206:
                have = 0  # server ignored the range: restart
            r.raise_for_status()
            with open(part, "ab" if have else "wb") as fh:
                for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                    if should_stop and should_stop():
                        raise DownloadCancelled()
                    fh.write(chunk)
                    have += len(chunk)
                    progress("download", have / source.size, f"{have / 1e9:.2f}/{source.size / 1e9:.2f} GB")
        progress("verify", 0.0, "Verifying MD5 checksum")
        md5 = hashlib.md5()
        with open(part, "rb") as fh:
            done = 0
            while chunk := fh.read(8 * 1024 * 1024):
                md5.update(chunk)
                done += len(chunk)
                progress("verify", done / source.size, "Verifying MD5 checksum")
        if md5.hexdigest() != source.md5:
            raise RuntimeError("MD5 mismatch: the downloaded archive is corrupt. Delete the .part file and retry.")
        part.replace(zip_path)
    summary = extract_archive(zip_path, dest, progress=progress, should_stop=should_stop)
    if not keep_zip:
        zip_path.unlink(missing_ok=True)
    summary.update({"source_url": source.url, "source_version": source.version, "mode": "full"})
    return summary


def extract_archive(zip_path: Path, dest: Path, progress: Progress = _noop,
                    should_stop: Callable[[], bool] | None = None, msl_only: bool = True) -> dict:
    """Safely extract a local AI4Mars ZIP. By default only the MSL image/label folders are extracted."""
    dest = Path(dest)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if msl_only:
            wanted = [m for m in members if _MSL_RE.search(m.filename)
                      or Path(m.filename).suffix.lower() in _DOC_EXT]
        else:
            wanted = members
        if not wanted:
            raise RuntimeError("No AI4Mars MSL image/label folders were found inside this archive.")
        total = len(wanted)
        for i, m in enumerate(wanted):
            if should_stop and should_stop():
                raise DownloadCancelled()
            target = safe_join(dest, m.filename)
            if target.exists() and target.stat().st_size == m.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as out:
                while chunk := src.read(1024 * 1024):
                    out.write(chunk)
            if i % 200 == 0:
                progress("extract", i / total, f"Extracting {i}/{total}")
    progress("extract", 1.0, f"Extracted {total} files")
    return {"files": total, "mode": "import", "archive": Path(zip_path).name}
