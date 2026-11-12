"""Locate an AI4Mars MSL NavCam dataset on disk without assuming one exact layout.

Known layouts:

* v0.6 (Zenodo 15995036): ``<root>/msl/ncam/images/edr``, ``.../labels/train``,
  ``.../labels/test/masked-gold-min{1,2,3}-100agree``
* older releases / mirrors: ``<root>/msl/images/edr``, ``<root>/msl/labels/...``

Rather than hard-coding either, we search for a directory that contains
``images/edr`` together with ``labels/train`` and describe what is actually there.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png"}


@dataclass
class DatasetLayout:
    base: Path                    # directory holding images/ and labels/
    images_dir: Path
    train_labels_dir: Path
    test_label_dirs: dict[str, Path] = field(default_factory=dict)   # variant -> dir
    rover_mask_dir: Path | None = None
    range_mask_dir: Path | None = None
    info_file: Path | None = None
    label_keys: dict | None = None

    def describe(self, root: Path) -> dict:
        def rel(p: Path | None) -> str | None:
            if p is None:
                return None
            try:
                return str(p.relative_to(root)).replace(os.sep, "/")
            except ValueError:
                return p.name

        return {
            "base": rel(self.base),
            "images_dir": rel(self.images_dir),
            "train_labels_dir": rel(self.train_labels_dir),
            "test_label_dirs": {k: rel(v) for k, v in self.test_label_dirs.items()},
            "rover_mask_dir": rel(self.rover_mask_dir),
            "range_mask_dir": rel(self.range_mask_dir),
            "info_file": rel(self.info_file),
            "label_keys_nav": (self.label_keys or {}).get("NAV"),
        }


def _variant_name(dirname: str) -> str:
    """``masked-gold-min3-100agree`` -> ``min3``; unknown names are kept verbatim."""
    for part in dirname.split("-"):
        if part.startswith("min") and part[3:].isdigit():
            return part
    return dirname


def find_layout(raw_root: Path, max_depth: int = 5) -> DatasetLayout | None:
    """Search ``raw_root`` for an MSL NavCam dataset. Returns None when nothing usable exists."""
    raw_root = Path(raw_root)
    if not raw_root.is_dir():
        return None
    candidates: list[Path] = []
    base_depth = len(raw_root.parts)
    for dirpath, dirnames, _files in os.walk(raw_root):
        p = Path(dirpath)
        if len(p.parts) - base_depth >= max_depth:
            dirnames[:] = []
            continue
        if (p / "images" / "edr").is_dir() and (p / "labels" / "train").is_dir():
            candidates.append(p)
            dirnames[:] = []  # do not descend into a dataset we already found
    if not candidates:
        return None
    # Prefer a NavCam folder when several exist (e.g. msl/ncam next to other missions).
    candidates.sort(key=lambda c: (0 if "ncam" in c.name.lower() or c.name.lower() == "msl" else 1, len(c.parts)))
    base = candidates[0]
    layout = DatasetLayout(base=base, images_dir=base / "images" / "edr", train_labels_dir=base / "labels" / "train")
    test_root = base / "labels" / "test"
    if test_root.is_dir():
        subdirs = sorted(d for d in test_root.iterdir() if d.is_dir())
        if subdirs:
            layout.test_label_dirs = {_variant_name(d.name): d for d in subdirs}
        elif any(f.suffix.lower() == ".png" for f in test_root.iterdir()):
            layout.test_label_dirs = {"default": test_root}
    for name, attr in (("mxy", "rover_mask_dir"), ("rng-30m", "range_mask_dir")):
        d = base / "images" / name
        if d.is_dir():
            setattr(layout, attr, d)
    # Dataset documentation shipped with the archive, when present.
    for parent in [base, *base.parents]:
        if parent == raw_root.parent:
            break
        info, keys = parent / "info.md", parent / "label_keys.json"
        if layout.info_file is None and info.is_file():
            layout.info_file = info
        if layout.label_keys is None and keys.is_file():
            try:
                layout.label_keys = json.loads(keys.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                layout.label_keys = None
    return layout


def directory_tree(raw_root: Path, max_depth: int = 6) -> list[dict]:
    """Summarise the directory structure (folders with file counts by extension)."""
    raw_root = Path(raw_root)
    out: list[dict] = []
    if not raw_root.is_dir():
        return out
    base_depth = len(raw_root.parts)
    for dirpath, dirnames, files in os.walk(raw_root):
        dirnames.sort()
        p = Path(dirpath)
        depth = len(p.parts) - base_depth
        if depth > max_depth:
            dirnames[:] = []
            continue
        ext: dict[str, int] = {}
        size = 0
        for f in files:
            if f.endswith(".part"):
                continue
            e = Path(f).suffix.lower() or "(none)"
            ext[e] = ext.get(e, 0) + 1
            try:
                size += (p / f).stat().st_size
            except OSError:
                pass
        out.append({
            "path": "." if p == raw_root else str(p.relative_to(raw_root)).replace(os.sep, "/"),
            "depth": depth,
            "files": sum(ext.values()),
            "extensions": ext,
            "bytes": size,
        })
    return out
