"""Terrain class definitions for the AI4Mars navigation (NAV) label set.

The integer encoding below is NOT an assumption: it is the encoding documented in
the dataset's own ``info.md`` / ``label_keys.json`` (AI4Mars merged dataset v0.6,
Zenodo record 15995036) and was verified against real label files:

    0 = soil, 1 = bedrock, 2 = sand, 3 = big rock, 255 = NULL (no label)

Class IDs are fixed. Display colours are presentation only and can be changed
without touching the IDs.
"""
from __future__ import annotations

NUM_CLASSES = 4
IGNORE_INDEX = 255

CLASS_KEYS: list[str] = ["soil", "bedrock", "sand", "big_rock"]
CLASS_LABELS: list[str] = ["Soil", "Bedrock", "Sand", "Big rock"]

# Default display palette (hex). Soil: earthy brown, bedrock: blue,
# sand: orange/yellow, big rock: purple.
DEFAULT_COLORS: dict[str, str] = {
    "soil": "#9C6B3F",
    "bedrock": "#3B82F6",
    "sand": "#F5B041",
    "big_rock": "#A855F7",
}
# Colour used for unlabelled (NULL) pixels in ground-truth visualisations.
UNLABELED_COLOR = "#1F2328"

VALID_LABEL_VALUES = frozenset({0, 1, 2, 3, IGNORE_INDEX})


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Convert ``#RRGGBB`` to an (R, G, B) tuple. Raises ValueError if malformed."""
    v = value.strip().lstrip("#")
    if len(v) != 6 or any(c not in "0123456789abcdefABCDEF" for c in v):
        raise ValueError(f"Invalid hex colour: {value!r}")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def palette_array(colors: dict[str, str] | None = None):
    """Return a (256, 3) uint8 lookup table: class id -> RGB, 255 -> unlabelled colour."""
    import numpy as np

    merged = {**DEFAULT_COLORS, **(colors or {})}
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:] = hex_to_rgb(UNLABELED_COLOR)
    for idx, key in enumerate(CLASS_KEYS):
        lut[idx] = hex_to_rgb(merged[key])
    return lut
