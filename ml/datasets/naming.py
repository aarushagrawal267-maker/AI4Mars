"""Parse Curiosity (MSL) NavCam product names.

Example: ``NLB_451023553EDR_F0310634NCAM00353M1``

Fields used here follow the MSL camera product naming convention referenced by the
dataset's ``info.md`` (MSL Camera SIS): instrument (``NLB`` = NavCam Left, B-side
computer; ``NR*`` = right eye), 9-digit spacecraft clock (SCLK), product type
(``EDR``), a one-letter sample/config code, a 3-character site index, a
4-character drive index and a sequence id.

Site + drive identify one rover position. Frames taken from the same position
(stereo pairs, panorama tiles) look very similar, so the train/validation split
groups by this key to avoid near-duplicate leakage.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_MSL = re.compile(r"^(N[LR][AB])_(\d{9})([A-Z]{3})_([A-Z])([0-9A-Z]{3})([0-9A-Z]{4})(.*)$")


@dataclass(frozen=True)
class MslName:
    camera: str
    sclk: int
    site: str
    drive: str
    parsed: bool

    @property
    def group(self) -> str:
        return f"{self.site}-{self.drive}"


def parse_msl_name(stem: str) -> MslName:
    """Return camera/SCLK/site/drive. Unparseable names get their own singleton group."""
    m = _MSL.match(stem)
    if not m:
        return MslName(camera="unknown", sclk=0, site="unparsed", drive=stem, parsed=False)
    return MslName(camera=m.group(1), sclk=int(m.group(2)), site=m.group(5), drive=m.group(6), parsed=True)
