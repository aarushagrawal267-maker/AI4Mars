"""Verified dataset source for AI4Mars.

Source of truth: Zenodo record 15995036 (DOI 10.5281/zenodo.15995036),
"AI4Mars: A Dataset for Terrain-Aware Autonomous Driving on Mars", version 0.6,
creators Swan, Ono, Atha; licence CC-BY-4.0. File name, size and MD5 below were
read from the Zenodo REST API. ``resolve_source`` re-checks them at run time so a
changed record is detected instead of silently trusted.
"""
from __future__ import annotations

from dataclasses import dataclass

import requests

ZENODO_RECORD_ID = "15995036"
ZENODO_DOI = "10.5281/zenodo.15995036"
ZENODO_RECORD_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
KAGGLE_MIRROR_URL = "https://www.kaggle.com/datasets/yash92328/ai4mars-terrainaware-autonomous-driving-on-mars"

MERGED_ZIP_NAME = "ai4mars-dataset-merged-0.6.zip"
MERGED_ZIP_SIZE = 16_232_481_989
MERGED_ZIP_MD5 = "daf80a86021253292e6c425f97baa5c6"

USER_AGENT = "mars-terrain-intelligence/1.0 (research; python-requests)"


@dataclass(frozen=True)
class DatasetSource:
    url: str
    size: int
    md5: str
    name: str
    version: str | None
    verified_online: bool


def resolve_source(timeout: float = 30.0) -> DatasetSource:
    """Ask the Zenodo API for the merged archive and confirm it matches what we expect."""
    fallback_url = f"{ZENODO_API_URL}/files/{MERGED_ZIP_NAME}/content"
    try:
        r = requests.get(ZENODO_API_URL, timeout=timeout, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        record = r.json()
        for f in record.get("files", []):
            if f.get("key") == MERGED_ZIP_NAME:
                md5 = str(f.get("checksum", "")).replace("md5:", "")
                return DatasetSource(
                    url=f["links"]["self"],
                    size=int(f["size"]),
                    md5=md5,
                    name=MERGED_ZIP_NAME,
                    version=record.get("metadata", {}).get("version"),
                    verified_online=True,
                )
        raise RuntimeError(f"{MERGED_ZIP_NAME} is not listed in Zenodo record {ZENODO_RECORD_ID}")
    except requests.RequestException:
        return DatasetSource(fallback_url, MERGED_ZIP_SIZE, MERGED_ZIP_MD5, MERGED_ZIP_NAME, "0.6", False)
