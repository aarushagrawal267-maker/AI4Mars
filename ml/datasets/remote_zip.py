"""Read selected members out of a remote ZIP archive using HTTP range requests.

The AI4Mars archive is 16 GB. A student laptop often cannot (or should not)
download all of it just to run a small experiment. Zenodo supports HTTP range
requests, so we read the ZIP central directory (a few MB) and then fetch only
the byte ranges that contain the files we want. Every extracted file is checked
against the CRC-32 stored in the archive.
"""
from __future__ import annotations

import io
import logging
import struct
import time
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import requests

from .sources import USER_AGENT

log = logging.getLogger(__name__)

_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
_LOCAL_SIG = 0x04034B50


class DownloadCancelled(Exception):
    """Raised when the caller asks a running download to stop."""


@dataclass(frozen=True)
class ZipEntry:
    name: str
    size: int
    compressed_size: int
    header_offset: int
    method: int
    crc: int


class _Throttle:
    """Keeps request rate under the host's published limit (Zenodo: ~133/min)."""

    def __init__(self, min_interval: float = 0.55):
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        delta = time.monotonic() - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.monotonic()


def _range_get(session: requests.Session, url: str, start: int, end: int, throttle: _Throttle,
               retries: int = 5, timeout: float = 300.0) -> bytes:
    """GET bytes [start, end] inclusive, with retry/back-off and 429 handling."""
    last_err: Exception | None = None
    for attempt in range(retries):
        throttle.wait()
        try:
            r = session.get(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT}, timeout=timeout)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", "30"))
                log.warning("Rate limited by host; sleeping %.0fs", wait)
                time.sleep(min(wait, 120))
                continue
            if r.status_code != 206:
                raise IOError(f"Server did not honour range request (HTTP {r.status_code})")
            data = r.content
            if len(data) != end - start + 1:
                raise IOError(f"Short read: wanted {end - start + 1} bytes, got {len(data)}")
            return data
        except (requests.RequestException, IOError) as exc:  # network hiccup: back off and retry
            last_err = exc
            time.sleep(min(2 ** attempt, 30))
    raise IOError(f"Range request failed after {retries} attempts: {last_err}")


class _HttpRangeFile(io.RawIOBase):
    """Minimal seekable file over HTTP ranges, enough for ``zipfile`` to parse the directory."""

    def __init__(self, url: str, session: requests.Session, throttle: _Throttle, block: int = 4 * 1024 * 1024):
        self.url, self.session, self.throttle, self.block = url, session, throttle, block
        r = session.get(url, headers={"Range": "bytes=0-0", "User-Agent": USER_AGENT}, timeout=60)
        if r.status_code != 206 or "Content-Range" not in r.headers:
            raise IOError("Remote host does not support HTTP range requests for this file")
        self.size = int(r.headers["Content-Range"].split("/")[1])
        self.pos = 0
        self._cache: list[tuple[int, bytes]] = []

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        for start, buf in self._cache:
            if start <= self.pos and self.pos + n <= start + len(buf):
                out = buf[self.pos - start:self.pos - start + n]
                self.pos += n
                return out
        end = min(self.size, self.pos + max(n, self.block))
        buf = _range_get(self.session, self.url, self.pos, end - 1, self.throttle)
        self._cache = (self._cache + [(self.pos, buf)])[-4:]
        self.pos += n
        return buf[:n]


def safe_join(root: Path, relative: str) -> Path:
    """Join an archive member name onto ``root``, refusing anything that escapes it (zip-slip)."""
    root = root.resolve()
    target = (root / relative).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"Archive member escapes destination: {relative!r}")
    return target


class RemoteZip:
    """A ZIP archive reachable over HTTP with range support."""

    def __init__(self, url: str):
        self.url = url
        self.session = requests.Session()
        self.throttle = _Throttle()
        fh = _HttpRangeFile(url, self.session, self.throttle)
        self.size = fh.size
        with zipfile.ZipFile(fh) as zf:
            self.entries: list[ZipEntry] = [
                ZipEntry(i.filename, i.file_size, i.compress_size, i.header_offset, i.compress_type, i.CRC)
                for i in zf.infolist() if not i.is_dir()
            ]
        offsets = sorted(e.header_offset for e in self.entries)
        # Upper bound for each member: start of the next local header (or archive size).
        self._next_offset = {a: b for a, b in zip(offsets, offsets[1:] + [self.size])}

    def _groups(self, entries: Sequence[ZipEntry], max_span: int, max_gap: int) -> list[list[ZipEntry]]:
        """Group members into byte ranges so nearby files share one request."""
        groups: list[list[ZipEntry]] = []
        for e in sorted(entries, key=lambda x: x.header_offset):
            if groups:
                first, last = groups[-1][0], groups[-1][-1]
                gap = e.header_offset - self._next_offset[last.header_offset]
                span = self._next_offset[e.header_offset] - first.header_offset
                if gap <= max_gap and span <= max_span:
                    groups[-1].append(e)
                    continue
            groups.append([e])
        return groups

    def fetch(self, entries: Iterable[ZipEntry], dest_root: Path, strip_prefix: str = "",
              progress: Callable[[int, int, int, int], None] | None = None,
              should_stop: Callable[[], bool] | None = None, skip_existing: bool = True,
              max_span: int = 48 * 1024 * 1024, max_gap: int = 1024 * 1024) -> int:
        """Download and extract ``entries`` beneath ``dest_root``. Returns files written.

        ``progress(files_done, files_total, bytes_done, bytes_total)`` is called after each range.
        Existing files with the expected size are skipped, so an interrupted download resumes.
        """
        dest_root = Path(dest_root)
        todo: list[ZipEntry] = []
        all_entries = list(entries)
        for e in all_entries:
            rel = e.name[len(strip_prefix):] if strip_prefix and e.name.startswith(strip_prefix) else e.name
            target = safe_join(dest_root, rel)
            if skip_existing and target.exists() and target.stat().st_size == e.size:
                continue
            todo.append(e)
        files_total, bytes_total = len(all_entries), sum(e.compressed_size for e in all_entries)
        files_done = files_total - len(todo)
        bytes_done = bytes_total - sum(e.compressed_size for e in todo)
        if progress:
            progress(files_done, files_total, bytes_done, bytes_total)
        written = 0
        for group in self._groups(todo, max_span, max_gap):
            if should_stop and should_stop():
                raise DownloadCancelled()
            start = group[0].header_offset
            end = self._next_offset[group[-1].header_offset] - 1
            buf = _range_get(self.session, self.url, start, end, self.throttle)
            for e in group:
                p = e.header_offset - start
                sig, _v, _f, _m, _t, _d, _crc, _cs, _us, name_len, extra_len = _LOCAL_HEADER.unpack_from(buf, p)
                if sig != _LOCAL_SIG:
                    raise IOError(f"Corrupt local header for {e.name}")
                data_start = p + _LOCAL_HEADER.size + name_len + extra_len
                raw = buf[data_start:data_start + e.compressed_size]
                if e.method == zipfile.ZIP_DEFLATED:
                    data = zlib.decompress(raw, -15)
                elif e.method == zipfile.ZIP_STORED:
                    data = raw
                else:
                    raise IOError(f"Unsupported compression method {e.method} for {e.name}")
                if zlib.crc32(data) & 0xFFFFFFFF != e.crc:
                    raise IOError(f"CRC mismatch for {e.name}")
                rel = e.name[len(strip_prefix):] if strip_prefix and e.name.startswith(strip_prefix) else e.name
                target = safe_join(dest_root, rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + ".part")
                tmp.write_bytes(data)
                tmp.replace(target)
                written += 1
                files_done += 1
                bytes_done += e.compressed_size
            if progress:
                progress(files_done, files_total, bytes_done, bytes_total)
        return written
