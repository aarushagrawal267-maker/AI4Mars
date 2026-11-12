"""Range-request ZIP extraction, tested against a local HTTP server (no internet needed)."""
from __future__ import annotations

import http.server
import threading
import zipfile
from pathlib import Path

import pytest

from ml.datasets.acquire import extract_archive, plan_subset
from ml.datasets.remote_zip import RemoteZip, safe_join


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = b""

    def do_GET(self):  # noqa: N802
        rng = self.headers.get("Range")
        if not rng:
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)
            return
        start, end = rng.replace("bytes=", "").split("-")
        start, end = int(start), min(int(end), len(self.payload) - 1)
        chunk = self.payload[start:end + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(self.payload)}")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)

    def log_message(self, *args):  # silence test output
        pass


def _make_zip(path: Path) -> dict[str, bytes]:
    root = "ai4mars-dataset-merged-0.6"
    files = {f"{root}/info.md": b"# info\n"}
    for i in range(6):
        stem = f"NLB_4000000{i:02d}EDR_F{i // 2 + 1:03d}0000NCAM00353M1"
        files[f"{root}/msl/ncam/images/edr/{stem}.JPG"] = bytes([i]) * (2000 + i)
        files[f"{root}/msl/ncam/labels/train/{stem}.png"] = bytes([i + 50]) * 300
    t = "NLB_500000000EDR_F0990000NCAM00353M1"
    files[f"{root}/msl/ncam/images/edr/{t}.JPG"] = b"T" * 1500
    files[f"{root}/msl/ncam/labels/test/masked-gold-min3-100agree/{t}_merged.png"] = b"L" * 200
    files[f"{root}/mer/images/eff/other.JPG"] = b"M" * 5000
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data, compress_type=zipfile.ZIP_STORED if name.endswith(".md") else zipfile.ZIP_DEFLATED)
    return files


@pytest.fixture()
def served_zip(tmp_path):
    zpath = tmp_path / "archive.zip"
    files = _make_zip(zpath)
    handler = type("H", (_RangeHandler,), {"payload": zpath.read_bytes()})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/archive.zip", files, zpath
    finally:
        server.shutdown()


def test_remote_zip_lists_and_fetches_subset(served_zip, tmp_path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    url, files, _ = served_zip
    rz = RemoteZip(url)
    rz.throttle.min_interval = 0
    assert {e.name for e in rz.entries} == set(files)
    entries, summary = plan_subset(rz.entries, train_count=2, seed=0)
    assert summary["test_images"] == 1 and summary["train_pairs"] >= 2
    names = {e.name for e in entries}
    assert not any("/mer/" in n for n in names)                 # other missions are not fetched
    assert any(n.endswith("info.md") for n in names)
    calls = []
    dest = tmp_path / "out"
    written = rz.fetch(entries, dest, progress=lambda *a: calls.append(a))
    assert written == len(entries) and calls[-1][0] == calls[-1][1] == len(entries)
    for e in entries:                                           # byte-exact, CRC-checked extraction
        assert (dest / e.name).read_bytes() == files[e.name]
    assert rz.fetch(entries, dest) == 0                         # resumable: nothing re-downloaded


def test_safe_join_blocks_zip_slip(tmp_path):
    assert safe_join(tmp_path, "a/b.txt") == (tmp_path / "a" / "b.txt").resolve()
    for evil in ("../escape.txt", "a/../../escape.txt", "/etc/passwd"):
        with pytest.raises(ValueError):
            safe_join(tmp_path, evil)


def test_extract_local_archive_only_msl(served_zip, tmp_path):
    _url, files, zpath = served_zip
    summary = extract_archive(zpath, tmp_path / "raw")
    assert summary["mode"] == "import"
    assert not (tmp_path / "raw" / "ai4mars-dataset-merged-0.6" / "mer").exists()
    assert len(list((tmp_path / "raw").rglob("*.JPG"))) == 7
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../msl/ncam/images/edr/x.JPG", b"x")
    with pytest.raises(ValueError):
        extract_archive(evil, tmp_path / "raw2")
