"""Download and safely extract the NEU-CLS archive.

Design choices
--------------
* The archive is verified with the MD5 published by the source before extraction,
  so a corrupted or altered download is rejected.
* Extraction is "safe": only ``.jpg`` images and ``.txt`` YOLO labels are kept,
  absolute paths / ``..`` components are refused (zip-slip protection), and the
  Ultralytics ``labels.cache`` files (Python pickles) are never extracted, because
  unpickling an untrusted file can execute arbitrary code.
* The mirror ships an arbitrary YOLO ``train/valid`` split (1771/31). It is
  discarded: files are flattened into ``images/`` and ``labels/`` and our own
  stratified split is built later in :mod:`ivai.data.split`.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ALLOWED_SUFFIXES = {".jpg": "images", ".txt": "labels"}


class DownloadError(RuntimeError):
    pass


def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, expected_md5: str | None = None, timeout: int = 60) -> Path:
    """Download ``url`` to ``dest`` (skipped if a file with the right MD5 already exists)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_md5 and md5sum(dest) == expected_md5:
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "ivai-dataset-downloader/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
    except OSError as exc:  # URLError, timeouts, disk errors
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"Download failed for {url}: {exc}") from exc

    if expected_md5:
        actual = md5sum(tmp)
        if actual != expected_md5:
            tmp.unlink(missing_ok=True)
            raise DownloadError(f"MD5 mismatch: expected {expected_md5}, got {actual}")
    tmp.replace(dest)
    return dest


def _safe_member_name(name: str) -> PurePosixPath | None:
    """Return the member path if it is safe and wanted, else ``None``."""
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or ":" in name:
        raise DownloadError(f"Unsafe path in archive: {name!r}")
    if p.suffix.lower() not in ALLOWED_SUFFIXES:
        return None
    return p


def extract_neu_archive(archive: Path, out_dir: Path) -> dict[str, int]:
    """Flatten images and YOLO labels from the archive into ``out_dir/{images,labels}``.

    Returns the number of extracted files per kind. Fails on duplicate file names,
    which would otherwise silently overwrite an image.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    counts = {"images": 0, "labels": 0}
    seen: set[str] = set()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            member = _safe_member_name(info.filename)
            if member is None:
                continue
            kind = ALLOWED_SUFFIXES[member.suffix.lower()]
            key = f"{kind}/{member.name}"
            if key in seen:
                raise DownloadError(f"Duplicate file name in archive: {member.name}")
            seen.add(key)
            target = out_dir / kind / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            counts[kind] += 1
    return counts
