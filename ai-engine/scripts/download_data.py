"""Download NEU-CLS, verify its MD5 and extract it into datasets/raw/neu-cls/.

Usage:  python scripts/download_data.py [--archive PATH]

--archive lets you use a zip you downloaded manually (e.g. from the official
Google Drive link) instead of the Figshare mirror; the MD5 check is then skipped
only if --no-md5 is given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ivai.config import NEU_RAW_DIR, RAW_DIR, load_data_config
from ivai.data.download import DownloadError, download_file, extract_neu_archive, md5sum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, help="use an already downloaded zip")
    parser.add_argument("--no-md5", action="store_true", help="skip the MD5 check for --archive")
    args = parser.parse_args()
    cfg = load_data_config()

    try:
        if args.archive:
            archive = args.archive
            if not args.no_md5 and md5sum(archive) != cfg.md5:
                raise DownloadError("MD5 of the provided archive does not match configs/data.yaml")
        else:
            archive = RAW_DIR / "NEU-CLS.zip"
            print(f"Downloading {cfg.source_url} ...")
            download_file(cfg.source_url, archive, expected_md5=cfg.md5)
        print(f"Archive OK : {archive} (md5 {md5sum(archive)})")
        counts = extract_neu_archive(archive, NEU_RAW_DIR)
    except DownloadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Extracted  : {counts['images']} images, {counts['labels']} YOLO label files -> {NEU_RAW_DIR}")
    expected = cfg.expected_per_class * len(cfg.classes)
    if counts["images"] != expected:
        print(f"ERROR: expected {expected} images", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
