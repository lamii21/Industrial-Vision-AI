"""Explicitly download the ImageNet weights used in Phase 3 (docs/phase3_protocol.md §5, §15).

This is the ONLY place where a download is allowed. Training never downloads.
torchvision verifies the SHA-256 prefix embedded in each file name.

Usage:  python scripts/download_weights.py
"""

from __future__ import annotations

import hashlib
import sys

from ivai.models.pretrained import WEIGHTS, load_imagenet_state_dict, weights_cache_path


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    for name, w in WEIGHTS.items():
        path = weights_cache_path(w)
        status = "already cached" if path.exists() else "downloading"
        print(f"{name}: {type(w).__name__}.{w.name} ({w.url}) -> {status}")
        state = load_imagenet_state_dict(w, allow_download=True)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"  {path} | {path.stat().st_size / 2**20:.1f} MB | sha256 {sha} | {len(state)} tensors")
        expected_prefix = path.stem.split("-")[-1]
        if not sha.startswith(expected_prefix):
            raise SystemExit(f"hash mismatch for {path.name}: expected prefix {expected_prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
