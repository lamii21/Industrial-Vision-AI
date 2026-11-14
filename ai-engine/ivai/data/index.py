"""Build a tabular index of the dataset (one row per image).

The index is the single source of truth used by the split, the duplicate check
and the EDA notebook. Labels come from the file name (``<class>_<n>.jpg``); the
YOLO label files are only parsed to cross-check the class and to collect box
statistics for the future localisation extension.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ivai.config import DataConfig


def parse_label_from_name(filename: str, classes: tuple[str, ...]) -> str:
    """``rolled-in_scale_12.jpg`` -> ``rolled-in_scale``. Raises if unknown."""
    m = re.fullmatch(r"(.+)_(\d+)\.jpg", filename, flags=re.IGNORECASE)
    if not m or m.group(1) not in classes:
        raise ValueError(f"Cannot infer class from file name: {filename}")
    return m.group(1)


def load_gray(path: Path) -> np.ndarray:
    """Load an image as a 2-D uint8 grayscale array."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def pixel_sha256(arr: np.ndarray) -> str:
    """Hash of the decoded pixels (detects identical images even if file bytes differ)."""
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def read_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file: ``class cx cy w h`` per line (normalised coordinates)."""
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"Malformed YOLO line in {path.name}: {line!r}")
        cls, *coords = parts
        boxes.append((int(cls), *map(float, coords)))
    return boxes


def build_index(raw_dir: Path, cfg: DataConfig) -> pd.DataFrame:
    images_dir, labels_dir = raw_dir / "images", raw_dir / "labels"
    files = sorted(images_dir.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No images in {images_dir}. Run scripts/download_data.py first.")

    c2i = cfg.class_to_idx
    rows = []
    for path in files:
        label = parse_label_from_name(path.name, cfg.classes)
        with Image.open(path) as im:
            mode, (w, h) = im.mode, im.size
            rgb = np.asarray(im.convert("RGB"))
        # A 3-channel JPEG whose channels are identical is grayscale in practice.
        is_gray = mode == "L" or bool(
            np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])
        )
        gray = load_gray(path)
        boxes = read_yolo_labels(labels_dir / f"{path.stem}.txt")
        box_classes = sorted({b[0] for b in boxes})
        own = c2i[label]
        rows.append(
            {
                "filename": path.name,
                "relpath": path.relative_to(raw_dir).as_posix(),
                "label": label,
                "label_idx": c2i[label],
                "width": w,
                "height": h,
                "mode": mode,
                "is_grayscale": is_gray,
                "mean_intensity": float(gray.mean()),
                "std_intensity": float(gray.std()),
                "pixel_sha256": pixel_sha256(gray),
                "file_bytes": path.stat().st_size,
                "n_boxes": len(boxes),
                "box_classes": ",".join(map(str, box_classes)),
                # The image-level label must appear among the boxes (else: annotation error).
                "box_contains_label": (own in box_classes) if boxes else None,
                # Other defect types annotated in the same image (multi-defect image).
                "secondary_classes": ",".join(str(c) for c in box_classes if c != own),
                "has_secondary_defects": any(c != own for c in box_classes),
                "box_area_mean": float(np.mean([b[3] * b[4] for b in boxes])) if boxes else np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_boxes_table(raw_dir: Path, index: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    """Long table of all YOLO boxes (one row per box) for the localisation EDA."""
    rows = []
    for fname in index["filename"]:
        for cls, cx, cy, w, h in read_yolo_labels(raw_dir / "labels" / f"{Path(fname).stem}.txt"):
            rows.append({"filename": fname, "box_class": cls, "cx": cx, "cy": cy, "w": w, "h": h})
    df = pd.DataFrame(rows, columns=["filename", "box_class", "cx", "cy", "w", "h"])
    df.to_csv(out_csv, index=False)
    return df
