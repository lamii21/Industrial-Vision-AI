"""Index the raw dataset, detect duplicates, build the stratified split and materialise it.

Usage:  python scripts/prepare_splits.py

Outputs
  datasets/splits/neu_cls_index.csv         full index (1 row / image, + duplicate group)
  datasets/splits/neu_cls_duplicates.csv    confirmed duplicate pairs (may be empty)
  datasets/splits/neu_cls_boxes.csv         YOLO boxes (for the future localisation work)
  datasets/splits/neu_cls_{train,val,test}.csv   the split (versioned with Git)
  datasets/splits/split_meta.json           seed, ratios, counts, checksums
  datasets/processed/neu-cls/{train,val,test}/<class>/*.jpg   ImageFolder layout
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone

import pandas as pd

from ivai.config import NEU_PROCESSED_DIR, NEU_RAW_DIR, SPLITS_DIR, load_data_config
from ivai.data.duplicates import find_duplicates
from ivai.data.index import build_index, load_gray, save_boxes_table
from ivai.data.split import SPLITS, materialize_split, save_split_csvs, stratified_group_split


def sha256_file(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    cfg = load_data_config()
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Index -----------------------------------------------------------------
    print("[1/5] Building index ...")
    index = build_index(NEU_RAW_DIR, cfg)
    counts = index["label"].value_counts().reindex(cfg.classes)
    print(counts.to_string())
    problems = []
    if (counts != cfg.expected_per_class).any():
        problems.append("unexpected class counts")
    sizes = index[["width", "height"]].drop_duplicates().values.tolist()
    if sizes != [list(cfg.expected_size)]:
        problems.append(f"unexpected image sizes: {sizes}")
    missing = int(index["box_contains_label"].eq(False).sum())
    if missing:
        problems.append(f"{missing} images whose YOLO boxes do not contain the file-name class")
    multi = int(index["has_secondary_defects"].sum())
    print(f"  sizes={sizes}, modes={index['mode'].unique().tolist()}, "
          f"grayscale content={index['is_grayscale'].mean():.0%}, "
          f"images without boxes={int((index['n_boxes'] == 0).sum())}, "
          f"label absent from boxes={missing}")
    print(f"  INFO: {multi} images ({multi / len(index):.1%}) also contain secondary defect types "
          f"(multi-defect images, label = dominant defect)")

    # 2. Duplicates ------------------------------------------------------------
    print("[2/5] Detecting exact and near duplicates (pHash x 8 transforms + correlation) ...")
    images = [load_gray(NEU_RAW_DIR / p) for p in index["relpath"]]
    groups, pairs, nearest = find_duplicates(
        images, index["pixel_sha256"].tolist(), cfg.phash_max_distance, cfg.min_pixel_correlation
    )
    index["group"] = groups
    index["nearest_phash_distance"] = nearest
    dup_df = pd.DataFrame(pairs, columns=["i", "j", "kind", "hamming", "correlation", "transform"])
    if len(dup_df):
        dup_df["file_i"] = index["filename"].values[dup_df["i"]]
        dup_df["file_j"] = index["filename"].values[dup_df["j"]]
        dup_df["label_i"] = index["label"].values[dup_df["i"]]
        dup_df["label_j"] = index["label"].values[dup_df["j"]]
    dup_df.to_csv(SPLITS_DIR / "neu_cls_duplicates.csv", index=False)
    n_groups_multi = int((index["group"].value_counts() > 1).sum())
    print(f"  exact pairs={int((dup_df['kind'] == 'exact').sum()) if len(dup_df) else 0}, "
          f"near pairs={int((dup_df['kind'] == 'near').sum()) if len(dup_df) else 0}, "
          f"groups with >1 image={n_groups_multi}")
    if len(dup_df) and (dup_df["label_i"] != dup_df["label_j"]).any():
        n = int((dup_df["label_i"] != dup_df["label_j"]).sum())
        print(f"  WARNING: {n} duplicate pair(s) have different labels (possible label noise)")

    # 3. Split -----------------------------------------------------------------
    print(f"[3/5] Stratified group split (seed={cfg.seed}, val={cfg.val_ratio}, test={cfg.test_ratio}) ...")
    index = stratified_group_split(index, cfg.val_ratio, cfg.test_ratio, cfg.seed)
    print(pd.crosstab(index["label"], index["split"]).reindex(columns=list(SPLITS)).to_string())

    # 4. Save ------------------------------------------------------------------
    print("[4/5] Saving CSV files ...")
    index.to_csv(SPLITS_DIR / "neu_cls_index.csv", index=False)
    save_boxes_table(NEU_RAW_DIR, index, SPLITS_DIR / "neu_cls_boxes.csv")
    paths = save_split_csvs(index, SPLITS_DIR)
    meta = {
        "dataset": cfg.name,
        "source_url": cfg.source_url,
        "archive_md5": cfg.md5,
        "seed": cfg.seed,
        "val_ratio": cfg.val_ratio,
        "test_ratio": cfg.test_ratio,
        "classes": list(cfg.classes),
        "counts": {s: int((index["split"] == s).sum()) for s in SPLITS},
        "duplicate_pairs": len(dup_df),
        "csv_sha256": {s: sha256_file(p) for s, p in paths.items()},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (SPLITS_DIR / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 5. Materialise -----------------------------------------------------------
    print(f"[5/5] Copying images to {NEU_PROCESSED_DIR} ...")
    materialize_split(index, NEU_RAW_DIR, NEU_PROCESSED_DIR)

    if problems:
        print("PROBLEMS: " + "; ".join(problems), file=sys.stderr)
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
