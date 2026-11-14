"""Project paths and configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# ai-engine/ directory (this file lives in ai-engine/ivai/config.py)
ENGINE_DIR = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ENGINE_DIR / "configs"
DATASETS_DIR = ENGINE_DIR / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
PROCESSED_DIR = DATASETS_DIR / "processed"
SPLITS_DIR = DATASETS_DIR / "splits"
REPORTS_DIR = ENGINE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

NEU_RAW_DIR = RAW_DIR / "neu-cls"
NEU_PROCESSED_DIR = PROCESSED_DIR / "neu-cls"


@dataclass(frozen=True)
class DataConfig:
    name: str
    source_url: str
    source_page: str
    md5: str
    classes: tuple[str, ...]
    expected_per_class: int
    expected_size: tuple[int, int]
    seed: int
    val_ratio: float
    test_ratio: float
    phash_max_distance: int
    min_pixel_correlation: float

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.classes)}


def load_data_config(path: Path | None = None) -> DataConfig:
    path = path or CONFIGS_DIR / "data.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ds, sp, dup = raw["dataset"], raw["split"], raw["duplicates"]
    cfg = DataConfig(
        name=ds["name"],
        source_url=ds["source_url"],
        source_page=ds["source_page"],
        md5=ds["md5"].lower(),
        classes=tuple(ds["classes"]),
        expected_per_class=int(ds["expected_per_class"]),
        expected_size=tuple(ds["expected_size"]),
        seed=int(sp["seed"]),
        val_ratio=float(sp["val_ratio"]),
        test_ratio=float(sp["test_ratio"]),
        phash_max_distance=int(dup["phash_max_distance"]),
        min_pixel_correlation=float(dup["min_pixel_correlation"]),
    )
    if not 0 < cfg.val_ratio + cfg.test_ratio < 1:
        raise ValueError("val_ratio + test_ratio must be in (0, 1)")
    return cfg
