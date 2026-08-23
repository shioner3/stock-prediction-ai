"""Typed access to v3/config/v3_settings.yaml.

A SEPARATE Pydantic model tree from config/loader.py::AppConfig AND from
v2/config/loader.py::V2Config (spec section 1 rule 6: "V3は完全に独立し
たnamespace/package/config/hashを持つ") - V3's config_hash is a pure
function of V3's own settings file, never perturbed by a V1 or V2 config
shape change. V3 still READS V1's raw OHLCV data and V1's pure Feature
formulas (see v3/__init__.py's module docstring) - that is a data/code
dependency, not a config one.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class V3UniverseConfig(BaseModel):
    segments: list[str] = ["Prime", "Standard", "Growth"]
    exclude_etf: bool = True
    exclude_reit: bool = True


class V3HorizonsConfig(BaseModel):
    # Spec section 2's HORIZONS list - fixed here, never chosen per-run.
    days: list[int] = [5, 10, 15, 20]


class V3Config(BaseModel):
    universe: V3UniverseConfig = V3UniverseConfig()
    horizons: V3HorizonsConfig = V3HorizonsConfig()
    # V1's already-fetched Full Universe caches V3 reads from (READ ONLY -
    # v3/ never writes into these directories) - same source data V2 reads,
    # a data dependency only (see module docstring).
    source_universe_manifest: Path = Path("data/phase7/_universe_fetch_manifest.json")
    source_processed_dir: Path = Path("data/phase7/processed")
    source_features_dir: Path = Path("data/phase7/features")
    # V3's own output directories.
    v3_dataset_dir: Path = Path("data/v3/dataset")
    v3_manifests_dir: Path = Path("data/v3/manifests")
    v3_models_dir: Path = Path("data/v3/models")
    v3_reports_dir: Path = Path("data/v3/reports")


def load_v3_config(path: Path = Path("v3/config/v3_settings.yaml")) -> V3Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return V3Config.model_validate(raw or {})
