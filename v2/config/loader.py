"""Typed access to v2/config/v2_settings.yaml.

A SEPARATE Pydantic model tree from config/loader.py::AppConfig -
deliberately not inheriting from or importing V1's config classes, so
V2's config_hash is a pure function of V2's own settings file and never
perturbed by (or coupled to) a V1 config shape change. V2 still reads
V1's DATA (Full Universe OHLCV/Feature Parquet caches) - see
v2/pipeline.py - but that is a data dependency, not a config one.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class V2UniverseConfig(BaseModel):
    """Mirrors config/loader.py::UniverseConfig's filter semantics
    (same segments, same ETF/REIT exclusion) without importing V1's
    class - see v2/__init__.py's module docstring for why V2 keeps its
    own copy of small, generic config shapes rather than importing V1's.
    """

    segments: list[str] = ["Prime", "Standard", "Growth"]
    exclude_etf: bool = True
    exclude_reit: bool = True


class V2ScoreWeightsConfig(BaseModel):
    """V2 Initial Research Score weights (spec section 9) - fixed,
    transparent, NOT tuned against any backtest result. Must sum to 1.0;
    validated in load_v2_config() rather than here so a config load
    failure names the offending file.
    """

    momentum: float = 0.25
    trend: float = 0.20
    volume: float = 0.15
    relative_strength: float = 0.20
    pullback: float = 0.10
    volatility: float = 0.10


class V2Config(BaseModel):
    universe: V2UniverseConfig = V2UniverseConfig()
    forward_windows: list[int] = [5, 10, 15, 20]
    score_weights: V2ScoreWeightsConfig = V2ScoreWeightsConfig()
    candidate_top_n: list[int] = [10, 20, 50]
    # V1's already-fetched Full Universe caches V2 reads from (READ ONLY -
    # v2/ never writes into these directories).
    source_universe_manifest: Path = Path("data/phase7/_universe_fetch_manifest.json")
    source_processed_dir: Path = Path("data/phase7/processed")
    source_features_dir: Path = Path("data/phase7/features")
    # V2's own output directories (see spec section 21).
    v2_features_dir: Path = Path("data/v2/features")
    v2_targets_dir: Path = Path("data/v2/targets")
    v2_rankings_dir: Path = Path("data/v2/rankings")
    v2_candidates_dir: Path = Path("data/v2/candidates")
    v2_manifests_dir: Path = Path("data/v2/manifests")


def load_v2_config(path: Path = Path("v2/config/v2_settings.yaml")) -> V2Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    config = V2Config.model_validate(raw or {})
    total_weight = (
        config.score_weights.momentum
        + config.score_weights.trend
        + config.score_weights.volume
        + config.score_weights.relative_strength
        + config.score_weights.pullback
        + config.score_weights.volatility
    )
    if abs(total_weight - 1.0) > 1e-9:
        raise ValueError(f"V2 score_weights must sum to 1.0, got {total_weight}")
    return config
