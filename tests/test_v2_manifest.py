from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from v2.manifest import build_v2_manifest, save_v2_manifest


def test_manifest_hashes_are_deterministic() -> None:
    m1 = build_v2_manifest(
        universe_size=10,
        date_range_start=date(2022, 1, 1),
        date_range_end=date(2026, 8, 20),
        feature_list=["return_5d", "sma_20"],
        score_weights={"momentum": 0.25},
        forward_windows=[5, 10, 15, 20],
    )
    m2 = build_v2_manifest(
        universe_size=10,
        date_range_start=date(2022, 1, 1),
        date_range_end=date(2026, 8, 20),
        feature_list=["return_5d", "sma_20"],
        score_weights={"momentum": 0.25},
        forward_windows=[5, 10, 15, 20],
    )
    assert m1.config_hash == m2.config_hash
    assert m1.code_hash == m2.code_hash


def test_manifest_config_hash_differs_from_v1_config_hash() -> None:
    """V2's config_hash must be computed over v2/config/v2_settings.yaml
    alone, never over config/settings.yaml - a genuinely independent
    hash namespace (spec section 20).
    """
    from common.hashing import hash_files
    from pipeline.run_walk_forward import CONFIG_FILES

    manifest = build_v2_manifest(
        universe_size=1, date_range_start=None, date_range_end=None,
        feature_list=[], score_weights={}, forward_windows=[],
    )
    v1_config_hash = hash_files(CONFIG_FILES)
    assert manifest.config_hash != v1_config_hash


def test_manifest_fields_present() -> None:
    manifest = build_v2_manifest(
        universe_size=2880,
        date_range_start=date(2022, 1, 1),
        date_range_end=date(2026, 8, 20),
        feature_list=["return_5d"],
        score_weights={"momentum": 0.25},
        forward_windows=[5, 10, 15, 20],
    )
    assert manifest.version
    assert manifest.universe_size == 2880
    assert manifest.forward_windows == [5, 10, 15, 20]
    assert manifest.score_weights == {"momentum": 0.25}


def test_save_v2_manifest_writes_valid_json(tmp_path: Path) -> None:
    manifest = build_v2_manifest(
        universe_size=1, date_range_start=date(2022, 1, 1), date_range_end=date(2022, 1, 2),
        feature_list=["return_5d"], score_weights={"momentum": 0.25}, forward_windows=[5],
    )
    path = save_v2_manifest(manifest, tmp_path / "manifest.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["universe_size"] == 1
    assert loaded["date_range_start"] == "2022-01-01"
