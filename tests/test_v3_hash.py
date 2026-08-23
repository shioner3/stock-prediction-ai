from __future__ import annotations

import pandas as pd

from v3.hash import (
    build_v3_manifest,
    current_v3_code_hash,
    current_v3_config_hash,
    current_v3_feature_hash,
    hash_dataframe,
    save_v3_manifest,
)


def test_hashes_are_deterministic() -> None:
    assert current_v3_code_hash() == current_v3_code_hash()
    assert current_v3_config_hash() == current_v3_config_hash()
    assert current_v3_feature_hash() == current_v3_feature_hash()


def test_hash_dataframe_stable_under_row_reordering() -> None:
    df = pd.DataFrame(
        {"date": ["2024-01-02", "2024-01-01"], "ticker": ["B", "A"], "x": [2.0, 1.0]}
    )
    reordered = df.iloc[::-1].reset_index(drop=True)
    assert hash_dataframe(df) == hash_dataframe(reordered)


def test_hash_dataframe_changes_with_content() -> None:
    df1 = pd.DataFrame({"date": ["2024-01-01"], "ticker": ["A"], "x": [1.0]})
    df2 = pd.DataFrame({"date": ["2024-01-01"], "ticker": ["A"], "x": [2.0]})
    assert hash_dataframe(df1) != hash_dataframe(df2)


def test_manifest_never_has_a_model_hash_in_phase_v3_1() -> None:
    manifest = build_v3_manifest(
        universe_size=1, date_range_start=None, date_range_end=None,
        feature_list=["return_5d"], target_list=["target_raw_5d"], horizons=[5],
    )
    assert manifest.model_hash is None


def test_save_v3_manifest_writes_valid_json(tmp_path) -> None:
    manifest = build_v3_manifest(
        universe_size=1, date_range_start=None, date_range_end=None,
        feature_list=["return_5d"], target_list=["target_raw_5d"], horizons=[5],
    )
    path = tmp_path / "manifest.json"
    save_v3_manifest(manifest, path)
    assert path.exists()
    assert "code_hash" in path.read_text(encoding="utf-8")
