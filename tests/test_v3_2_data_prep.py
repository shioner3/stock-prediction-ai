from __future__ import annotations

import pandas as pd
import pytest

from v3.features.registry import CORE_FEATURE_NAMES
from v3.models.data_prep import (
    assert_no_target_leakage_in_features,
    binary_target,
    feature_matrix,
    prepare_training_set,
)
from v3.targets.registry import TARGET_COLUMN_NAMES


def test_assert_no_target_leakage_passes_for_real_core_features() -> None:
    assert_no_target_leakage_in_features(CORE_FEATURE_NAMES)  # must not raise


def test_assert_no_target_leakage_raises_when_a_target_column_included() -> None:
    with pytest.raises(ValueError, match="TARGET_LEAKAGE_IN_FEATURES"):
        assert_no_target_leakage_in_features([*CORE_FEATURE_NAMES, TARGET_COLUMN_NAMES[0]])


def test_assert_no_target_leakage_raises_for_identifier_columns() -> None:
    with pytest.raises(ValueError, match="identifier"):
        assert_no_target_leakage_in_features([*CORE_FEATURE_NAMES, "date"])


def test_feature_matrix_selects_exactly_core_feature_names() -> None:
    row = {name: 1.0 for name in CORE_FEATURE_NAMES}
    row.update({"date": "2024-01-01", "ticker": "T0", "target_raw_5d": 0.01})
    df = pd.DataFrame([row])
    X = feature_matrix(df)
    assert list(X.columns) == CORE_FEATURE_NAMES


def test_prepare_training_set_drops_nan_target_rows() -> None:
    row = {name: 1.0 for name in CORE_FEATURE_NAMES}
    row["date"] = "2024-01-01"
    row["ticker"] = "T0"
    for col in TARGET_COLUMN_NAMES:
        row[col] = 0.01
    valid_row = dict(row)
    nan_row = dict(row)
    nan_row["target_raw_5d"] = float("nan")
    df = pd.DataFrame([valid_row, nan_row])
    training_set = prepare_training_set(df, "target_raw_5d")
    assert len(training_set.X) == 1
    assert len(training_set.y) == 1


def test_binary_target_is_positive_return_indicator() -> None:
    y = pd.Series([0.02, -0.01, 0.0, 0.05])
    binary = binary_target(y)
    assert list(binary) == [1, 0, 0, 1]
