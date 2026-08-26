"""v3/frozen/realize_returns.py - maturity detection: a pending
prediction only gets a realized_return entry once its Horizon's forward
Target column is non-NaN at (observation_date, ticker); already-realized
keys are never re-emitted.
"""

from __future__ import annotations

import pandas as pd

from v3.frozen.realize_returns import compute_realized_return_entries


def _augmented_dataset() -> pd.DataFrame:
    dates = pd.to_datetime(["2026-08-21", "2026-08-22"])
    return pd.DataFrame({
        "date": [dates[0], dates[0], dates[1], dates[1]],
        "ticker": ["7203", "9984", "7203", "9984"],
        "target_raw_5d": [0.02, None, 0.03, 0.01],  # 9984 not yet matured on 2026-08-21
        "target_beta_residual_5d": [0.01, None, None, None],
    })


def test_matured_prediction_is_realized() -> None:
    pending = [{
        "observation_date": "2026-08-21", "ticker": "7203", "model_id": "V3-FROZEN-RAW-5D",
        "target_definition": "raw", "horizon": 5,
    }]
    entries = compute_realized_return_entries(pending, _augmented_dataset())
    assert len(entries) == 1
    assert abs(entries[0].realized_return - 0.02) < 1e-9
    assert entries[0].observation_date == "2026-08-21"
    assert entries[0].ticker == "7203"


def test_unmatured_prediction_is_not_realized() -> None:
    pending = [{
        "observation_date": "2026-08-21", "ticker": "9984", "model_id": "V3-FROZEN-RAW-5D",
        "target_definition": "raw", "horizon": 5,
    }]
    entries = compute_realized_return_entries(pending, _augmented_dataset())
    assert entries == []


def test_already_realized_key_is_skipped() -> None:
    pending = [{
        "observation_date": "2026-08-21", "ticker": "7203", "model_id": "V3-FROZEN-RAW-5D",
        "target_definition": "raw", "horizon": 5,
    }]
    already_realized = {("2026-08-21", "7203", "V3-FROZEN-RAW-5D")}
    entries = compute_realized_return_entries(pending, _augmented_dataset(), already_realized)
    assert entries == []


def test_different_targets_use_their_own_column() -> None:
    pending = [{
        "observation_date": "2026-08-21", "ticker": "7203", "model_id": "V3-FROZEN-BETA-5D",
        "target_definition": "beta_residual", "horizon": 5,
    }]
    entries = compute_realized_return_entries(pending, _augmented_dataset())
    assert len(entries) == 1
    assert abs(entries[0].realized_return - 0.01) < 1e-9
