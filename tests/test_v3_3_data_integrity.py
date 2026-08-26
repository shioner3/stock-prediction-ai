from __future__ import annotations

import pandas as pd

from v2.stats import MAX_PLAUSIBLE_FORWARD_RETURN
from v3.validation.data_integrity import clean_predictions


def test_drops_rows_above_the_plausible_bound() -> None:
    predictions = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "ticker": ["A", "B", "C"],
            "actual": [0.02, MAX_PLAUSIBLE_FORWARD_RETURN + 1.0, -0.01],
            "prediction": [0.1, 0.2, 0.3],
        }
    )
    cleaned = clean_predictions(predictions)
    assert len(cleaned) == 2
    assert "B" not in cleaned["ticker"].to_numpy()


def test_keeps_all_rows_when_nothing_implausible() -> None:
    predictions = pd.DataFrame(
        {"date": ["2024-01-01"], "ticker": ["A"], "actual": [0.03], "prediction": [0.1]}
    )
    cleaned = clean_predictions(predictions)
    assert len(cleaned) == 1


def test_uses_a_different_actual_column_when_given() -> None:
    predictions = pd.DataFrame(
        {"date": ["2024-01-01"], "ticker": ["A"], "y": [MAX_PLAUSIBLE_FORWARD_RETURN + 10]}
    )
    cleaned = clean_predictions(predictions, actual_col="y")
    assert len(cleaned) == 0
