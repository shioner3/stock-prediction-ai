"""v3/residual/reproduce.py - target_column_for mapping and primary
spread reproduction check.
"""

from __future__ import annotations

import pandas as pd

from v3.residual.reproduce import (
    TARGET_A_RAW,
    TARGET_B_TOPIX_RELATIVE,
    TARGET_C_BETA_RESIDUAL,
    TARGET_D_SECTOR_RELATIVE,
    target_column_for,
    verify_primary_reproduction,
)
from v3.robustness.v3_3_reference import PRIMARY_Q5_Q1_SPREAD_REFERENCE


def test_target_column_for_all_definitions() -> None:
    assert target_column_for(TARGET_A_RAW, 5) == "target_raw_5d"
    assert target_column_for(TARGET_B_TOPIX_RELATIVE, 10) == "target_topix_relative_10d"
    assert target_column_for(TARGET_C_BETA_RESIDUAL, 15) == "target_beta_residual_15d"
    assert target_column_for(TARGET_D_SECTOR_RELATIVE, 20) == "target_sector_relative_20d"


def test_target_column_for_unknown_definition_raises() -> None:
    try:
        target_column_for("bogus", 5)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_verify_primary_reproduction_matches_reference_spread() -> None:
    # Construct predictions whose Q5-Q1 spread reproduces the V3-3
    # reference value exactly - 10 tickers, prediction == actual so
    # Q1..Q5 buckets separate cleanly.
    n = 10
    predictions = pd.DataFrame({
        "date": ["2023-01-02"] * n,
        "ticker": [f"T{i}" for i in range(n)],
        "prediction": list(range(n)),
        # buckets are 2 rows each (n=10, q=5); Q5 = indices 8,9.
        "actual": [0.0] * 8 + [PRIMARY_Q5_Q1_SPREAD_REFERENCE, PRIMARY_Q5_Q1_SPREAD_REFERENCE],
    })
    assert verify_primary_reproduction(predictions) is True


def test_verify_primary_reproduction_rejects_wrong_spread() -> None:
    n = 10
    predictions = pd.DataFrame({
        "date": ["2023-01-02"] * n,
        "ticker": [f"T{i}" for i in range(n)],
        "prediction": list(range(n)),
        "actual": [0.0] * (n - 1) + [0.5],
    })
    assert verify_primary_reproduction(predictions) is False
