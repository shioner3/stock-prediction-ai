"""v3/robustness/matched_control.py - matching-tier fallback logic
verified on a small constructed panel where the correct match (or
fallback tier) is known by construction.
"""

from __future__ import annotations

import pandas as pd

from v3.robustness.matched_control import build_full_day_panel, build_matched_control_pairs


def _day_panel_9_large() -> pd.DataFrame:
    # 9 Large tickers, turnover and price perfectly rank-correlated and
    # evenly spaced -> qcut(3) splits unambiguously into 3 groups of 3:
    # {T0,T1,T2} lowest tercile, {T3,T4,T5} mid, {T6,T7,T8} highest.
    return pd.DataFrame(
        {
            "date": ["2023-01-02"] * 9,
            "ticker": [f"T{i}" for i in range(9)],
            "scale": ["Large"] * 9,
            "turnover": [(i + 1) * 100 for i in range(9)],
            "close": [(i + 1) * 10 for i in range(9)],
        }
    )


def test_exact_tier_match_when_available() -> None:
    day_panel = _day_panel_9_large()
    q5_predictions = pd.DataFrame({"date": ["2023-01-02"], "ticker": ["T0"]})
    pairs = build_matched_control_pairs(q5_predictions, day_panel, seed=1)
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["control_ticker"] in {"T1", "T2"}  # T0's own turnover/price tercile
    assert row["match_tier"] == "scale_turnover_price"


def test_no_match_excludes_ticker_from_pairs() -> None:
    day_panel = pd.DataFrame(
        {
            "date": ["2023-01-02"] * 4,
            "ticker": ["T0", "T1", "T2", "T3"],
            "scale": ["Small", "Large", "Large", "Large"],
            "turnover": [10, 100, 200, 300],
            "close": [100, 1000, 2000, 3000],
        }
    )
    q5_predictions = pd.DataFrame({"date": ["2023-01-02"], "ticker": ["T0"]})
    pairs = build_matched_control_pairs(q5_predictions, day_panel, seed=1)
    # T0 is the only "Small" scale ticker on this day - no candidate
    # shares its scale at ANY tier, so it must go unmatched.
    assert len(pairs) == 0


def test_fallback_to_scale_only_tier() -> None:
    # T0's turnover/price value (1000) is the sole occupant of the "high"
    # tercile for BOTH dimensions (verified via pd.qcut on this exact
    # series) - tier 1 (scale+turnover+price) and tier 2 (scale+turnover)
    # both have zero candidates, forcing the scale-only fallback.
    day_panel = pd.DataFrame(
        {
            "date": ["2023-01-02"] * 4,
            "ticker": ["T0", "T1", "T2", "T3"],
            "scale": ["Large", "Large", "Large", "Large"],
            "turnover": [1000, 1, 500, 600],
            "close": [1000, 1, 500, 600],
        }
    )
    q5_predictions = pd.DataFrame({"date": ["2023-01-02"], "ticker": ["T0"]})
    pairs = build_matched_control_pairs(q5_predictions, day_panel, seed=1)
    assert len(pairs) == 1
    assert pairs.iloc[0]["control_ticker"] in {"T1", "T2", "T3"}
    assert pairs.iloc[0]["match_tier"] == "scale_only"


def test_deterministic_given_fixed_seed() -> None:
    day_panel = pd.DataFrame(
        {
            "date": ["2023-01-02"] * 4,
            "ticker": ["T0", "T1", "T2", "T3"],
            "scale": ["Large"] * 4,
            "turnover": [1_000_000] * 4,
            "close": [3000] * 4,
        }
    )
    q5_predictions = pd.DataFrame({"date": ["2023-01-02"], "ticker": ["T0"]})
    pairs_a = build_matched_control_pairs(q5_predictions, day_panel, seed=42)
    pairs_b = build_matched_control_pairs(q5_predictions, day_panel, seed=42)
    assert pairs_a.iloc[0]["control_ticker"] == pairs_b.iloc[0]["control_ticker"]


def test_implausible_return_outcome_masked_but_vol_adjusted_untouched() -> None:
    # Regression test for the real Full Universe bug: an implausible
    # target_raw_10d row (20.0, an artifact) must be masked to NaN, while
    # target_vol_adjusted_5d (a ratio, not return-bounded) is left as-is
    # even at the same magnitude - the two columns are filtered
    # differently on purpose (see build_full_day_panel's own docstring).
    dataset = pd.DataFrame({
        "date": ["2023-01-02", "2023-01-02"], "ticker": ["A", "B"],
        "target_raw_5d": [0.02, 0.03], "target_raw_10d": [0.04, 20.0],
        "target_raw_15d": [0.05, 0.06], "target_raw_20d": [0.07, 0.08],
        "target_topix_relative_5d": [0.01, 0.02], "target_vol_adjusted_5d": [0.3, 20.0],
    })
    price_volume_panel = pd.DataFrame({
        "date": ["2023-01-02", "2023-01-02"], "ticker": ["A", "B"],
        "close": [1000, 1000], "volume": [10000, 10000],
    })
    sector_map = pd.DataFrame({"ticker": ["A", "B"], "scale": ["Large", "Large"]})

    panel = build_full_day_panel(dataset, price_volume_panel, sector_map)
    row_b = panel[panel["ticker"] == "B"].iloc[0]
    row_a = panel[panel["ticker"] == "A"].iloc[0]

    assert pd.isna(row_b["target_raw_10d"])
    assert row_b["target_vol_adjusted_5d"] == 20.0  # deliberately unfiltered
    assert row_a["target_raw_10d"] == 0.04  # unaffected ticker stays intact
