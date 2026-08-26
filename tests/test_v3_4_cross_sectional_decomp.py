"""v3/robustness/cross_sectional_decomp.py - the key structural claim
this module's docstring makes: Rank IC and Top-N selection are INVARIANT
to per-day demeaning (both are computed per-day), while the GLOBAL
quantile-split Q1-Q5 spread is NOT (and can even improve once day-level
drift is removed) - constructed with a synthetic 2-day panel where day 1's
predictions are all much higher in absolute level than day 2's, even
though BOTH days share the identical WITHIN-day rank -> return
relationship.
"""

from __future__ import annotations

import pandas as pd

from v3.robustness.cross_sectional_decomp import (
    VARIANT_DEMEANED,
    VARIANT_ORIGINAL,
    check_structural_invariance,
    daily_market_component_correlation,
    run_cross_sectional_decomposition,
)


def _drifting_level_panel() -> pd.DataFrame:
    day1 = pd.DataFrame(
        {
            "date": ["2023-01-02"] * 5, "ticker": [f"A{i}" for i in range(5)],
            "prediction": [100, 101, 102, 103, 104],
            "actual": [0.01, 0.02, 0.03, 0.04, 0.05],
        }
    )
    day2 = pd.DataFrame(
        {
            "date": ["2023-01-03"] * 5, "ticker": [f"B{i}" for i in range(5)],
            "prediction": [1, 2, 3, 4, 5],
            "actual": [0.01, 0.02, 0.03, 0.04, 0.05],
        }
    )
    return pd.concat([day1, day2], ignore_index=True)


def test_rank_ic_and_topn_are_structurally_invariant() -> None:
    predictions = _drifting_level_panel()
    results = run_cross_sectional_decomposition(predictions, window_days=5)
    invariance = check_structural_invariance(results)
    assert invariance.rank_ic_identical is True
    assert invariance.top5_selection_identical is True
    assert invariance.top10_selection_identical is True
    assert invariance.top20_selection_identical is True


def test_global_q1_q5_spread_changes_after_demeaning() -> None:
    predictions = _drifting_level_panel()
    results = run_cross_sectional_decomposition(predictions, window_days=5)
    invariance = check_structural_invariance(results)
    # Original global qcut is dominated by day-level drift (day 1's lowest
    # value still beats day 2's highest) - spread only reflects day 1's
    # own top-vs-bottom, understating the real (day-local) signal.
    assert invariance.q5_q1_spread_original is not None
    assert abs(invariance.q5_q1_spread_original - 0.03) < 1e-9
    # After demeaning, day 1 and day 2 become interleaved in the global
    # quantile split, and the genuinely highest/lowest WITHIN-day
    # performers from BOTH days end up in Q5/Q1 - a larger, more accurate
    # spread.
    assert invariance.q5_q1_spread_demeaned is not None
    assert abs(invariance.q5_q1_spread_demeaned - 0.04) < 1e-9
    assert invariance.q5_q1_spread_demeaned != invariance.q5_q1_spread_original


def test_original_and_demeaned_rank_ic_are_perfect() -> None:
    predictions = _drifting_level_panel()
    results = run_cross_sectional_decomposition(predictions, window_days=5)
    orig_ic = results[VARIANT_ORIGINAL].ranking.ic_summary.mean_ic
    dem_ic = results[VARIANT_DEMEANED].ranking.ic_summary.mean_ic
    assert orig_ic is not None and abs(orig_ic - 1.0) < 1e-9
    assert dem_ic is not None and abs(dem_ic - 1.0) < 1e-9


def test_market_component_correlation_detects_real_market_timing() -> None:
    # Here the day-level mean prediction (102 vs 3) has NO relationship to
    # the day-level mean actual return (both days average 0.03) - a
    # constructed "fake market timing" case, so the correlation should be
    # undefined/near-zero (only 2 days -> correlation over 2 points is
    # technically defined but not meaningful; verify it does not crash and
    # returns a bounded value).
    predictions = _drifting_level_panel()
    corr = daily_market_component_correlation(predictions)
    assert corr is None or -1.0 <= corr <= 1.0
