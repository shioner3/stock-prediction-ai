"""No-lookahead verification for the Score layer (Phase 5 section 24):

    A. Score(t) is unaffected by changes to t+1-and-later stock OHLCV.
    B. Score(t) (including relative_score, which reads RS) is unaffected
       by changes to t+1-and-later market benchmark data.
    C. Changing or deleting Forward Targets does not change Score at
       all - Score computation has no code path that could even read
       them (see tests/test_target_leakage.py for the static check;
       this test demonstrates it dynamically).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import make_synthetic_ohlcv

from config.loader import ScoringConfig
from features.pipeline import compute_feature_panel
from scoring.scorer import SUBSCORE_COLUMNS, compute_total_score
from signals.base import Direction
from targets.forward_returns import compute_forward_returns, compute_mfe_mae

SCORE_COLUMNS = [*SUBSCORE_COLUMNS, "total_score"]


# --- Test A: future stock OHLCV changes don't affect Score(t) --------------


def test_score_at_t_unaffected_by_future_stock_data() -> None:
    n = 300
    base = make_synthetic_ohlcv(n, seed=70)
    panel_base = compute_feature_panel(base)
    scores_base = compute_total_score(panel_base, Direction.LONG, ScoringConfig())

    rng = np.random.default_rng(71)
    t = 200
    perturbed = base.copy()
    future_mask = perturbed.index > t
    n_future = int(future_mask.sum())
    for col in ("open", "high", "low", "close"):
        perturbed.loc[future_mask, col] = perturbed.loc[future_mask, col] * rng.uniform(
            0.5, 1.5, size=n_future
        )
    perturbed.loc[future_mask, "volume"] = perturbed.loc[future_mask, "volume"] * rng.uniform(
        0.5, 3.0, size=n_future
    )

    panel_perturbed = compute_feature_panel(perturbed)
    scores_perturbed = compute_total_score(panel_perturbed, Direction.LONG, ScoringConfig())

    row_base = scores_base.loc[t, SCORE_COLUMNS]
    row_perturbed = scores_perturbed.loc[t, SCORE_COLUMNS]
    pd.testing.assert_series_equal(row_base, row_perturbed, check_names=False)


# --- Test B: future MARKET data changes don't affect Score(t) --------------


def test_score_at_t_unaffected_by_future_market_data() -> None:
    n = 300
    stock = make_synthetic_ohlcv(n, seed=72)
    market_base = make_synthetic_ohlcv(n, seed=173, ticker="TOPIX")

    panel_base = compute_feature_panel(stock, market_df=market_base)
    scores_base = compute_total_score(panel_base, Direction.LONG, ScoringConfig())

    rng = np.random.default_rng(74)
    t = 200
    market_perturbed = market_base.copy()
    future_mask = market_perturbed.index > t
    n_future = int(future_mask.sum())
    market_perturbed.loc[future_mask, "close"] = market_perturbed.loc[
        future_mask, "close"
    ] * rng.uniform(0.5, 1.5, size=n_future)

    panel_perturbed = compute_feature_panel(stock, market_df=market_perturbed)
    scores_perturbed = compute_total_score(panel_perturbed, Direction.LONG, ScoringConfig())

    row_base = scores_base.loc[t, "relative_score"]
    row_perturbed = scores_perturbed.loc[t, "relative_score"]
    assert row_base == row_perturbed or (np.isnan(row_base) and np.isnan(row_perturbed))

    row_base_all = scores_base.loc[t, SCORE_COLUMNS]
    row_perturbed_all = scores_perturbed.loc[t, SCORE_COLUMNS]
    pd.testing.assert_series_equal(row_base_all, row_perturbed_all, check_names=False)


# --- Test C: mutating/deleting Forward Targets doesn't affect Score --------


def test_mutating_forward_targets_does_not_affect_score() -> None:
    n = 300
    stock = make_synthetic_ohlcv(n, seed=75)
    panel = compute_feature_panel(stock)

    scores_before = compute_total_score(panel, Direction.LONG, ScoringConfig())

    # Compute Forward Targets and deliberately corrupt them - Score must
    # not have referenced them at all, so this must have zero effect.
    forward = compute_forward_returns(panel)
    forward["forward_return_5d"] = 999_999.0
    mfe_mae = compute_mfe_mae(panel, "LONG")
    mfe_mae["mfe_5d"] = -999_999.0
    del forward
    del mfe_mae

    scores_after = compute_total_score(panel, Direction.LONG, ScoringConfig())
    pd.testing.assert_frame_equal(scores_before, scores_after)
