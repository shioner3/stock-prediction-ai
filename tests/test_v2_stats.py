from __future__ import annotations

import pandas as pd
import pytest

from v2.stats import (
    compute_q5_q1_spread,
    compute_quantile_bucket_stats,
    compute_return_stats,
    exclude_implausible_returns,
)


def test_compute_return_stats_basic() -> None:
    stats = compute_return_stats(pd.Series([0.01, -0.02, 0.03, -0.01, 0.02]))
    assert stats.n == 5
    assert stats.win_rate == pytest.approx(3 / 5)
    assert stats.max_loss == pytest.approx(-0.02)
    assert stats.mean_return == pytest.approx(0.006)


def test_compute_return_stats_empty_series() -> None:
    stats = compute_return_stats(pd.Series([], dtype=float))
    assert stats.n == 0
    assert stats.mean_return is None
    assert stats.profit_factor is None


def test_compute_return_stats_drops_nan() -> None:
    stats = compute_return_stats(pd.Series([0.01, float("nan"), 0.02]))
    assert stats.n == 2


def test_profit_factor_all_wins_is_inf() -> None:
    stats = compute_return_stats(pd.Series([0.01, 0.02, 0.03]))
    assert stats.profit_factor == float("inf")


def test_profit_factor_all_losses_is_none() -> None:
    stats = compute_return_stats(pd.Series([-0.01, -0.02]))
    assert stats.profit_factor == 0.0


def test_quantile_bucket_stats_grouped_correctly() -> None:
    scored = pd.DataFrame(
        {
            "bucket": ["Q1", "Q1", "Q5", "Q5", "Q5"],
            "forward_return_5d": [0.01, -0.01, 0.05, 0.04, 0.06],
        }
    )
    results = compute_quantile_bucket_stats(scored, "bucket", "forward_return_5d", 5)
    by_bucket = {r.bucket: r.stats for r in results}
    assert by_bucket["Q1"].n == 2
    assert by_bucket["Q5"].n == 3
    assert by_bucket["Q5"].mean_return > by_bucket["Q1"].mean_return


def test_q5_q1_spread() -> None:
    scored = pd.DataFrame(
        {
            "bucket": ["Q1", "Q1", "Q5", "Q5"],
            "forward_return_5d": [0.0, 0.0, 0.10, 0.10],
        }
    )
    results = compute_quantile_bucket_stats(scored, "bucket", "forward_return_5d", 5)
    spread = compute_q5_q1_spread(results)
    assert spread == pytest.approx(0.10)


def test_q5_q1_spread_none_when_bucket_missing() -> None:
    scored = pd.DataFrame({"bucket": ["Q1", "Q1"], "forward_return_5d": [0.0, 0.01]})
    results = compute_quantile_bucket_stats(scored, "bucket", "forward_return_5d", 5)
    assert compute_q5_q1_spread(results) is None


def test_exclude_implausible_returns_drops_only_extreme_rows() -> None:
    df = pd.DataFrame({"forward_return_5d": [0.01, -0.02, 19785410.0, 0.03]})
    clean = exclude_implausible_returns(df, "forward_return_5d")
    assert len(clean) == 3
    assert 19785410.0 not in clean["forward_return_5d"].to_numpy()


def test_exclude_implausible_returns_respects_custom_bound() -> None:
    df = pd.DataFrame({"forward_return_5d": [0.5, 1.5, -1.5]})
    clean = exclude_implausible_returns(df, "forward_return_5d", max_abs_return=1.0)
    assert len(clean) == 1
    assert clean["forward_return_5d"].iloc[0] == 0.5


def test_exclude_implausible_returns_never_mutates_input() -> None:
    df = pd.DataFrame({"forward_return_5d": [0.01, 100.0]})
    exclude_implausible_returns(df, "forward_return_5d")
    assert len(df) == 2
