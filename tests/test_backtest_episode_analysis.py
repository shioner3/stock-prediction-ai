from __future__ import annotations

import math
from datetime import date

import pandas as pd

from backtest.episode_analysis import (
    compute_episode_metrics,
    compute_rich_episode_metrics,
    identify_regime_episodes,
)


def _regime_df(rows: list[tuple[date, str | None]]) -> pd.DataFrame:
    return pd.DataFrame({"date": [r[0] for r in rows], "regime": [r[1] for r in rows]})


def _d(offset: int) -> date:
    return date(2024, 1, 1 + offset)


def test_identify_regime_episodes_single_contiguous_run() -> None:
    regime_df = _regime_df(
        [(_d(0), "BULL"), (_d(1), "BEAR"), (_d(2), "BEAR"), (_d(3), "BEAR"), (_d(4), "BULL")]
    )
    episodes = identify_regime_episodes(regime_df, "BEAR")
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.start_date == _d(1)
    assert ep.end_date == _d(3)
    assert ep.trading_days == 3
    assert ep.index == 0


def test_identify_regime_episodes_multiple_runs() -> None:
    regime_df = _regime_df(
        [
            (_d(0), "BEAR"), (_d(1), "BULL"),
            (_d(2), "BEAR"), (_d(3), "BEAR"), (_d(4), "BULL"),
            (_d(5), "BEAR"),
        ]
    )
    episodes = identify_regime_episodes(regime_df, "BEAR")
    assert len(episodes) == 3
    assert [e.trading_days for e in episodes] == [1, 2, 1]
    assert [e.index for e in episodes] == [0, 1, 2]
    assert episodes[1].start_date == _d(2)
    assert episodes[1].end_date == _d(3)


def test_identify_regime_episodes_none_present() -> None:
    regime_df = _regime_df([(_d(0), "BULL"), (_d(1), "NEUTRAL")])
    assert identify_regime_episodes(regime_df, "BEAR") == []


def test_identify_regime_episodes_nan_breaks_run() -> None:
    regime_df = _regime_df([(_d(0), "BEAR"), (_d(1), None), (_d(2), "BEAR")])
    episodes = identify_regime_episodes(regime_df, "BEAR")
    assert len(episodes) == 2


def test_identify_regime_episodes_entire_series_is_one_episode() -> None:
    regime_df = _regime_df([(_d(0), "BEAR"), (_d(1), "BEAR")])
    episodes = identify_regime_episodes(regime_df, "BEAR")
    assert len(episodes) == 1
    assert episodes[0].trading_days == 2


def test_identify_regime_episodes_unsorted_input_is_sorted_first() -> None:
    regime_df = _regime_df([(_d(2), "BEAR"), (_d(0), "BEAR"), (_d(1), "BEAR")])
    episodes = identify_regime_episodes(regime_df, "BEAR")
    assert len(episodes) == 1
    assert episodes[0].start_date == _d(0)
    assert episodes[0].end_date == _d(2)


def _trades(rows: list[tuple[str, date, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [r[0] for r in rows],
            "signal_date": [r[1] for r in rows],
            "return": [r[2] for r in rows],
        }
    )


def test_compute_episode_metrics_assigns_trades_within_range() -> None:
    regime_df = _regime_df([(_d(0), "BULL"), (_d(1), "BEAR"), (_d(2), "BEAR"), (_d(3), "BULL")])
    episodes = identify_regime_episodes(regime_df, "BEAR")

    trades = _trades(
        [
            ("A", _d(0), 0.01),  # outside the episode
            ("B", _d(1), 0.05),  # inside
            ("C", _d(2), -0.02),  # inside
            ("D", _d(3), 0.03),  # outside
        ]
    )
    results = compute_episode_metrics(trades, episodes)
    assert len(results) == 1
    assert results[0].metrics.n_trades == 2
    assert results[0].cumulative_return is not None
    assert math.isclose(results[0].cumulative_return, 0.05 - 0.02)


def test_compute_episode_metrics_empty_episode_gives_zero_trades() -> None:
    regime_df = _regime_df([(_d(0), "BULL"), (_d(1), "BEAR"), (_d(2), "BULL")])
    episodes = identify_regime_episodes(regime_df, "BEAR")
    trades = _trades([("A", _d(0), 0.01), ("D", _d(2), 0.03)])  # none fall in the BEAR day

    results = compute_episode_metrics(trades, episodes)
    assert len(results) == 1
    assert results[0].metrics.n_trades == 0
    assert results[0].cumulative_return is None


# --- compute_rich_episode_metrics (Phase 9) -----------------------------------


def test_rich_episode_metrics_unique_tickers_and_median() -> None:
    regime_df = _regime_df([(_d(0), "BEAR"), (_d(1), "BEAR")])
    episodes = identify_regime_episodes(regime_df, "BEAR")
    trades = _trades([("A", _d(0), 0.05), ("A", _d(1), 0.01), ("B", _d(1), -0.03)])

    results = compute_rich_episode_metrics(trades, episodes)
    assert len(results) == 1
    r = results[0]
    assert r.unique_tickers == 2  # A and B, A appears twice
    assert math.isclose(r.median_trade_return, 0.01)


def test_rich_episode_metrics_pnl_contribution_sums_to_one_across_episodes() -> None:
    regime_df = _regime_df(
        [(_d(0), "BEAR"), (_d(1), "BULL"), (_d(2), "BEAR"), (_d(3), "BEAR")]
    )
    episodes = identify_regime_episodes(regime_df, "BEAR")
    assert len(episodes) == 2

    trades = _trades(
        [
            ("A", _d(0), 0.09),  # episode 0: cum=0.09
            ("B", _d(2), 0.01),  # episode 1: cum=0.01
            ("C", _d(3), 0.00),
        ]
    )
    results = compute_rich_episode_metrics(trades, episodes)
    shares = [r.pnl_contribution_share for r in results]
    assert all(s is not None for s in shares)
    assert math.isclose(sum(shares), 1.0)
    assert results[0].pnl_contribution_share > results[1].pnl_contribution_share


def test_rich_episode_metrics_max_drawdown_on_losing_streak() -> None:
    regime_df = _regime_df([(_d(i), "BEAR") for i in range(4)])
    episodes = identify_regime_episodes(regime_df, "BEAR")
    # Cumulative curve: 0.05, 0.03 (-0.02 dd), -0.02 (-0.07 dd from peak), 0.01
    trades = _trades(
        [("A", _d(0), 0.05), ("B", _d(1), -0.02), ("C", _d(2), -0.05), ("D", _d(3), 0.03)]
    )
    results = compute_rich_episode_metrics(trades, episodes)
    assert results[0].max_drawdown is not None
    assert math.isclose(results[0].max_drawdown, -0.07, abs_tol=1e-9)


def test_rich_episode_metrics_empty_episode_gives_none_fields() -> None:
    regime_df = _regime_df([(_d(0), "BULL"), (_d(1), "BEAR"), (_d(2), "BULL")])
    episodes = identify_regime_episodes(regime_df, "BEAR")
    trades = _trades([("A", _d(0), 0.01)])  # nothing in the BEAR day

    results = compute_rich_episode_metrics(trades, episodes)
    assert results[0].unique_tickers == 0
    assert results[0].median_trade_return is None
    assert results[0].max_drawdown is None
    assert results[0].cumulative_return is None
    assert results[0].pnl_contribution_share is None
