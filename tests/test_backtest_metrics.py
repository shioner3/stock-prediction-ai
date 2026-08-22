from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.metrics import compute_metrics, compute_metrics_by_signal


def _trades(returns: list[float], signal_name: str = "sig") -> pd.DataFrame:
    n = len(returns)
    return pd.DataFrame(
        {
            "ticker": ["TEST"] * n,
            "signal_name": [signal_name] * n,
            "direction": ["LONG"] * n,
            "return": returns,
        }
    )


def test_empty_trades_gives_none_metrics() -> None:
    m = compute_metrics(pd.DataFrame(columns=["return", "signal_name"]))
    assert m.n_trades == 0
    assert m.win_rate is None
    assert m.average_return is None
    assert m.profit_factor is None


def test_basic_metrics_hand_computed() -> None:
    # 3 wins (+0.10,+0.05,+0.05), 2 losses (-0.05,-0.05)
    trades = _trades([0.10, 0.05, 0.05, -0.05, -0.05])
    m = compute_metrics(trades)

    assert m.n_trades == 5
    assert np.isclose(m.win_rate, 3 / 5)
    assert np.isclose(m.average_return, (0.10 + 0.05 + 0.05 - 0.05 - 0.05) / 5)
    assert np.isclose(m.median_return, 0.05)
    assert np.isclose(m.total_return, 0.10)
    assert np.isclose(m.gross_profit, 0.20)
    assert np.isclose(m.gross_loss, -0.10)
    assert np.isclose(m.profit_factor, 0.20 / 0.10)
    assert np.isclose(m.expectancy, m.average_return)


def test_all_wins_gives_infinite_profit_factor() -> None:
    trades = _trades([0.05, 0.03, 0.01])
    m = compute_metrics(trades)
    assert m.gross_loss == 0.0
    assert m.profit_factor == float("inf")


def test_all_flat_trades_gives_none_profit_factor() -> None:
    trades = _trades([0.0, 0.0, 0.0])
    m = compute_metrics(trades)
    assert m.gross_profit == 0.0
    assert m.gross_loss == 0.0
    assert m.profit_factor is None
    assert m.win_rate == 0.0


def test_all_losses_gives_zero_win_rate_and_finite_profit_factor() -> None:
    trades = _trades([-0.02, -0.04])
    m = compute_metrics(trades)
    assert m.win_rate == 0.0
    assert np.isclose(m.gross_profit, 0.0)
    assert np.isclose(m.gross_loss, -0.06)
    assert m.profit_factor == 0.0


def test_compute_metrics_by_signal_groups_correctly() -> None:
    trades = pd.concat(
        [_trades([0.05, 0.05], signal_name="a"), _trades([-0.10], signal_name="b")],
        ignore_index=True,
    )
    result = compute_metrics_by_signal(trades)
    assert set(result.keys()) == {"a", "b"}
    assert result["a"].n_trades == 2
    assert result["b"].n_trades == 1
    assert result["b"].win_rate == 0.0


def test_compute_metrics_by_signal_empty_input() -> None:
    result = compute_metrics_by_signal(pd.DataFrame(columns=["return", "signal_name"]))
    assert result == {}
