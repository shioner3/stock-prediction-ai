from __future__ import annotations

import pandas as pd
import pytest

from backtest.cross_sectional import compute_concentration, compute_top_k_shares
from backtest.metrics import compute_metrics_by_ticker


def _trades(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [r[0] for r in rows],
            "signal_name": ["sig"] * len(rows),
            "direction": ["LONG"] * len(rows),
            "return": [r[1] for r in rows],
        }
    )


def test_compute_metrics_by_ticker_groups_correctly() -> None:
    trades = _trades([("7203", 0.05), ("7203", -0.02), ("6758", 0.10)])
    result = compute_metrics_by_ticker(trades)
    assert set(result.keys()) == {"7203", "6758"}
    assert result["7203"].n_trades == 2
    assert result["6758"].n_trades == 1
    assert result["6758"].win_rate == 1.0


def test_compute_metrics_by_ticker_empty_input() -> None:
    assert compute_metrics_by_ticker(pd.DataFrame(columns=["return", "ticker"])) == {}


def test_compute_concentration_empty_trades() -> None:
    c = compute_concentration(pd.DataFrame(columns=["ticker", "return"]))
    assert c.n_trades == 0
    assert c.n_tickers == 0
    assert c.top1_ticker_return_share is None
    assert c.top1_ticker_trade_share is None


def test_compute_concentration_single_ticker_is_fully_concentrated() -> None:
    trades = _trades([("7203", 0.05), ("7203", 0.03), ("7203", -0.01)])
    c = compute_concentration(trades)
    assert c.n_tickers == 1
    assert c.n_trades == 3
    assert c.top1_ticker_return_share == 1.0
    assert c.top1_ticker_trade_share == 1.0
    assert c.top5_ticker_return_share == 1.0
    assert c.top10_ticker_trade_share == 1.0


def test_compute_concentration_spread_evenly_across_many_tickers() -> None:
    # 10 tickers, each contributing one identical +0.01 trade - no
    # concentration: the top-1 ticker should own only 1/10 of the total.
    trades = _trades([(f"T{i}", 0.01) for i in range(10)])
    c = compute_concentration(trades)
    assert c.n_tickers == 10
    assert c.top1_ticker_trade_share == 0.1
    assert abs(c.top1_ticker_return_share - 0.1) < 1e-9
    assert c.top5_ticker_trade_share == 0.5
    assert c.top10_ticker_trade_share == 1.0


def test_compute_concentration_return_share_can_exceed_one_when_others_net_negative() -> None:
    # One ticker made all the money; the rest collectively lost money -
    # the "top-1 owns >100% of total profit" case flagged in the module
    # docstring as a legitimate (not clipped) concentration signal.
    trades = _trades([("WINNER", 0.20), ("LOSER1", -0.05), ("LOSER2", -0.05)])
    c = compute_concentration(trades)
    assert c.top1_ticker_return_share > 1.0


def test_compute_concentration_zero_total_return_gives_none_share() -> None:
    trades = _trades([("A", 0.05), ("B", -0.05)])
    c = compute_concentration(trades)
    assert c.top1_ticker_return_share is None
    assert c.top5_ticker_return_share is None
    # Trade share is still well-defined even when return share isn't.
    assert c.top1_ticker_trade_share == 0.5


# --- compute_top_k_shares (Phase 8 section 5B3: Top20 support) --------------


def test_top_k_shares_matches_compute_concentration_for_1_5_10() -> None:
    trades = _trades([(f"T{i}", 0.01 * (i + 1)) for i in range(15)])
    c = compute_concentration(trades)
    shares = compute_top_k_shares(trades, (1, 5, 10))
    assert shares[1].trade_share == c.top1_ticker_trade_share
    assert shares[5].trade_share == c.top5_ticker_trade_share
    assert shares[10].trade_share == c.top10_ticker_trade_share
    assert shares[1].return_share == c.top1_ticker_return_share
    assert shares[5].return_share == c.top5_ticker_return_share
    assert shares[10].return_share == c.top10_ticker_return_share


def test_top_k_shares_supports_top20() -> None:
    trades = _trades([(f"T{i}", 0.01) for i in range(30)])
    shares = compute_top_k_shares(trades, (20,))
    assert shares[20].trade_share == pytest.approx(20 / 30)
    assert shares[20].return_share == pytest.approx(20 / 30)


def test_top_k_shares_k_larger_than_ticker_count_caps_at_100_percent() -> None:
    trades = _trades([("A", 0.01), ("B", 0.01)])
    shares = compute_top_k_shares(trades, (20,))
    assert shares[20].trade_share == 1.0
    assert shares[20].return_share == 1.0


def test_top_k_shares_empty_trades() -> None:
    shares = compute_top_k_shares(pd.DataFrame(columns=["ticker", "return"]), (1, 20))
    assert shares[1].return_share is None
    assert shares[1].trade_share is None
    assert shares[20].return_share is None
    assert shares[20].trade_share is None
