"""Phase 6.5 section 17: cross-sectional concentration analysis.

A signal that looks profitable in aggregate can still be worthless
evidence if the profit comes from one or two heavily-traded tickers -
this module measures exactly that, from an already-computed Trade
Record DataFrame (backtest/engine.py's output, or any subset of it such
as one signal's aggregate OOS trades). It does not change how trades or
returns are computed; it only summarizes an existing trades DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ConcentrationMetrics:
    n_tickers: int
    n_trades: int
    # Share of TOTAL RETURN (sum of per-trade returns) contributed by the
    # top-K tickers, ranked by each ticker's own summed return,
    # descending. None when total_return == 0 (share is undefined, not
    # zero - see compute_concentration()'s docstring).
    top1_ticker_return_share: float | None
    top5_ticker_return_share: float | None
    top10_ticker_return_share: float | None
    # Share of TOTAL TRADE COUNT contributed by the top-K tickers, ranked
    # by each ticker's own trade count, descending.
    top1_ticker_trade_share: float | None
    top5_ticker_trade_share: float | None
    top10_ticker_trade_share: float | None


def compute_concentration(trades: pd.DataFrame) -> ConcentrationMetrics:
    """top-K return share can exceed 1.0 (or be negative) when the
    non-top-K tickers net to a loss that partly offsets the top-K
    tickers' profit - that is itself a meaningful concentration signal,
    not a bug, so it is reported as-is rather than clipped to [0, 1].
    """
    n_trades = len(trades)
    if n_trades == 0:
        return ConcentrationMetrics(0, 0, None, None, None, None, None, None)

    by_ticker_return = trades.groupby("ticker")["return"].sum().sort_values(ascending=False)
    total_return = float(by_ticker_return.sum())

    def _return_share(k: int) -> float | None:
        if total_return == 0:
            return None
        return float(by_ticker_return.iloc[:k].sum() / total_return)

    by_ticker_count = trades.groupby("ticker").size().sort_values(ascending=False)

    def _trade_share(k: int) -> float:
        return float(by_ticker_count.iloc[:k].sum() / n_trades)

    return ConcentrationMetrics(
        n_tickers=int(trades["ticker"].nunique()),
        n_trades=n_trades,
        top1_ticker_return_share=_return_share(1),
        top5_ticker_return_share=_return_share(5),
        top10_ticker_return_share=_return_share(10),
        top1_ticker_trade_share=_trade_share(1),
        top5_ticker_trade_share=_trade_share(5),
        top10_ticker_trade_share=_trade_share(10),
    )


@dataclass
class TopKShare:
    k: int
    return_share: float | None
    trade_share: float | None


def compute_top_k_shares(trades: pd.DataFrame, k_values: tuple[int, ...]) -> dict[int, TopKShare]:
    """Phase 8 section 5B3: same underlying computation as
    compute_concentration() above (top-K tickers by their own summed
    return / trade count, descending), generalized to an arbitrary set
    of K values (e.g. Top20, which compute_concentration() does not
    provide) - added as a new function rather than changing
    compute_concentration()'s fixed 1/5/10 output, since Phase 6.5's and
    Phase 7's saved reports already depend on that exact shape.
    """
    n_trades = len(trades)
    if n_trades == 0:
        return {k: TopKShare(k, None, None) for k in k_values}

    by_ticker_return = trades.groupby("ticker")["return"].sum().sort_values(ascending=False)
    total_return = float(by_ticker_return.sum())
    by_ticker_count = trades.groupby("ticker").size().sort_values(ascending=False)

    result = {}
    for k in k_values:
        return_share = (
            float(by_ticker_return.iloc[:k].sum() / total_return) if total_return != 0 else None
        )
        trade_share = float(by_ticker_count.iloc[:k].sum() / n_trades)
        result[k] = TopKShare(k, return_share, trade_share)
    return result
