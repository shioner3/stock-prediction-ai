"""Market Regime cross-tab (spec section 12): reuses V1's Market Regime
classification UNCHANGED (backtest.market_regime.compute_market_regime,
its default thresholds from config/settings.yaml) - spec section 12's
own explicit instruction ("既存のV2/V1で定義済みのMarket Regimeがある
場合は、それをそのまま使用する。新しいRegime定義を作らない。"). No new
Regime definition is introduced here; this module only joins V1's
regime series onto V2's panel by date and re-runs the SAME Q1-Q5/IC
analysis already built (v2/stats.py, v2/validation/ic.py) within each
regime slice.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.market_regime import compute_market_regime
from config.loader import MarketRegimeConfig
from v2.stats import QuantileBucketStats, compute_q5_q1_spread, compute_quantile_bucket_stats
from v2.validation.ic import ICSummary, compute_daily_spearman_ic, summarize_ic

REGIMES = ("BULL", "NEUTRAL", "BEAR")


def build_regime_series(topix_ohlcv: pd.DataFrame, config: MarketRegimeConfig) -> pd.DataFrame:
    return compute_market_regime(topix_ohlcv, config)


@dataclass(frozen=True)
class RegimeSliceResult:
    regime: str
    window_days: int
    n: int
    bucket_stats: list[QuantileBucketStats]
    q5_q1_spread: float | None
    ic_summary: ICSummary


def analyze_by_regime(
    scored_panel: pd.DataFrame,
    regime_df: pd.DataFrame,
    return_col: str,
    window_days: int,
    score_col: str = "total_score",
    bucket_col: str = "score_bucket",
    date_col: str = "date",
) -> list[RegimeSliceResult]:
    merged = scored_panel.merge(regime_df, on=date_col, how="left")
    results = []
    for regime in REGIMES:
        subset = merged[merged["regime"] == regime]
        window_subset = subset.dropna(subset=[return_col])
        bucket_stats = compute_quantile_bucket_stats(
            window_subset, bucket_col, return_col, window_days
        )
        daily_ic = compute_daily_spearman_ic(subset, score_col, return_col, date_col)
        results.append(
            RegimeSliceResult(
                regime=regime, window_days=window_days, n=len(window_subset),
                bucket_stats=bucket_stats,
                q5_q1_spread=compute_q5_q1_spread(bucket_stats),
                ic_summary=summarize_ic(daily_ic, window_days),
            )
        )
    return results
