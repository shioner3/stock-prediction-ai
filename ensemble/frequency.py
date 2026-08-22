"""Phase 12 section 22/23: occurrence frequency metrics.

A bucket/combination with a strong Forward Return but only a handful of
occurrences per year is not usable for a real swing-selection tool (spec
section 22: "PFが高いが年間数件しか発生しないケースは実運用候補として
扱わない") - this module makes that frequency explicit and independent
of the return/PF numbers themselves, so the Decision Framework
(ensemble/decision.py) can gate on it directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FrequencyMetrics:
    label: str
    total_occurrences: int
    unique_tickers: int
    unique_dates: int
    total_trading_days_in_period: int
    avg_signals_per_day: float
    median_signals_per_day: float
    p5_signals_per_day: float
    p95_signals_per_day: float
    pct_trading_days_with_occurrence: float


def compute_frequency_metrics(
    label: str,
    occurrence_dates: pd.Series,
    occurrence_tickers: pd.Series,
    all_trading_dates: pd.Series,
) -> FrequencyMetrics:
    """occurrence_dates/occurrence_tickers: parallel Series, one row per
    occurrence (e.g. one row per (ticker,date) that fell into this
    bucket). all_trading_dates: every trading date in the analysis
    period (from the TOPIX/market panel), the denominator for "what
    fraction of trading days did this bucket occur on at all" - this is
    NOT the same as avg_signals_per_day (which can exceed 1, since
    multiple tickers can hit this bucket on the same day).
    """
    all_dates = pd.Series(sorted(set(all_trading_dates)))
    per_day_counts = occurrence_dates.value_counts()
    per_day_counts_full = per_day_counts.reindex(all_dates, fill_value=0)

    total_days = len(all_dates)
    days_with_occurrence = int((per_day_counts_full > 0).sum())

    return FrequencyMetrics(
        label=label,
        total_occurrences=len(occurrence_dates),
        unique_tickers=occurrence_tickers.nunique(),
        unique_dates=int((per_day_counts_full > 0).sum()),
        total_trading_days_in_period=total_days,
        avg_signals_per_day=float(per_day_counts_full.mean()) if total_days else 0.0,
        median_signals_per_day=float(per_day_counts_full.median()) if total_days else 0.0,
        p5_signals_per_day=float(np.percentile(per_day_counts_full, 5)) if total_days else 0.0,
        p95_signals_per_day=float(np.percentile(per_day_counts_full, 95)) if total_days else 0.0,
        pct_trading_days_with_occurrence=(
            days_with_occurrence / total_days if total_days else 0.0
        ),
    )
