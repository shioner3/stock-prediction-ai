"""Phase 12 section 8-11: Signal Count / Direction Consensus aggregation.

Reads already-computed Signal Records (signals/pipeline.py output, one
row per (ticker, date, signal_name, direction) where that Signal
triggered - storage/parquet_store.py::load_signal_records only ever
persists triggered=True rows) across ALL 12 frozen Signals and
aggregates them per (ticker, date) into LONG_COUNT / SHORT_COUNT /
TOTAL_SIGNAL_COUNT / NET_SIGNAL_COUNT / DOMINANT_DIRECTION /
DIRECTION_CONSENSUS, plus the sorted tuple of triggered Signal names per
direction (consumed by ensemble/combinations.py). Only same-day,
already-triggered Signal rows are read - never a future date's rows -
see tests/test_ensemble_no_lookahead.py.

This module reads Signal output; it must never be imported by
features/, signals/, or scoring/ - see
tests/test_ensemble_no_lookahead.py's AST dependency-direction check
(mirrors tests/test_phase9_no_lookahead.py's existing pattern).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LONG_COUNT_BUCKET_ORDER = ["1", "2", "3", "4+"]
SHORT_COUNT_BUCKET_ORDER = ["1", "2", "3", "4+"]
# Fixed before running any Phase 12 analysis (spec section 10) - never
# adjusted after seeing results.
NET_SIGNAL_COUNT_BUCKET_ORDER = [
    "<=-4", "-3", "-2", "-1", "0", "+1", "+2", "+3", ">=+4",
]


def signal_count_bucket(n: int) -> str:
    """Only meaningful for n >= 1 - a direction with 0 triggered Signals
    has no bucket membership at all (spec section 10 only defines
    1/2/3/4+, no "0" bucket).
    """
    return "4+" if n >= 4 else str(n)


def net_signal_count_bucket(net: int) -> str:
    if net <= -4:
        return "<=-4"
    if net >= 4:
        return ">=+4"
    if net == 0:
        return "0"
    return f"{net:+d}"


def aggregate_signal_counts(all_signal_records: pd.DataFrame) -> pd.DataFrame:
    """all_signal_records: concatenation of load_signal_records() output
    across ALL 12 Signals x ALL tickers (columns include at least
    ticker, date, signal_name, direction) - every row is already a
    triggered occurrence, so this function only counts/groups, it never
    re-evaluates trigger conditions.

    Returns one row per (ticker, date) that had >=1 triggered Signal in
    EITHER direction, with columns:
        ticker, date, long_count, short_count, total_signal_count,
        net_signal_count, dominant_direction ("LONG"/"SHORT"/"NEUTRAL"),
        direction_consensus (max(long,short)/total),
        long_signals (tuple[str,...], sorted), short_signals (tuple).
    """
    if all_signal_records.empty:
        return pd.DataFrame(
            columns=[
                "ticker", "date", "long_count", "short_count", "total_signal_count",
                "net_signal_count", "dominant_direction", "direction_consensus",
                "long_signals", "short_signals",
            ]
        )

    long_df = all_signal_records[all_signal_records["direction"] == "LONG"]
    short_df = all_signal_records[all_signal_records["direction"] == "SHORT"]

    def _grouped_names(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            idx = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
            return pd.Series([], index=idx, dtype=object)
        return df.groupby(["ticker", "date"])["signal_name"].apply(lambda s: tuple(sorted(s)))

    long_names = _grouped_names(long_df).rename("long_signals")
    short_names = _grouped_names(short_df).rename("short_signals")

    combined = pd.concat([long_names, short_names], axis=1)
    combined["long_signals"] = combined["long_signals"].apply(
        lambda x: x if isinstance(x, tuple) else ()
    )
    combined["short_signals"] = combined["short_signals"].apply(
        lambda x: x if isinstance(x, tuple) else ()
    )
    combined = combined.reset_index()

    combined["long_count"] = combined["long_signals"].apply(len)
    combined["short_count"] = combined["short_signals"].apply(len)
    combined["total_signal_count"] = combined["long_count"] + combined["short_count"]
    combined["net_signal_count"] = combined["long_count"] - combined["short_count"]
    combined["dominant_direction"] = np.select(
        [
            combined["long_count"] > combined["short_count"],
            combined["short_count"] > combined["long_count"],
        ],
        ["LONG", "SHORT"],
        default="NEUTRAL",
    )
    combined["direction_consensus"] = (
        combined[["long_count", "short_count"]].max(axis=1) / combined["total_signal_count"]
    )

    return combined[
        [
            "ticker", "date", "long_count", "short_count", "total_signal_count",
            "net_signal_count", "dominant_direction", "direction_consensus",
            "long_signals", "short_signals",
        ]
    ]
