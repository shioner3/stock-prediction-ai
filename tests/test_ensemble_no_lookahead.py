"""Phase 12 section 28/33: No-lookahead and existing-Signal-immutability
checks specific to the Ensemble analysis.

Feature(t)/Signal(t)/Score(t) themselves are already covered by the
existing generic suites (tests/test_no_lookahead.py,
tests/test_backtest_no_lookahead.py, tests/test_scoring_no_lookahead.py,
tests/test_target_leakage.py) - Phase 12 never touches that code, so
those guarantees carry over unchanged. What's NEW here is Ensemble's own
aggregation step: Signal Count / NET / combination bucket membership for
date t must depend ONLY on Signal occurrences dated <= t (in practice,
== t, since Signal Count is a same-day cross-section), never on a future
date's occurrences.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from config.loader import SignalsConfig
from ensemble.signal_count import aggregate_signal_counts
from signals.registry import all_signal_meta

EXPECTED_12_SIGNALS = {
    ("LONG", "long_breakout"),
    ("LONG", "long_ma_rebound"),
    ("LONG", "long_momentum_continuation"),
    ("LONG", "long_oversold_rebound"),
    ("LONG", "long_pullback"),
    ("LONG", "long_volume_breakout"),
    ("SHORT", "short_breakdown"),
    ("SHORT", "short_ma_rejection"),
    ("SHORT", "short_momentum_continuation"),
    ("SHORT", "short_overbought_reversal"),
    ("SHORT", "short_pullback"),
    ("SHORT", "short_volume_breakdown"),
}


def test_exactly_12_signals_registered_unchanged() -> None:
    """Phase 12 section 2: no new Signal, none skipped, none renamed."""
    meta = all_signal_meta(SignalsConfig())
    assert len(meta) == 12
    pairs = {(m.direction.value, m.name) for m in meta}
    assert pairs == EXPECTED_12_SIGNALS


def test_signal_count_for_earlier_date_unaffected_by_later_date_removal() -> None:
    """If Ensemble's Signal Count for an EARLIER date changed depending
    on whether a LATER date's Signal occurrences are present in the
    input, that would prove future information is leaking backward.
    Removing the later rows must leave the earlier date's row identical.
    """
    d1, d2 = date(2026, 1, 5), date(2026, 1, 6)
    full = pd.DataFrame(
        [
            ("7203", d1, "long_pullback", "LONG"),
            ("7203", d1, "long_ma_rebound", "LONG"),
            ("7203", d2, "long_oversold_rebound", "LONG"),
            ("7203", d2, "short_breakdown", "SHORT"),
        ],
        columns=["ticker", "date", "signal_name", "direction"],
    )
    truncated = full[full["date"] == d1]

    out_full = aggregate_signal_counts(full)
    out_truncated = aggregate_signal_counts(truncated)

    row_full = out_full[out_full["date"] == d1].iloc[0]
    row_truncated = out_truncated[out_truncated["date"] == d1].iloc[0]

    assert row_full["long_count"] == row_truncated["long_count"]
    assert row_full["short_count"] == row_truncated["short_count"]
    assert row_full["long_signals"] == row_truncated["long_signals"]
    assert row_full["short_signals"] == row_truncated["short_signals"]


def test_signal_count_never_pools_across_dates() -> None:
    """A Signal firing on ticker X on date d2 must never be counted
    toward ticker X's Signal Count on date d1, even for the same ticker.
    """
    d1, d2 = date(2026, 1, 5), date(2026, 1, 6)
    records = pd.DataFrame(
        [
            ("7203", d1, "long_pullback", "LONG"),
            ("7203", d2, "long_ma_rebound", "LONG"),
            ("7203", d2, "long_oversold_rebound", "LONG"),
        ],
        columns=["ticker", "date", "signal_name", "direction"],
    )
    out = aggregate_signal_counts(records)
    row_d1 = out[out["date"] == d1].iloc[0]
    row_d2 = out[out["date"] == d2].iloc[0]
    assert row_d1["long_count"] == 1
    assert row_d2["long_count"] == 2
