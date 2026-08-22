"""Phase 9 section 11: Timing Robustness / Placebo sweep.

Shifts a Signal's own triggered dates by a fixed number of TRADING-DAY
index positions (never calendar days) within each ticker's own date
index, so the result can be re-run through the UNCHANGED Backtest Engine
(backtest/engine.py::run_backtest_for_ticker, untouched). This is a
RETROSPECTIVE DIAGNOSTIC only - it answers "if this exact Signal had
fired `offset_days` trading days earlier/later than it actually did,
would it still show an edge?" - it never feeds back into live Signal
generation (see tests/test_phase9_dependency_direction.py).

Negative offset = earlier, the direction Phase 8 already used (safe by
construction: a backward shift can only reference dates chronologically
<= the original, so it introduces no future information relative to what
the real Signal used). Positive offset = later; still safe for this
offline/retrospective purpose, since the shifted date is always a real,
already-known historical trading day within the SAME already-fetched
dataset - it is never used to make a live decision, only to describe
what a Backtest over already-collected history would have shown.
"""

from __future__ import annotations

import pandas as pd


def shift_signal_records(
    signal_records: pd.DataFrame, panel: pd.DataFrame, offset_days: int
) -> pd.DataFrame:
    """offset_days: signed trading-day INDEX shift within `panel`'s own
    date sequence (negative = earlier, positive = later, 0 = no-op).
    Rows that would shift outside [0, len(panel)-1] are dropped, not
    wrapped or clamped to the boundary.
    """
    dates = panel["date"].tolist()
    pos_by_date = {d: i for i, d in enumerate(dates)}
    n = len(dates)

    rows: list[dict] = []
    for row in signal_records.itertuples(index=False):
        pos = pos_by_date.get(row.date)
        if pos is None:
            continue
        new_pos = pos + offset_days
        if new_pos < 0 or new_pos >= n:
            continue
        shifted = row._asdict()
        shifted["date"] = dates[new_pos]
        rows.append(shifted)

    if not rows:
        return signal_records.iloc[0:0].copy()
    return pd.DataFrame(rows)[list(signal_records.columns)]
