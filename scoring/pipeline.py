"""Combines scoring/scorer.py's Total Score with Signal Records
(signals/pipeline.py's output) into the ScoreRecord shape:

    ticker, date, signal_name, direction,
    trend_score, momentum_score, volume_score, relative_score,
    setup_score, risk_score, total_score

Score is computed ONLY for rows where a Signal actually triggered - no
"virtual" Score is assigned to non-triggered rows (Phase 5 spec section
5). Internally, the Total Score is still computed once per direction
over the WHOLE Feature panel (vectorized, cheap) and then subset down to
the triggered dates - this is an implementation detail, not a change to
what gets saved.
"""

from __future__ import annotations

import pandas as pd

from config.loader import ScoringConfig
from scoring.scorer import SUBSCORE_COLUMNS, compute_total_score
from signals.base import Direction

SCORE_RECORD_COLUMNS = [
    "ticker", "date", "signal_name", "direction", *SUBSCORE_COLUMNS, "total_score",
]


def compute_score_records(
    panel: pd.DataFrame, signal_records: pd.DataFrame, config: ScoringConfig
) -> pd.DataFrame:
    """panel: one ticker's Feature panel. signal_records: that SAME
    ticker's triggered SignalRecord rows (signals/pipeline.py's
    compute_signal_records() output, ticker/date/signal_name/direction/
    triggered/signal_version).
    """
    if signal_records.empty:
        return pd.DataFrame(columns=SCORE_RECORD_COLUMNS)

    score_cols = ["date", *SUBSCORE_COLUMNS, "total_score"]
    scored_groups = []
    for direction_value, group in signal_records.groupby("direction"):
        direction = Direction(direction_value)
        scores = compute_total_score(panel, direction, config)
        scores_by_date = panel[["date"]].join(scores)[score_cols]
        merged = group.merge(scores_by_date, on="date", how="left")
        scored_groups.append(merged)

    out = pd.concat(scored_groups, ignore_index=True)
    out = out.sort_values(["date", "signal_name"]).reset_index(drop=True)
    return out[SCORE_RECORD_COLUMNS]
