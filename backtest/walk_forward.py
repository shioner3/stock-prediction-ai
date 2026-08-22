"""Walk Forward window generation (Phase 6): TRAIN -> VALIDATION -> OOS,
stepping forward in fixed calendar increments. A pure, deterministic
function of (data_start, data_end, config) - never shuffles, never
samples randomly, and never looks at any Signal/Score/Backtest result to
decide where windows fall (train_test_split/k-fold/random shuffling are
explicitly forbidden by the Phase 6 spec).

Windows are generated once, from calendar dates only, BEFORE any
Signal/Score is evaluated against them - this is what "OOS period fixed
before looking at results" means structurally, not just as a promise.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from config.loader import WalkForwardConfig


@dataclass(frozen=True)
class WalkForwardWindow:
    index: int
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    oos_start: date
    oos_end: date
    # True when data_end cut the OOS period shorter than
    # config.oos_months would otherwise produce - still included as long
    # as it clears config.min_oos_completeness, but flagged so callers
    # can treat it differently if they choose to.
    oos_truncated: bool

    def contains_train(self, d: Any) -> bool:
        return self.train_start <= d < self.train_end

    def contains_validation(self, d: Any) -> bool:
        return self.validation_start <= d < self.validation_end

    def contains_oos(self, d: Any) -> bool:
        return self.oos_start <= d <= self.oos_end


def generate_windows(
    data_start: date, data_end: date, config: WalkForwardConfig
) -> list[WalkForwardWindow]:
    """[train_start, train_end) / [validation_start, validation_end) /
    [oos_start, oos_end] - each window's train_start is the previous
    window's train_start + step_months (windows overlap in TRAIN by
    design; OOS periods are sequential and non-overlapping as long as
    step_months >= oos_months, which is the default and every value
    committed to config/settings.yaml for this project).
    """
    windows: list[WalkForwardWindow] = []
    train_start = pd.Timestamp(data_start)
    data_end_ts = pd.Timestamp(data_end)
    index = 0

    while True:
        train_end = train_start + pd.DateOffset(months=config.train_months)
        validation_start = train_end
        validation_end = validation_start + pd.DateOffset(months=config.validation_months)
        oos_start = validation_end
        oos_end_full = oos_start + pd.DateOffset(months=config.oos_months)

        if oos_start >= data_end_ts:
            break

        oos_end = min(oos_end_full, data_end_ts)
        full_span_days = (oos_end_full - oos_start).days
        actual_span_days = (oos_end - oos_start).days
        completeness = actual_span_days / full_span_days if full_span_days > 0 else 0.0
        truncated = oos_end < oos_end_full

        if completeness >= config.min_oos_completeness:
            windows.append(
                WalkForwardWindow(
                    index=index,
                    train_start=train_start.date(),
                    train_end=train_end.date(),
                    validation_start=validation_start.date(),
                    validation_end=validation_end.date(),
                    oos_start=oos_start.date(),
                    oos_end=oos_end.date(),
                    oos_truncated=truncated,
                )
            )
            index += 1

        if truncated:
            # Data ran out - stepping further forward would only produce
            # windows with even less OOS data, never more.
            break

        train_start = train_start + pd.DateOffset(months=config.step_months)

    return windows
