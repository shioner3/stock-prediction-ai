"""Thin wrapper around `backtest.walk_forward.generate_windows()` (V1,
unmodified) using this Phase's own pre-registered `WFO_CONFIG`
(`v3/validation/wfo_config.py`). No window arithmetic is re-derived here.
"""

from __future__ import annotations

from datetime import date as date_type

from backtest.walk_forward import WalkForwardWindow, generate_windows
from v3.validation.wfo_config import DATA_END, DATA_START, WFO_CONFIG


def get_v3_3_windows(
    data_start: date_type | None = None, data_end: date_type | None = None
) -> list[WalkForwardWindow]:
    """data_start/data_end default to this Phase's pre-registered
    DATA_START/DATA_END (`v3/validation/wfo_config.py`) - the override
    parameters exist ONLY so tests can exercise the real window-generation
    logic against a smaller synthetic date range; the real Full Universe
    run never passes them.
    """
    start = data_start if data_start is not None else date_type.fromisoformat(DATA_START)
    end = data_end if data_end is not None else date_type.fromisoformat(DATA_END)
    return generate_windows(start, end, WFO_CONFIG)
