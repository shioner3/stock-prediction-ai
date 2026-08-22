"""Phase 11 section 8: Daily Performance Log.

Append-only (JSON Lines), one row per Forward Test run date. A date
already present is never overwritten - re-running the same day is a
silent no-op here (append_entry returns False), matching the Signal
Log's and Paper Portfolio's idempotent-by-key design (spec section 5/28).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class PerformanceLogEntry:
    date: str
    strategy_version: str
    strategy_hash: str
    universe_size: int
    data_timestamp: str
    market_regime_summary: str
    signal_count: int
    candidate_count: int
    open_positions: int
    closed_positions: int
    realized_pnl: float
    unrealized_pnl: float
    equity: float
    daily_return: float | None
    cumulative_return: float
    max_drawdown: float
    data_quality_status: str


def load_entries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_entry(path: Path, entry: PerformanceLogEntry) -> bool:
    """Returns True if a new row was written, False if `entry.date`
    already had a row (idempotent no-op, not an error).
    """
    existing_dates = {e["date"] for e in load_entries(path)}
    if entry.date in existing_dates:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    return True


def compute_cumulative_return(latest_equity: float, initial_capital: float) -> float:
    if initial_capital == 0:
        return 0.0
    return latest_equity / initial_capital - 1


def compute_daily_return(latest_equity: float, previous_equity: float | None) -> float | None:
    if previous_equity is None or previous_equity == 0:
        return None
    return latest_equity / previous_equity - 1


def compute_max_drawdown(equity_series: list[float]) -> float:
    """Max peak-to-trough decline (<=0) across the full equity history so
    far, INCLUDING today's equity as the last point.
    """
    if not equity_series:
        return 0.0
    peak = equity_series[0]
    max_dd = 0.0
    for e in equity_series:
        peak = max(peak, e)
        if peak != 0:
            max_dd = min(max_dd, (e - peak) / peak)
    return max_dd
