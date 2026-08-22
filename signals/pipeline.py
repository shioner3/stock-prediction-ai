"""Combines every Signal in signals/registry.py into one panel for a
single ticker's Feature panel (ticker + date + triggered/not-triggered
per Signal), and melts that into the long-format SignalRecord shape used
for storage:

    ticker, date, signal_name, direction, triggered, signal_version

Only triggered=True rows are kept when saving (see
pipeline/build_signals.py) - this is explicitly permitted by the Phase 4
spec ("Signalが発生した行だけを保存する方式でもよい") and avoids storing
a mostly-False boolean column per Signal per ticker per day. Nothing
about reproducibility depends on the saved file: the same Feature panel +
the same config always regenerates the exact same signals, so the saved
file is a convenience cache, not the source of truth.
"""

from __future__ import annotations

import pandas as pd

from config.loader import SignalsConfig
from signals.registry import SIGNAL_REGISTRY, all_signal_meta

RECORD_COLUMNS = ["ticker", "date", "signal_name", "direction", "triggered", "signal_version"]


def compute_signal_panel(panel: pd.DataFrame, config: SignalsConfig) -> pd.DataFrame:
    """Wide format: ticker, date, and one boolean column per signal_name."""
    out = pd.DataFrame(index=panel.index)
    out["ticker"] = panel["ticker"]
    out["date"] = panel["date"]

    for entry in SIGNAL_REGISTRY:
        sub_config = entry.get_config(config)
        meta = entry.meta_fn(sub_config)
        out[meta.name] = entry.compute_fn(panel, sub_config)

    return out


def compute_signal_records(panel: pd.DataFrame, config: SignalsConfig) -> pd.DataFrame:
    """Long format (RECORD_COLUMNS), triggered=True rows only."""
    wide = compute_signal_panel(panel, config)
    signal_names = [c for c in wide.columns if c not in ("ticker", "date")]

    records = wide.melt(
        id_vars=["ticker", "date"],
        value_vars=signal_names,
        var_name="signal_name",
        value_name="triggered",
    )
    records = records[records["triggered"]].copy()

    meta_by_name = {m.name: m for m in all_signal_meta(config)}
    records["direction"] = records["signal_name"].map(lambda n: meta_by_name[n].direction.value)
    records["signal_version"] = records["signal_name"].map(
        lambda n: meta_by_name[n].signal_version
    )

    records = records.sort_values(["date", "signal_name"]).reset_index(drop=True)
    return records[RECORD_COLUMNS]
