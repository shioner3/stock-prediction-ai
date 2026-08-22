"""Central registry of all 12 Signals - lets the pipeline and tests
iterate over every Signal generically instead of hardcoding 12 call
sites. Adding a 13th Signal later means adding one entry here and
nothing else changes in pipeline/backtest code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from config.loader import SignalsConfig
from signals.base import SignalMeta
from signals.long import breakout as long_breakout
from signals.long import ma_rebound as long_ma_rebound
from signals.long import momentum_continuation as long_momentum_continuation
from signals.long import oversold_rebound as long_oversold_rebound
from signals.long import pullback as long_pullback
from signals.long import volume_breakout as long_volume_breakout
from signals.short import breakdown as short_breakdown
from signals.short import ma_rejection as short_ma_rejection
from signals.short import momentum_continuation as short_momentum_continuation
from signals.short import overbought_reversal as short_overbought_reversal
from signals.short import pullback as short_pullback
from signals.short import volume_breakdown as short_volume_breakdown


@dataclass(frozen=True)
class SignalRegistryEntry:
    meta_fn: Callable[[Any], SignalMeta]
    compute_fn: Callable[[pd.DataFrame, Any], pd.Series]
    get_config: Callable[[SignalsConfig], Any]


SIGNAL_REGISTRY: list[SignalRegistryEntry] = [
    SignalRegistryEntry(
        long_breakout.meta, long_breakout.compute_signal, lambda c: c.long.breakout
    ),
    SignalRegistryEntry(
        long_pullback.meta, long_pullback.compute_signal, lambda c: c.long.pullback
    ),
    SignalRegistryEntry(
        long_ma_rebound.meta, long_ma_rebound.compute_signal, lambda c: c.long.ma_rebound
    ),
    SignalRegistryEntry(
        long_momentum_continuation.meta,
        long_momentum_continuation.compute_signal,
        lambda c: c.long.momentum_continuation,
    ),
    SignalRegistryEntry(
        long_volume_breakout.meta,
        long_volume_breakout.compute_signal,
        lambda c: c.long.volume_breakout,
    ),
    SignalRegistryEntry(
        long_oversold_rebound.meta,
        long_oversold_rebound.compute_signal,
        lambda c: c.long.oversold_rebound,
    ),
    SignalRegistryEntry(
        short_breakdown.meta, short_breakdown.compute_signal, lambda c: c.short.breakdown
    ),
    SignalRegistryEntry(
        short_pullback.meta, short_pullback.compute_signal, lambda c: c.short.pullback
    ),
    SignalRegistryEntry(
        short_ma_rejection.meta, short_ma_rejection.compute_signal, lambda c: c.short.ma_rejection
    ),
    SignalRegistryEntry(
        short_momentum_continuation.meta,
        short_momentum_continuation.compute_signal,
        lambda c: c.short.momentum_continuation,
    ),
    SignalRegistryEntry(
        short_volume_breakdown.meta,
        short_volume_breakdown.compute_signal,
        lambda c: c.short.volume_breakdown,
    ),
    SignalRegistryEntry(
        short_overbought_reversal.meta,
        short_overbought_reversal.compute_signal,
        lambda c: c.short.overbought_reversal,
    ),
]


def all_signal_meta(config: SignalsConfig) -> list[SignalMeta]:
    return [entry.meta_fn(entry.get_config(config)) for entry in SIGNAL_REGISTRY]
