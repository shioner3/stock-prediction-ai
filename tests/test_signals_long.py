from __future__ import annotations

import pytest
from conftest import make_panel

from config.loader import (
    LongBreakoutSignalConfig,
    LongMaReboundSignalConfig,
    LongMomentumContinuationSignalConfig,
    LongOversoldReboundSignalConfig,
    LongPullbackSignalConfig,
    LongVolumeBreakoutSignalConfig,
)
from signals.long import (
    breakout,
    ma_rebound,
    momentum_continuation,
    oversold_rebound,
    pullback,
    volume_breakout,
)

EPS = 1e-6


# --- LONG Breakout -----------------------------------------------------------


def test_breakout_requires_missing_columns_to_raise() -> None:
    config = LongBreakoutSignalConfig()
    panel = make_panel(close=[100.0])
    with pytest.raises(ValueError, match="missing"):
        breakout.compute_signal(panel, config)


@pytest.mark.parametrize(
    ("close", "expected"),
    [(100.0 - EPS, False), (100.0, False), (100.0 + EPS, True)],
)
def test_breakout_close_vs_high20d_boundary(close: float, expected: bool) -> None:
    config = LongBreakoutSignalConfig(lookback=20, volume_multiple=1.5)
    panel = make_panel(close=[close], high_20d=[100.0], volume_ratio_20d=[3.0])
    result = breakout.compute_signal(panel, config)
    assert result.iloc[0] == expected


@pytest.mark.parametrize(
    ("volume_ratio", "expected"),
    [(1.5 - EPS, False), (1.5, False), (1.5 + EPS, True)],
)
def test_breakout_volume_multiple_boundary(volume_ratio: float, expected: bool) -> None:
    config = LongBreakoutSignalConfig(lookback=20, volume_multiple=1.5)
    panel = make_panel(close=[110.0], high_20d=[100.0], volume_ratio_20d=[volume_ratio])
    result = breakout.compute_signal(panel, config)
    assert result.iloc[0] == expected


def test_breakout_invalid_lookback_raises() -> None:
    config = LongBreakoutSignalConfig(lookback=17)
    panel = make_panel(close=[100.0])
    with pytest.raises(ValueError, match="lookback"):
        breakout.compute_signal(panel, config)


# --- LONG Pullback -------------------------------------------------------------


@pytest.mark.parametrize(
    ("depth", "expected"),
    [(0.03 - EPS, False), (0.03, True), (0.03 + EPS, True), (0.15, True), (0.15 + EPS, False)],
)
def test_pullback_depth_boundaries(depth: float, expected: bool) -> None:
    config = LongPullbackSignalConfig(sma_fast=20, sma_slow=50, min_depth=0.03, max_depth=0.15)
    panel = make_panel(close=[105.0], sma_20=[100.0], sma_50=[90.0], pullback_depth=[depth])
    result = pullback.compute_signal(panel, config)
    assert result.iloc[0] == expected


def test_pullback_requires_uptrend() -> None:
    config = LongPullbackSignalConfig()
    panel = make_panel(close=[95.0], sma_20=[90.0], sma_50=[100.0], pullback_depth=[0.05])
    result = pullback.compute_signal(panel, config)
    assert result.iloc[0] is False or not result.iloc[0]


def test_pullback_requires_close_above_sma_fast() -> None:
    config = LongPullbackSignalConfig()
    panel = make_panel(close=[80.0], sma_20=[100.0], sma_50=[90.0], pullback_depth=[0.05])
    result = pullback.compute_signal(panel, config)
    assert not result.iloc[0]


# --- LONG MA Rebound -----------------------------------------------------------


@pytest.mark.parametrize(
    ("prev_close", "today_close", "expected"),
    [
        (98.0, 99.0, False),  # still below SMA both days - no rebound
        (98.0, 101.0, True),  # below yesterday, above today - rebound
        (101.0, 99.0, False),  # above yesterday - not a rebound day
        (100.0, 101.0, True),  # AT the MA yesterday (<=), above today
    ],
)
def test_ma_rebound_transition(prev_close: float, today_close: float, expected: bool) -> None:
    config = LongMaReboundSignalConfig(sma_fast=20, sma_slow=50)
    panel = make_panel(
        n=2,
        close=[prev_close, today_close],
        sma_20=[100.0, 100.0],
        sma_50=[90.0, 90.0],  # uptrend both days
    )
    result = ma_rebound.compute_signal(panel, config)
    assert result.iloc[1] == expected
    assert not result.iloc[0]  # row 0 has no "yesterday" -> shift(1) is NaN -> never triggers


def test_ma_rebound_requires_uptrend() -> None:
    config = LongMaReboundSignalConfig()
    panel = make_panel(n=2, close=[98.0, 101.0], sma_20=[100.0, 100.0], sma_50=[110.0, 110.0])
    result = ma_rebound.compute_signal(panel, config)
    assert not result.iloc[1]


# --- LONG Momentum Continuation --------------------------------------------


@pytest.mark.parametrize(
    ("return_5d", "expected"), [(0.03 - EPS, False), (0.03, False), (0.03 + EPS, True)]
)
def test_momentum_continuation_return_5d_boundary(return_5d: float, expected: bool) -> None:
    config = LongMomentumContinuationSignalConfig(
        return_5d_min=0.03, return_20d_min=0.0, sma_period=20
    )
    panel = make_panel(close=[110.0], return_5d=[return_5d], return_20d=[0.10], sma_20=[100.0])
    result = momentum_continuation.compute_signal(panel, config)
    assert result.iloc[0] == expected


def test_momentum_continuation_requires_close_above_sma() -> None:
    config = LongMomentumContinuationSignalConfig()
    panel = make_panel(close=[90.0], return_5d=[0.10], return_20d=[0.10], sma_20=[100.0])
    result = momentum_continuation.compute_signal(panel, config)
    assert not result.iloc[0]


# --- LONG Volume Breakout ----------------------------------------------------


@pytest.mark.parametrize(
    ("return_1d", "expected"), [(0.03 - EPS, False), (0.03, False), (0.03 + EPS, True)]
)
def test_volume_breakout_return_1d_boundary(return_1d: float, expected: bool) -> None:
    config = LongVolumeBreakoutSignalConfig(return_1d_min=0.03, volume_ratio_min=2.0)
    panel = make_panel(return_1d=[return_1d], volume_ratio_20d=[5.0])
    result = volume_breakout.compute_signal(panel, config)
    assert result.iloc[0] == expected


@pytest.mark.parametrize(
    ("volume_ratio", "expected"), [(2.0 - EPS, False), (2.0, False), (2.0 + EPS, True)]
)
def test_volume_breakout_volume_ratio_boundary(volume_ratio: float, expected: bool) -> None:
    config = LongVolumeBreakoutSignalConfig(return_1d_min=0.03, volume_ratio_min=2.0)
    panel = make_panel(return_1d=[0.10], volume_ratio_20d=[volume_ratio])
    result = volume_breakout.compute_signal(panel, config)
    assert result.iloc[0] == expected


def test_volume_breakout_is_not_identical_condition_to_breakout() -> None:
    """A day can trigger long_volume_breakout without being anywhere near
    a 20-day high - this is what keeps the two Signals independent
    hypotheses (see signals/long/volume_breakout.py's docstring).
    """
    config = LongVolumeBreakoutSignalConfig(return_1d_min=0.03, volume_ratio_min=2.0)
    panel = make_panel(return_1d=[0.10], volume_ratio_20d=[5.0])
    result = volume_breakout.compute_signal(panel, config)
    assert result.iloc[0]  # triggers with no reference to any high_Nd column at all


# --- LONG Oversold Rebound -----------------------------------------------------


@pytest.mark.parametrize(
    ("rsi", "expected"), [(30.0 - EPS, True), (30.0, False), (30.0 + EPS, False)]
)
def test_oversold_rebound_rsi_boundary(rsi: float, expected: bool) -> None:
    config = LongOversoldReboundSignalConfig(rsi_period=14, rsi_max=30.0)
    panel = make_panel(n=2, close=[100.0, 101.0], rsi_14=[rsi, rsi])
    result = oversold_rebound.compute_signal(panel, config)
    assert result.iloc[1] == expected


def test_oversold_rebound_requires_up_day() -> None:
    config = LongOversoldReboundSignalConfig(rsi_max=30.0)
    panel = make_panel(n=2, close=[100.0, 99.0], rsi_14=[20.0, 20.0])
    result = oversold_rebound.compute_signal(panel, config)
    assert not result.iloc[1]


def test_oversold_rebound_invalid_rsi_period_raises() -> None:
    config = LongOversoldReboundSignalConfig(rsi_period=9)
    panel = make_panel(close=[100.0])
    with pytest.raises(ValueError, match="rsi_period"):
        oversold_rebound.compute_signal(panel, config)
