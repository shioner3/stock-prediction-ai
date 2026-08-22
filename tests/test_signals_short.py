from __future__ import annotations

import pytest
from conftest import make_panel

from config.loader import (
    ShortBreakdownSignalConfig,
    ShortMaRejectionSignalConfig,
    ShortMomentumContinuationSignalConfig,
    ShortOverboughtReversalSignalConfig,
    ShortPullbackSignalConfig,
    ShortVolumeBreakdownSignalConfig,
)
from signals.short import (
    breakdown,
    ma_rejection,
    momentum_continuation,
    overbought_reversal,
    pullback,
    volume_breakdown,
)

EPS = 1e-6


# --- SHORT Breakdown -----------------------------------------------------------


@pytest.mark.parametrize(
    ("close", "expected"),
    [(100.0 + EPS, False), (100.0, False), (100.0 - EPS, True)],
)
def test_breakdown_close_vs_low20d_boundary(close: float, expected: bool) -> None:
    config = ShortBreakdownSignalConfig(lookback=20, volume_multiple=1.5)
    panel = make_panel(close=[close], low_20d=[100.0], volume_ratio_20d=[3.0])
    result = breakdown.compute_signal(panel, config)
    assert result.iloc[0] == expected


@pytest.mark.parametrize(
    ("volume_ratio", "expected"),
    [(1.5 - EPS, False), (1.5, False), (1.5 + EPS, True)],
)
def test_breakdown_volume_multiple_boundary(volume_ratio: float, expected: bool) -> None:
    config = ShortBreakdownSignalConfig(lookback=20, volume_multiple=1.5)
    panel = make_panel(close=[90.0], low_20d=[100.0], volume_ratio_20d=[volume_ratio])
    result = breakdown.compute_signal(panel, config)
    assert result.iloc[0] == expected


def test_breakdown_invalid_lookback_raises() -> None:
    config = ShortBreakdownSignalConfig(lookback=17)
    panel = make_panel(close=[100.0])
    with pytest.raises(ValueError, match="lookback"):
        breakdown.compute_signal(panel, config)


# --- SHORT Pullback ("戻り売り") ------------------------------------------------


@pytest.mark.parametrize(
    ("depth", "expected"),
    [(0.03 - EPS, False), (0.03, True), (0.03 + EPS, True), (0.15, True), (0.15 + EPS, False)],
)
def test_short_pullback_depth_boundaries(depth: float, expected: bool) -> None:
    config = ShortPullbackSignalConfig(sma_fast=20, sma_slow=50, min_depth=0.03, max_depth=0.15)
    panel = make_panel(close=[95.0], sma_20=[100.0], sma_50=[110.0], bounce_depth=[depth])
    result = pullback.compute_signal(panel, config)
    assert result.iloc[0] == expected


def test_short_pullback_requires_downtrend() -> None:
    config = ShortPullbackSignalConfig()
    panel = make_panel(close=[105.0], sma_20=[110.0], sma_50=[100.0], bounce_depth=[0.05])
    result = pullback.compute_signal(panel, config)
    assert not result.iloc[0]


def test_short_pullback_requires_close_below_sma_fast() -> None:
    config = ShortPullbackSignalConfig()
    panel = make_panel(close=[120.0], sma_20=[100.0], sma_50=[110.0], bounce_depth=[0.05])
    result = pullback.compute_signal(panel, config)
    assert not result.iloc[0]


# --- SHORT MA Rejection --------------------------------------------------------


@pytest.mark.parametrize(
    ("prev_close", "today_close", "expected"),
    [
        (102.0, 101.0, False),  # still above SMA both days - no rejection
        (102.0, 99.0, True),  # above yesterday, below today - rejected
        (99.0, 101.0, False),  # below yesterday - not a rejection day
        (100.0, 99.0, True),  # AT the MA yesterday (>=), below today
    ],
)
def test_ma_rejection_transition(prev_close: float, today_close: float, expected: bool) -> None:
    config = ShortMaRejectionSignalConfig(sma_fast=20, sma_slow=50)
    panel = make_panel(
        n=2,
        close=[prev_close, today_close],
        sma_20=[100.0, 100.0],
        sma_50=[110.0, 110.0],  # downtrend both days
    )
    result = ma_rejection.compute_signal(panel, config)
    assert result.iloc[1] == expected
    assert not result.iloc[0]


def test_ma_rejection_requires_downtrend() -> None:
    config = ShortMaRejectionSignalConfig()
    panel = make_panel(n=2, close=[102.0, 99.0], sma_20=[100.0, 100.0], sma_50=[90.0, 90.0])
    result = ma_rejection.compute_signal(panel, config)
    assert not result.iloc[1]


# --- SHORT Momentum Continuation -----------------------------------------------


@pytest.mark.parametrize(
    ("return_5d", "expected"), [(-0.03 + EPS, False), (-0.03, False), (-0.03 - EPS, True)]
)
def test_short_momentum_continuation_return_5d_boundary(
    return_5d: float, expected: bool
) -> None:
    config = ShortMomentumContinuationSignalConfig(
        return_5d_max=-0.03, return_20d_max=0.0, sma_period=20
    )
    panel = make_panel(close=[90.0], return_5d=[return_5d], return_20d=[-0.10], sma_20=[100.0])
    result = momentum_continuation.compute_signal(panel, config)
    assert result.iloc[0] == expected


def test_short_momentum_continuation_requires_close_below_sma() -> None:
    config = ShortMomentumContinuationSignalConfig()
    panel = make_panel(close=[110.0], return_5d=[-0.10], return_20d=[-0.10], sma_20=[100.0])
    result = momentum_continuation.compute_signal(panel, config)
    assert not result.iloc[0]


# --- SHORT Volume Breakdown -----------------------------------------------------


@pytest.mark.parametrize(
    ("return_1d", "expected"), [(-0.03 + EPS, False), (-0.03, False), (-0.03 - EPS, True)]
)
def test_volume_breakdown_return_1d_boundary(return_1d: float, expected: bool) -> None:
    config = ShortVolumeBreakdownSignalConfig(return_1d_max=-0.03, volume_ratio_min=2.0)
    panel = make_panel(return_1d=[return_1d], volume_ratio_20d=[5.0])
    result = volume_breakdown.compute_signal(panel, config)
    assert result.iloc[0] == expected


@pytest.mark.parametrize(
    ("volume_ratio", "expected"), [(2.0 - EPS, False), (2.0, False), (2.0 + EPS, True)]
)
def test_volume_breakdown_volume_ratio_boundary(volume_ratio: float, expected: bool) -> None:
    config = ShortVolumeBreakdownSignalConfig(return_1d_max=-0.03, volume_ratio_min=2.0)
    panel = make_panel(return_1d=[-0.10], volume_ratio_20d=[volume_ratio])
    result = volume_breakdown.compute_signal(panel, config)
    assert result.iloc[0] == expected


# --- SHORT Overbought Reversal --------------------------------------------------


@pytest.mark.parametrize(
    ("rsi", "expected"), [(70.0 + EPS, True), (70.0, False), (70.0 - EPS, False)]
)
def test_overbought_reversal_rsi_boundary(rsi: float, expected: bool) -> None:
    config = ShortOverboughtReversalSignalConfig(rsi_period=14, rsi_min=70.0)
    panel = make_panel(n=2, close=[100.0, 99.0], rsi_14=[rsi, rsi])
    result = overbought_reversal.compute_signal(panel, config)
    assert result.iloc[1] == expected


def test_overbought_reversal_requires_down_day() -> None:
    config = ShortOverboughtReversalSignalConfig(rsi_min=70.0)
    panel = make_panel(n=2, close=[100.0, 101.0], rsi_14=[80.0, 80.0])
    result = overbought_reversal.compute_signal(panel, config)
    assert not result.iloc[1]


def test_overbought_reversal_invalid_rsi_period_raises() -> None:
    config = ShortOverboughtReversalSignalConfig(rsi_period=9)
    panel = make_panel(close=[100.0])
    with pytest.raises(ValueError, match="rsi_period"):
        overbought_reversal.compute_signal(panel, config)
