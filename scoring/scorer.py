"""Score: "how good is this setup", on a fixed 0-100 scale, split into 6
categories - Trend/Momentum/Volume/Relative/Setup/Risk (20/20/15/15/20/10
points). Every category function is direction-aware (LONG vs SHORT get
DIFFERENT conditions, not a blind sign-flip of the same formula - see
each function's docstring for why).

Score is computed on the Feature panel, exactly the same way a Signal is
(features/pipeline.py's output; no vendor/provider access, no targets/
import - see tests/test_target_leakage.py). It reads ONLY already-computed,
backward-looking Feature columns; no new rolling/shift logic is
introduced here, so Score(t) inherits the no-lookahead guarantee already
proven for every Feature it reads (see tests/test_scoring_no_lookahead.py
for the direct verification anyway).

Every point allocation here is a fixed initial hypothesis - Phase 5
forbids adjusting any of these thresholds based on validation results
(see scoring/validation.py and README's "Score Mapping").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.loader import (
    ScoreMomentumConfig,
    ScoreRelativeConfig,
    ScoreRiskConfig,
    ScoreSetupConfig,
    ScoreTrendConfig,
    ScoreVolumeConfig,
    ScoringConfig,
)
from features.indicators import RSI_PERIODS
from features.momentum import RETURN_WINDOWS
from signals.base import Direction

SUBSCORE_COLUMNS = [
    "trend_score", "momentum_score", "volume_score",
    "relative_score", "setup_score", "risk_score",
]


@dataclass(frozen=True)
class ScoreCategoryMeta:
    name: str
    max_points: int
    description: str


def category_metadata(config: ScoringConfig) -> list[ScoreCategoryMeta]:
    w = config.weights
    return [
        ScoreCategoryMeta("trend_score", w.trend, "4 conditions x 5pts: MA order, Close vs "
                           "fast MA, and both MAs' slope, all direction-aware"),
        ScoreCategoryMeta("momentum_score", w.momentum, "4 conditions x 5pts: two return "
                           "windows, RSI vs midline, MACD histogram sign, direction-aware"),
        ScoreCategoryMeta("volume_score", w.volume, "3 conditions x 5pts: above-average "
                           "volume, price-move+volume confirmation, rising volume trend"),
        ScoreCategoryMeta("relative_score", w.relative, "rs_5d/20d/60d each mapped to 0-5pts "
                           "via config.scoring.relative.thresholds, direction-aware"),
        ScoreCategoryMeta("setup_score", w.setup, "4 conditions x 5pts, stricter than any "
                           "Signal's own trigger threshold: strong 5d return, strong volume, "
                           "3-MA stack alignment, ATR-normalized day move"),
        ScoreCategoryMeta("risk_score", w.risk, "2 conditions x 5pts, direction-agnostic: "
                           "ATR% and 20d volatility both below a ceiling"),
    ]


def _rsi_column(period: int) -> str:
    if period not in RSI_PERIODS:
        raise ValueError(f"rsi_period={period} has no matching feature column - must be one "
                          f"of {RSI_PERIODS} (features/indicators.py)")
    return f"rsi_{period}"


def _return_column(period: int) -> str:
    if period not in RETURN_WINDOWS:
        raise ValueError(f"return period={period} has no matching feature column - must be "
                          f"one of {RETURN_WINDOWS} (features/momentum.py)")
    return f"return_{period}d"


def _points(condition: pd.Series, points: float) -> pd.Series:
    """condition -> points where True, 0 where False or NaN (never NaN
    itself - a NaN Feature at warmup means "condition not established",
    not "unknown bonus").
    """
    return condition.fillna(False).astype(float) * points


# --- Trend Score (direction-aware) ------------------------------------------


def compute_trend_score(
    panel: pd.DataFrame, direction: Direction, config: ScoreTrendConfig
) -> pd.Series:
    """LONG rewards an uptrend (fast MA above slow MA, price above the
    fast MA, both MAs sloping up); SHORT rewards the mirror-image
    downtrend. This is NOT "compute the LONG score and negate it" - each
    condition is its own direction-aware comparison, so a flat/sideways
    market scores low on BOTH sides rather than SHORT automatically
    getting whatever points LONG didn't.
    """
    sma_fast = panel[f"sma_{config.sma_fast}"]
    sma_slow = panel[f"sma_{config.sma_slow}"]
    close = panel["close"]
    slope_fast = panel[f"sma_{config.sma_fast}_slope"]
    slope_slow = panel[f"sma_{config.sma_slow}_slope"]
    eps = config.slope_epsilon

    if direction == Direction.LONG:
        c1 = sma_fast > sma_slow
        c2 = close > sma_fast
        c3 = slope_fast > eps
        c4 = slope_slow > eps
    else:
        c1 = sma_fast < sma_slow
        c2 = close < sma_fast
        c3 = slope_fast < -eps
        c4 = slope_slow < -eps

    return _points(c1, 5) + _points(c2, 5) + _points(c3, 5) + _points(c4, 5)


# --- Momentum Score (direction-aware) ---------------------------------------


def compute_momentum_score(
    panel: pd.DataFrame, direction: Direction, config: ScoreMomentumConfig
) -> pd.Series:
    """LONG rewards positive short/medium-term returns, RSI above its
    midline, and a positive MACD histogram; SHORT rewards the mirror.
    """
    return_5d = panel[_return_column(config.return_5d_period)]
    return_10d = panel[_return_column(config.return_10d_period)]
    rsi = panel[_rsi_column(config.rsi_period)]
    macd_hist = panel["macd_hist"]

    if direction == Direction.LONG:
        c1 = return_5d > 0
        c2 = return_10d > 0
        c3 = rsi > config.rsi_midline
        c4 = macd_hist > 0
    else:
        c1 = return_5d < 0
        c2 = return_10d < 0
        c3 = rsi < config.rsi_midline
        c4 = macd_hist < 0

    return _points(c1, 5) + _points(c2, 5) + _points(c3, 5) + _points(c4, 5)


# --- Volume Score (direction-aware) -----------------------------------------


def compute_volume_score(
    panel: pd.DataFrame, direction: Direction, config: ScoreVolumeConfig
) -> pd.Series:
    """Both directions reward above-average volume and a rising volume
    trend (volume confirms conviction regardless of direction); only the
    "price move + volume" confirmation condition is direction-specific
    (an up move needs volume to confirm LONG; a down move needs volume to
    confirm SHORT).
    """
    volume_ratio_20d = panel["volume_ratio_20d"]
    volume_ratio_5d = panel["volume_ratio_5d"]
    return_1d = panel["return_1d"]
    volume_trend = panel["volume_trend"]

    c1 = volume_ratio_20d > config.ratio_confirm_threshold
    if direction == Direction.LONG:
        c2 = (return_1d > 0) & (volume_ratio_5d > 1.0)
    else:
        c2 = (return_1d < 0) & (volume_ratio_5d > 1.0)
    c3 = volume_trend > 0

    return _points(c1, 5) + _points(c2, 5) + _points(c3, 5)


# --- Relative Score (direction-aware mapping, not a sign flip) -------------


def _map_rs_to_points(value: pd.Series, thresholds: list[float]) -> pd.Series:
    """thresholds must be strictly descending, length 5 -> 5/4/3/2/1
    points for clearing the corresponding threshold, 0 below all of them.
    NaN (RS unavailable, e.g. no market benchmark) -> 0 points, never a
    penalty beyond that and never propagated as NaN into total_score.
    """
    conditions = [value >= th for th in thresholds]
    choices = list(range(len(thresholds), 0, -1))
    points = np.select(conditions, choices, default=0)
    return pd.Series(points, index=value.index, dtype=float).where(~value.isna(), 0.0)


def compute_relative_score(
    panel: pd.DataFrame, direction: Direction, config: ScoreRelativeConfig
) -> pd.Series:
    """LONG: higher rs_Nd (outperforming the market) scores higher.
    SHORT: lower/more negative rs_Nd (underperforming the market) scores
    higher - achieved by mapping -rs_Nd through the SAME threshold
    ladder as LONG, not by inventing a second set of thresholds (but
    also not by just negating the LONG score, since the mapping is a
    step function, not a linear one - see _map_rs_to_points).
    Each of rs_5d/rs_20d/rs_60d contributes 0-5 points -> 0-15 total.
    """
    total = pd.Series(0.0, index=panel.index)
    for window in ("rs_5d", "rs_20d", "rs_60d"):
        rs = panel[window]
        value = rs if direction == Direction.LONG else -rs
        total = total + _map_rs_to_points(value, config.thresholds)
    return total


# --- Setup Score (direction-aware, deliberately NOT the trigger condition) -


def compute_setup_score(
    panel: pd.DataFrame, direction: Direction, config: ScoreSetupConfig
) -> pd.Series:
    """Measures "how far past the bar", using thresholds strictly
    stricter than any Signal's own trigger threshold, plus two
    dimensions no Signal's trigger condition uses at all (3-MA stack
    alignment breadth, and the day's move size normalized by ATR% -
    "conviction" rather than raw direction).
    """
    return_5d = panel["return_5d"]
    volume_ratio_20d = panel["volume_ratio_20d"]
    ma_fast = panel[f"sma_{config.ma_stack_fast}"]
    ma_mid = panel[f"sma_{config.ma_stack_mid}"]
    ma_slow = panel[f"sma_{config.ma_stack_slow}"]
    return_1d = panel["return_1d"]
    atr_pct = panel["atr_pct"]

    if direction == Direction.LONG:
        c1 = return_5d > config.return_5d_strong
        c3 = (ma_fast > ma_mid) & (ma_mid > ma_slow)
    else:
        c1 = return_5d < -config.return_5d_strong
        c3 = (ma_fast < ma_mid) & (ma_mid < ma_slow)

    c2 = volume_ratio_20d > config.volume_ratio_strong
    move_ratio = (return_1d.abs()) / atr_pct.replace(0, np.nan)
    c4 = move_ratio > config.atr_move_ratio_threshold

    return _points(c1, 5) + _points(c2, 5) + _points(c3, 5) + _points(c4, 5)


# --- Risk Score (direction-agnostic; higher = LOWER risk) ------------------


def compute_risk_score(panel: pd.DataFrame, config: ScoreRiskConfig) -> pd.Series:
    """Direction-agnostic: instability is a bad sign whether the trade is
    LONG or SHORT. Higher risk_score always means LOWER risk (a stock
    trading calmly), by convention stated once here.
    """
    atr_pct = panel["atr_pct"]
    volatility_20d = panel["volatility_20d"]

    c1 = atr_pct < config.atr_pct_max
    c2 = volatility_20d < config.volatility_max

    return _points(c1, 5) + _points(c2, 5)


# --- Total ---------------------------------------------------------------


def compute_total_score(
    panel: pd.DataFrame, direction: Direction, config: ScoringConfig
) -> pd.DataFrame:
    """Returns a DataFrame indexed like panel, columns SUBSCORE_COLUMNS +
    "total_score". total_score is always the literal sum of the six
    subscores (never independently computed) and is guaranteed in
    [0, 100] as long as each subscore stays within its own declared
    range - see tests/test_scoring_scorer.py's range tests.
    """
    out = pd.DataFrame(index=panel.index)
    out["trend_score"] = compute_trend_score(panel, direction, config.trend)
    out["momentum_score"] = compute_momentum_score(panel, direction, config.momentum)
    out["volume_score"] = compute_volume_score(panel, direction, config.volume)
    out["relative_score"] = compute_relative_score(panel, direction, config.relative)
    out["setup_score"] = compute_setup_score(panel, direction, config.setup)
    out["risk_score"] = compute_risk_score(panel, config.risk)
    out["total_score"] = out[SUBSCORE_COLUMNS].sum(axis=1)
    return out
