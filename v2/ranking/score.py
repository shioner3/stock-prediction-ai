"""V2 Initial Research Score (spec section 8/9).

Every category is a plain (equal-weighted) AVERAGE of its member
features' own cross-sectional percentile ranks (v2/ranking/
cross_sectional.py), and the total V2 Score is a FIXED-weight sum of
the six category ranks (v2/config/v2_settings.yaml::score_weights).
Neither the feature-to-category assignment nor the weights below were
tuned against any backtest/Phase 6.5-14 result - see
research/phase_v2_1_report.md for the reasoning behind each choice.

Directionality note (spec section 6/9): each (column, higher_is_better)
pair is V2's OWN judgment for what "more attractive as a swing
candidate" means for that raw Feature - it does NOT reuse features/
metadata.py::FeatureMeta.directionality wholesale, because V1's tags
encode a MOMENTUM reading everywhere (e.g. pullback_depth is tagged
"bullish_low" there - "shallower pullback is more bullish"), while V2's
own Pullback/Oversold category is deliberately a MEAN-REVERSION signal
(a DEEPER pullback / lower RSI is treated as MORE attractive for a
rebound candidate) - the same interpretive choice V1's own
long_oversold_rebound Signal already makes via its `rsi_14 < 30`
trigger, just expressed here as a continuous rank instead of a discrete
threshold. Momentum/Trend/Volume/Relative Strength keep the same
directional reading V1's metadata already uses.

Volatility is graded LOWER-is-better here (spec section 0's own framing:
"期待値・勝率・リスクのバランスが比較的良い銘柄" - risk-balance is part
of the ranking goal) - V1's own FeatureMeta tags volatility features
"neutral" (no opinion), so this is a genuine NEW judgment call V2 makes
for its own purpose, not a reuse of any V1 tag.

A row's category rank is NaN if EVERY member feature is NaN that day
(most commonly: still in Feature warmup) - average_category_rank()
already skips NaN members, but an all-NaN category stays NaN by
pandas' default skipna mean. compute_v2_score() does NOT impute or
renormalize a NaN category away: a missing category makes the row's
total V2 Score NaN, and such a row is excluded from Candidate Ranking
(v2/candidate.py) rather than silently scored on fewer categories than
every other row - this keeps the Score's meaning identical across every
ticker/day it IS computed for.
"""

from __future__ import annotations

import pandas as pd

from v2.config.loader import V2ScoreWeightsConfig
from v2.ranking.cross_sectional import average_category_rank, percentile_rank_by_day

# category -> [(feature_column, higher_is_better), ...]
CATEGORY_FEATURES: dict[str, list[tuple[str, bool]]] = {
    "momentum": [
        ("return_1d", True),
        ("return_3d", True),
        ("return_5d", True),
        ("return_10d", True),
        ("return_20d", True),
        ("return_60d", True),
        ("close_to_sma_5", True),
        ("close_to_sma_20", True),
        ("price_vs_ma60", True),
        ("ma5_vs_ma20", True),
        ("ma20_vs_ma60", True),
    ],
    "trend": [
        ("sma_20_slope", True),
        ("sma_50_slope", True),
        ("distance_from_recent_high", True),
        ("distance_from_60d_high", True),
    ],
    "volume": [
        ("volume_ratio_5d", True),
        ("volume_ratio_20d", True),
        ("volume_trend", True),
    ],
    "volatility": [
        ("volatility_5d", False),
        ("volatility_10d", False),
        ("volatility_20d", False),
        ("atr_pct", False),
    ],
    "relative_strength": [
        ("rs_5d", True),
        ("rs_20d", True),
        ("rs_60d", True),
    ],
    "pullback": [
        ("pullback_depth", True),
        ("consecutive_down_days", True),
        ("rsi_14", False),
    ],
}

CATEGORY_RANK_COLUMNS = {category: f"{category}_rank" for category in CATEGORY_FEATURES}


def compute_category_ranks(panel: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """panel: date/ticker + every raw feature column CATEGORY_FEATURES
    references (v2/features_adapter.py's output, one day's Universe
    snapshot stacked across tickers - see v2/pipeline.py). Returns
    date/ticker + one f"{category}_rank" column per category, each in
    (0, 1].
    """
    out = panel[[date_col, "ticker"]].copy()
    for category, members in CATEGORY_FEATURES.items():
        member_ranks = pd.DataFrame(index=panel.index)
        for column, higher_is_better in members:
            member_ranks[column] = percentile_rank_by_day(
                panel, column, date_col=date_col, higher_is_better=higher_is_better
            )
        out[CATEGORY_RANK_COLUMNS[category]] = average_category_rank(member_ranks)
    return out


def compute_v2_score(category_ranks: pd.DataFrame, weights: V2ScoreWeightsConfig) -> pd.Series:
    """Weighted sum of the six category rank columns. NaN propagates: a
    row missing any one category's rank gets a NaN total_score (see
    module docstring - never imputed or renormalized).
    """
    weight_by_category = {
        "momentum": weights.momentum,
        "trend": weights.trend,
        "volume": weights.volume,
        "volatility": weights.volatility,
        "relative_strength": weights.relative_strength,
        "pullback": weights.pullback,
    }
    total = pd.Series(0.0, index=category_ranks.index)
    for category, weight in weight_by_category.items():
        total = total + category_ranks[CATEGORY_RANK_COLUMNS[category]] * weight
    return total
