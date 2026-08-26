"""V3 Feature Registry (spec section 6): the single source of truth for
every Feature `v3/dataset.py` computes - name / category / formula /
required_history / availability / leakage_risk. Deliberately SCOPED to
exactly the categories spec section 5 lists (52 core + 3 conditional
entries), not every column V1's `compute_feature_panel()` happens to
produce (e.g. MACD, EMA, high/low_Nd, bounce_depth are real V1 columns
but were never requested by spec section 5, so they are not registered
or exposed by V3's dataset builder - "Feature数を無制限に増やさない",
spec section 6's own instruction).

`required_history`: trading days of PRIOR history needed before the
value is non-NaN (same convention as `features.metadata.FeatureMeta.
warmup_period`). `leakage_risk`: every entry here is "none (backward-only)"
because every underlying formula (v1_reuse: V1's tested feature modules;
v3_new: v3/features/*.py, built only from V1's own backward-looking
utilities) only ever reads rows <= t - this is not a rubber stamp, it is
mechanically re-verified by `v3/leakage/availability_check.py`'s AST scan
(no `shift(-n)`/`.iloc[i+n]`-style forward read anywhere in v3/features/)
and by `v3/leakage/shock_tests.py`'s empirical future-shock tests.
"""

from __future__ import annotations

from dataclasses import dataclass

CATEGORY_MOMENTUM = "momentum"
CATEGORY_MOVING_AVERAGE = "moving_average"
CATEGORY_VOLATILITY = "volatility"
CATEGORY_OSCILLATOR = "oscillator"
CATEGORY_VOLUME = "volume"
CATEGORY_BREAKOUT_DRAWDOWN = "breakout_drawdown"
CATEGORY_CROSS_SECTIONAL = "cross_sectional"
CATEGORY_MARKET_CONTEXT = "market_context"
CATEGORY_SECTOR = "sector"

SOURCE_V1_REUSE = "v1_reuse"  # computed by features.pipeline.compute_feature_panel(), unmodified
SOURCE_V1_UTIL_REUSE = "v3_new_via_v1_utils"  # v3/features/price_features.py etc.
SOURCE_V2_REUSE = "v3_new_via_v2_reuse"  # v3/features/cross_sectional.py
SOURCE_V3_NEW = "v3_new"  # genuinely new computation (market breadth, sector)

AVAILABILITY_CORE = "core"  # always computed, part of the default dataset
AVAILABILITY_CONDITIONAL = "conditional"  # opt-in (sector - see v3/features/sector.py)

LEAKAGE_RISK_NONE = "none (backward-only, mechanically verified)"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    category: str
    formula: str
    required_history: int
    availability: str
    leakage_risk: str
    source: str


def _spec(
    name: str, category: str, formula: str, required_history: int, source: str,
    availability: str = AVAILABILITY_CORE,
) -> FeatureSpec:
    return FeatureSpec(
        name=name, category=category, formula=formula, required_history=required_history,
        availability=availability, leakage_risk=LEAKAGE_RISK_NONE, source=source,
    )


_MOMENTUM: list[FeatureSpec] = [
    _spec(f"return_{w}d", CATEGORY_MOMENTUM, f"Close[t]/Close[t-{w}] - 1", w + 1, SOURCE_V1_REUSE)
    for w in (1, 3, 5, 10, 20, 60)
] + [
    _spec("return_120d", CATEGORY_MOMENTUM, "Close[t]/Close[t-120] - 1", 121, SOURCE_V1_UTIL_REUSE)
]

_MOVING_AVERAGE: list[FeatureSpec] = [
    _spec(f"close_to_sma_{w}", CATEGORY_MOVING_AVERAGE, f"Close/SMA{w} - 1", w,
          SOURCE_V1_REUSE if w in (5, 20) else SOURCE_V1_UTIL_REUSE)
    for w in (5, 10, 20, 60, 120)
] + [
    _spec("ma5_to_ma20", CATEGORY_MOVING_AVERAGE, "SMA5/SMA20 - 1", 20, SOURCE_V1_UTIL_REUSE),
    _spec("ma20_to_ma60", CATEGORY_MOVING_AVERAGE, "SMA20/SMA60 - 1", 60, SOURCE_V1_UTIL_REUSE),
    _spec("ma60_to_ma120", CATEGORY_MOVING_AVERAGE, "SMA60/SMA120 - 1", 120, SOURCE_V1_UTIL_REUSE),
]

_VOLATILITY: list[FeatureSpec] = [
    _spec(f"volatility_{w}d", CATEGORY_VOLATILITY, f"std(return_1d[t-{w - 1}..t])", w + 1,
          SOURCE_V1_REUSE)
    for w in (5, 10, 20)
] + [
    _spec("atr", CATEGORY_VOLATILITY, "Wilder-smoothed ATR, period=14", 15, SOURCE_V1_REUSE),
    _spec("atr_pct", CATEGORY_VOLATILITY, "ATR / Close", 15, SOURCE_V1_REUSE),
    _spec("volatility_change", CATEGORY_VOLATILITY, "5-day fractional slope of volatility_20d",
          26, SOURCE_V1_UTIL_REUSE),
]

_OSCILLATOR: list[FeatureSpec] = [
    _spec(f"rsi_{p}", CATEGORY_OSCILLATOR, "100 - 100/(1+RS), Wilder-smoothed", p + 1,
          SOURCE_V1_REUSE if p == 14 else SOURCE_V1_UTIL_REUSE)
    for p in (5, 14, 20)
]

_VOLUME: list[FeatureSpec] = [
    _spec(f"volume_ratio_{w}d", CATEGORY_VOLUME, f"Volume / SMA(Volume,{w})", w, SOURCE_V1_REUSE)
    for w in (5, 20)
] + [
    _spec("volume_trend", CATEGORY_VOLUME, "5-day fractional slope of SMA5(Volume)", 10,
          SOURCE_V1_REUSE),
    _spec("turnover", CATEGORY_VOLUME, "Close * Volume", 1, SOURCE_V1_UTIL_REUSE),
    _spec("turnover_ratio", CATEGORY_VOLUME, "Turnover / SMA(Turnover,20)", 20,
          SOURCE_V1_UTIL_REUSE),
]

_BREAKOUT_DRAWDOWN: list[FeatureSpec] = [
    _spec(f"distance_from_{w}d_high", CATEGORY_BREAKOUT_DRAWDOWN,
          f"Close/max(Close[t-{w - 1}..t]) - 1 (<=0)", w,
          SOURCE_V1_REUSE if w == 20 else SOURCE_V1_UTIL_REUSE)
    for w in (20, 60, 120)
] + [
    _spec(f"drawdown_from_{w}d_high", CATEGORY_BREAKOUT_DRAWDOWN,
          f"-distance_from_{w}d_high (>=0)", w,
          SOURCE_V1_REUSE if w == 20 else SOURCE_V1_UTIL_REUSE)
    for w in (20, 60, 120)
]

_CROSS_SECTIONAL: list[FeatureSpec] = [
    _spec("return_percentile", CATEGORY_CROSS_SECTIONAL,
          "day-grouped percentile rank of return_5d", 6, SOURCE_V2_REUSE),
    _spec("volume_percentile", CATEGORY_CROSS_SECTIONAL,
          "day-grouped percentile rank of volume_ratio_20d", 20, SOURCE_V2_REUSE),
    _spec("volatility_percentile", CATEGORY_CROSS_SECTIONAL,
          "day-grouped percentile rank of volatility_20d", 21, SOURCE_V2_REUSE),
    _spec("momentum_percentile", CATEGORY_CROSS_SECTIONAL,
          "day-grouped percentile rank of return_20d", 21, SOURCE_V2_REUSE),
    _spec("drawdown_percentile", CATEGORY_CROSS_SECTIONAL,
          "day-grouped percentile rank of distance_from_20d_high", 20, SOURCE_V2_REUSE),
    _spec("relative_strength_percentile", CATEGORY_CROSS_SECTIONAL,
          "day-grouped percentile rank of rs_20d", 21, SOURCE_V2_REUSE),
]

_MARKET_CONTEXT: list[FeatureSpec] = [
    _spec(f"topix_return_{w}d", CATEGORY_MARKET_CONTEXT,
          f"TOPIX Proxy Close[t]/Close[t-{w}] - 1", w + 1, SOURCE_V1_REUSE)
    for w in (1, 3, 5, 10, 20, 60)
] + [
    _spec("topix_volatility_20d", CATEGORY_MARKET_CONTEXT,
          "20-day std. dev. of TOPIX Proxy 1-day return", 21, SOURCE_V1_REUSE),
    _spec("topix_drawdown_20d", CATEGORY_MARKET_CONTEXT,
          "TOPIX Proxy distance from its 20-day rolling high", 20, SOURCE_V1_REUSE),
    _spec("advancing_ratio", CATEGORY_MARKET_CONTEXT,
          "fraction of Universe with return_1d > 0, same day", 2, SOURCE_V3_NEW),
    _spec("declining_ratio", CATEGORY_MARKET_CONTEXT,
          "fraction of Universe with return_1d < 0, same day", 2, SOURCE_V3_NEW),
    _spec("market_breadth", CATEGORY_MARKET_CONTEXT, "advancing_ratio - declining_ratio", 2,
          SOURCE_V3_NEW),
]

_SECTOR: list[FeatureSpec] = [
    _spec("industry_return", CATEGORY_SECTOR,
          "mean return_1d within the same sector33 group, same day", 2, SOURCE_V3_NEW,
          availability=AVAILABILITY_CONDITIONAL),
    _spec("stock_vs_industry", CATEGORY_SECTOR, "return_1d - industry_return", 2, SOURCE_V3_NEW,
          availability=AVAILABILITY_CONDITIONAL),
    _spec("industry_relative_strength", CATEGORY_SECTOR,
          "return_20d - mean(return_20d) within the same sector33 group, same day", 21,
          SOURCE_V3_NEW, availability=AVAILABILITY_CONDITIONAL),
]

FEATURE_REGISTRY: list[FeatureSpec] = (
    _MOMENTUM + _MOVING_AVERAGE + _VOLATILITY + _OSCILLATOR + _VOLUME + _BREAKOUT_DRAWDOWN
    + _CROSS_SECTIONAL + _MARKET_CONTEXT + _SECTOR
)

CORE_FEATURE_NAMES: list[str] = [
    f.name for f in FEATURE_REGISTRY if f.availability == AVAILABILITY_CORE
]
CONDITIONAL_FEATURE_NAMES: list[str] = [
    f.name for f in FEATURE_REGISTRY if f.availability == AVAILABILITY_CONDITIONAL
]


def feature_registry_by_name() -> dict[str, FeatureSpec]:
    return {f.name: f for f in FEATURE_REGISTRY}
