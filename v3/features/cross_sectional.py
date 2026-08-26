"""V3 Cross-sectional percentile features (spec section 5's "Cross-sectional
features" category). Reuses `v2/ranking/cross_sectional.py::percentile_rank_by_day()`
UNMODIFIED (frozen V2 code - the same day-grouped percentile-rank primitive
V2-1's own Score already uses) rather than reimplementing an identical
formula. This is a READ-only import of a pure function; nothing in v2/ is
written to or changed.

Each percentile column summarizes exactly ONE already-computed raw column,
chosen and fixed here BEFORE any dataset run - never picked or swapped
after seeing OOS results (spec section 27's explicit prohibition).
"""

from __future__ import annotations

import pandas as pd

from v2.ranking.cross_sectional import percentile_rank_by_day

# percentile_column_name -> (source_column, higher_is_better)
CROSS_SECTIONAL_FEATURES: dict[str, tuple[str, bool]] = {
    "return_percentile": ("return_5d", True),
    "volume_percentile": ("volume_ratio_20d", True),
    "volatility_percentile": ("volatility_20d", True),
    "momentum_percentile": ("return_20d", True),
    # distance_from_20d_high is <= 0 (0 = at the high); higher_is_better=True
    # means "closer to the high" gets the higher percentile.
    "drawdown_percentile": ("distance_from_20d_high", True),
    "relative_strength_percentile": ("rs_20d", True),
}


def add_v3_cross_sectional_features(panel: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """panel: the FULL stacked Universe panel (every ticker's rows for
    every date) - percentiles are only meaningful cross-sectionally, so
    (unlike price_features.py) this must run AFTER every ticker's panel
    is concatenated, never per-ticker.
    """
    out = panel.copy()
    for pct_name, (source_col, higher_is_better) in CROSS_SECTIONAL_FEATURES.items():
        out[pct_name] = percentile_rank_by_day(
            panel, source_col, date_col=date_col, higher_is_better=higher_is_better
        )
    return out
