"""V3 Industry/Sector relative strength (spec section 5: "利用可能なデータ
がある場合" - explicitly optional). Reuses `universe/jpx_master.py`'s LOCAL
CACHED snapshot (`data/reference/jpx_master_current.xls`, the same file
`v2/causal/segment.py` already reads - no new network fetch) for the
sector33 mapping.

Marked `availability="conditional"` in the Feature Registry (spec
section 6) and NOT included in the default dataset build
(`v3/dataset.py`) - callers opt in explicitly via `attach_sector_features()`.
This is a CURRENT-DAY snapshot projected backward across the whole
2022-2026 sample (the same survivorship-bias caveat Phase V2-3's report
documented for this same file), so it is kept structurally separate
rather than silently baked into the core panel.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from universe.jpx_master import load_jpx_master

JPX_MASTER_CACHE_PATH = Path("data/reference/jpx_master_current.xls")


def load_ticker_sector_map(path: Path = JPX_MASTER_CACHE_PATH) -> pd.DataFrame:
    segment_map = load_jpx_master(path, force_refresh=False)
    return segment_map[["code", "sector33"]].rename(columns={"code": "ticker"})


def attach_sector_features(
    universe_panel: pd.DataFrame,
    sector_map: pd.DataFrame,
    date_col: str = "date",
    return_col: str = "return_1d",
    return_20d_col: str = "return_20d",
) -> pd.DataFrame:
    """Adds industry_return/stock_vs_industry/industry_relative_strength -
    all computed via groupby([date, sector33]).transform("mean"), so each
    row's industry_return is the SAME-DAY sector average (a pure
    cross-sectional aggregate, never any other date's rows).
    """
    merged = universe_panel.merge(sector_map, on="ticker", how="left")
    merged["industry_return"] = merged.groupby([date_col, "sector33"])[return_col].transform(
        "mean"
    )
    industry_return_20d = merged.groupby([date_col, "sector33"])[return_20d_col].transform("mean")
    merged["stock_vs_industry"] = merged[return_col] - merged["industry_return"]
    merged["industry_relative_strength"] = merged[return_20d_col] - industry_return_20d
    return merged
