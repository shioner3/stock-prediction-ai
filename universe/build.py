"""Universe construction: load the JPX listed-company master list and apply
static (market-segment / instrument-type) filters.

Price/liquidity filters that require actual fetched OHLCV data live in
universe/filters.py and are applied later in the pipeline, after data has
been fetched - they cannot be evaluated from the master list alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_MASTER_COLUMNS = {"code", "name", "market_segment", "sector33"}

# Substring match against the sector33 classification text. The real JPX
# master list labels ETFs/ETNs and REITs distinctly in this column (see
# data/reference/jpx_listed_companies.sample.csv for the expected format).
ETF_KEYWORDS = ["ETF", "ETN"]
REIT_KEYWORDS = ["REIT", "不動産投資"]


@dataclass
class UniverseBuildResult:
    included: pd.DataFrame
    excluded: pd.DataFrame


def load_master_list(path: Path) -> pd.DataFrame:
    """Load a JPX-style listed-company master list.

    Expected columns: code, name, market_segment, sector33.

    Production use requires downloading the official list from JPX
    ("東証上場銘柄一覧") and placing it at this path (or pointing
    config/settings.yaml's universe.master_list_path elsewhere) - the file
    shipped in data/reference/ is a small hand-written sample for
    development and tests only, not the real universe.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"universe master list not found at {path}. Download the JPX "
            "listed-company list and place it here, or point "
            "config/settings.yaml's universe.master_list_path elsewhere."
        )
    df = pd.read_csv(path, dtype={"code": str})
    missing = REQUIRED_MASTER_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"universe master list at {path} is missing columns: {sorted(missing)}")
    return df


def classify_instrument_type(sector33: str) -> str:
    text = str(sector33)
    if any(k in text.upper() for k in ETF_KEYWORDS):
        return "ETF"
    if any(k in text for k in REIT_KEYWORDS) or "REIT" in text.upper():
        return "REIT"
    return "STOCK"


def apply_static_filters(
    df: pd.DataFrame,
    segments: list[str],
    exclude_etf: bool = True,
    exclude_reit: bool = True,
) -> UniverseBuildResult:
    """df must have "code"/"name"/"market_segment" columns, plus either
    "instrument_type" (universe/jpx_master.py's real-JPX-file output,
    classified directly from 市場・商品区分 - preferred, more reliable)
    or "sector33" (the Phase 1 sample-CSV shape, classified here via the
    sector33-keyword heuristic classify_instrument_type() for backward
    compatibility with that fixture/test data).
    """
    df = df.copy()
    if "instrument_type" not in df.columns:
        df["instrument_type"] = df["sector33"].apply(classify_instrument_type)

    reasons = pd.Series([""] * len(df), index=df.index, dtype=object)
    exclude_mask = pd.Series([False] * len(df), index=df.index)

    segment_mask = ~df["market_segment"].isin(segments)
    reasons = reasons.mask(segment_mask, f"market_segment not in {segments}")
    exclude_mask |= segment_mask

    if exclude_etf:
        etf_mask = (df["instrument_type"] == "ETF") & ~exclude_mask
        reasons = reasons.mask(etf_mask, "instrument_type=ETF")
        exclude_mask |= etf_mask

    if exclude_reit:
        reit_mask = (df["instrument_type"] == "REIT") & ~exclude_mask
        reasons = reasons.mask(reit_mask, "instrument_type=REIT")
        exclude_mask |= reit_mask

    df["exclusion_reason"] = reasons
    included = df.loc[~exclude_mask].drop(columns=["exclusion_reason"]).reset_index(drop=True)
    excluded = df.loc[exclude_mask].reset_index(drop=True)
    return UniverseBuildResult(included=included, excluded=excluded)
