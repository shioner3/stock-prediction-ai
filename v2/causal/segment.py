"""Sector / Market Segment / Size / Liquidity profile of Q1 (spec sections
18/19): is Q1's negative relationship concentrated in a particular market
segment, industry, size class, or is it simply "Q1 = small/illiquid
stocks"?

Uses universe/jpx_master.py's LOCAL CACHED snapshot (data/reference/
jpx_master_current.xls, already on disk from an earlier Phase - no new
network fetch) for market_segment (Prime/Standard/Growth), sector33, and
scale (規模区分, JPX's own TOPIX Core30/Large70/Mid400/Small size
classification - a real, official size proxy, not a new Feature this
Phase invents). This file is a CURRENT-DAY snapshot only (jpx_master.py's
own documented survivorship-bias caveat: today's classification is
projected backward across the whole 2022-2026 sample) - carried into this
Phase's own Limitations section, not silently assumed away.

Liquidity uses the ALREADY-PRESENT raw close/volume columns that survive
into V2-1's ranked panel (v2/pipeline.py merges V1's cached Feature panel,
which includes close/volume, straight through) - no new Feature column is
defined; this is a one-off descriptive turnover proxy (close * volume)
computed inline for this report only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from universe.jpx_master import load_jpx_master
from v2.stats import ReturnStats, compute_return_stats

JPX_MASTER_CACHE_PATH = Path("data/reference/jpx_master_current.xls")


def load_ticker_segment_map(path: Path = JPX_MASTER_CACHE_PATH) -> pd.DataFrame:
    """code/market_segment/sector33/sector17/scale, one row per ticker -
    force_refresh=False always (never re-fetches from JPX's live server;
    fails loudly if the cached file is ever removed rather than silently
    downloading a possibly-different snapshot mid-Phase)."""
    return load_jpx_master(path, force_refresh=False)


def attach_segment_columns(scored: pd.DataFrame, segment_map: pd.DataFrame) -> pd.DataFrame:
    lookup = segment_map[["code", "market_segment", "sector33", "scale"]].rename(
        columns={"code": "ticker"}
    )
    return scored.merge(lookup, on="ticker", how="left")


@dataclass(frozen=True)
class GroupReturnStats:
    group_value: str
    n: int
    stats: ReturnStats


def compute_group_stats(
    scored_with_segments: pd.DataFrame,
    group_col: str,
    return_col: str,
    bucket_col: str = "score_bucket",
    bucket_label: str = "Q1",
) -> list[GroupReturnStats]:
    subset = scored_with_segments[scored_with_segments[bucket_col] == bucket_label].dropna(
        subset=[return_col, group_col]
    )
    results = []
    for value, group in subset.groupby(group_col, observed=True):
        results.append(
            GroupReturnStats(
                group_value=str(value), n=len(group), stats=compute_return_stats(group[return_col])
            )
        )
    return sorted(results, key=lambda r: r.n, reverse=True)


LIQUIDITY_TURNOVER_COL = "_turnover"


def attach_liquidity_columns(scored: pd.DataFrame) -> pd.DataFrame:
    """Adds a same-day turnover proxy (close * volume) - descriptive only,
    computed inline, never persisted as a Feature column V2-1's Score
    could reference."""
    out = scored.copy()
    out[LIQUIDITY_TURNOVER_COL] = out["close"] * out["volume"]
    return out


@dataclass(frozen=True)
class LiquidityProfile:
    bucket: str
    n: int
    turnover_mean: float | None
    turnover_median: float | None
    close_mean: float | None
    close_median: float | None
    volume_mean: float | None
    volume_median: float | None


def compute_liquidity_profile(
    scored_with_liquidity: pd.DataFrame, bucket_col: str = "score_bucket", bucket_label: str = "Q1"
) -> LiquidityProfile:
    subset = scored_with_liquidity[scored_with_liquidity[bucket_col] == bucket_label]
    turnover = subset[LIQUIDITY_TURNOVER_COL].dropna()
    close = subset["close"].dropna()
    volume = subset["volume"].dropna()
    return LiquidityProfile(
        bucket=bucket_label,
        n=len(subset),
        turnover_mean=float(turnover.mean()) if len(turnover) else None,
        turnover_median=float(turnover.median()) if len(turnover) else None,
        close_mean=float(close.mean()) if len(close) else None,
        close_median=float(close.median()) if len(close) else None,
        volume_mean=float(volume.mean()) if len(volume) else None,
        volume_median=float(volume.median()) if len(volume) else None,
    )
