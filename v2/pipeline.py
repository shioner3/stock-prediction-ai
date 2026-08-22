"""V2 orchestration: Universe -> Feature/Target -> Cross-sectional Rank ->
V2 Score -> Candidate Ranking (spec section 1/2/21).

Reads V1's ALREADY-FETCHED Full Universe caches (READ ONLY -
pipeline.universe_ingest.load_manifest(), storage.parquet_store.
load_ohlcv()/load_feature_panel()) rather than re-fetching from
providers/ - spec section 16 treats the existing 2022-2026 dataset as
V2-1's Research/Development Dataset, not a fresh fetch target, and spec
section 3 forbids V2 from writing into any V1-owned directory. This
module never calls save_ohlcv()/save_feature_panel() etc. against a V1
path - only against config.v2_*_dir (data/v2/*), via v2/manifest.py and
the CLI script.

Uses V1's CACHED Feature panel (add_v2_derived_features(), the fast
path - see v2/features_adapter.py) rather than recomputing
compute_feature_panel() from raw OHLCV for all ~2,880 tickers, since
V1's cache is already byte-identical to what a fresh compute would
produce (same frozen, unmodified V1 code) - see
tests/test_v2_pipeline.py for the equivalence check against
build_v2_feature_panel()'s fresh-compute path.
"""

from __future__ import annotations

import logging
from datetime import date as date_type

import pandas as pd

from pipeline.universe_ingest import load_manifest
from storage.parquet_store import load_feature_panel
from v2.candidate import CandidateRecord, build_candidate_table
from v2.config.loader import V2Config
from v2.features_adapter import add_v2_derived_features
from v2.ranking.score import CATEGORY_RANK_COLUMNS, compute_category_ranks, compute_v2_score
from v2.targets_adapter import build_v2_forward_targets

logger = logging.getLogger(__name__)

MARKET_TICKER = "TOPIX"


def load_universe_tickers(config: V2Config) -> list[str]:
    manifest = load_manifest(config.source_universe_manifest)
    entries: dict = manifest.get("tickers", {})
    return sorted(t for t, e in entries.items() if e.get("included_in_universe"))


def build_ticker_panel(ticker: str, config: V2Config) -> pd.DataFrame | None:
    """One ticker's V2 panel: V1's cached Feature columns + V2's derived
    columns + Forward Target columns, all in one row-per-date frame.
    Returns None if this ticker has no cached Feature panel (never
    raises - a missing ticker is simply excluded, same convention every
    other Phase's per-ticker loop in this project already uses).
    """
    try:
        v1_panel = load_feature_panel(ticker, config.source_features_dir)
    except FileNotFoundError:
        return None
    v2_panel = add_v2_derived_features(v1_panel)
    targets = build_v2_forward_targets(v1_panel)
    return pd.concat([v2_panel, targets], axis=1)


def build_universe_panel(tickers: list[str], config: V2Config) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        panel = build_ticker_panel(ticker, config)
        if panel is not None and not panel.empty:
            frames.append(panel)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_v2_rankings(universe_panel: pd.DataFrame, config: V2Config) -> pd.DataFrame:
    """Adds category rank + total_score columns to universe_panel, cross-
    sectionally per date (v2/ranking - see that package for the no-
    pooling-across-days guarantee). Returns a NEW DataFrame; does not
    mutate universe_panel.
    """
    if universe_panel.empty:
        return universe_panel
    category_ranks = compute_category_ranks(universe_panel)
    category_ranks = category_ranks.assign(
        total_score=compute_v2_score(category_ranks, config.score_weights)
    )
    return universe_panel.merge(category_ranks, on=["date", "ticker"], how="left")


def candidate_table_for_date(
    ranked_panel: pd.DataFrame,
    as_of_date: date_type,
    market_by_ticker: dict[str, str] | None = None,
) -> list[CandidateRecord]:
    day = ranked_panel[ranked_panel["date"] == as_of_date]
    return build_candidate_table(day, list(CATEGORY_RANK_COLUMNS.values()), market_by_ticker)


def run_v2_ranking(config: V2Config, tickers: list[str] | None = None) -> pd.DataFrame:
    """Full pipeline: load Universe (if tickers not given) -> build every
    ticker's panel -> cross-sectional rank + score. Returns the combined
    ranked panel (date, ticker, raw features, category ranks,
    total_score, forward_return_{n}d) - callers slice a single date via
    candidate_table_for_date() or run V2-1's own Q1-Q5/holding-period
    research statistics (v2/stats.py) directly against it.
    """
    if tickers is None:
        tickers = load_universe_tickers(config)
    logger.info("V2: building panel for %d tickers", len(tickers))
    universe_panel = build_universe_panel(tickers, config)
    logger.info("V2: ranking %d rows cross-sectionally", len(universe_panel))
    return compute_v2_rankings(universe_panel, config)
