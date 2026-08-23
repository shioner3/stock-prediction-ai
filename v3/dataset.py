"""V3 dataset construction (spec section 2): OHLCV -> V1 Feature panel
(`features.pipeline.compute_feature_panel()`, unmodified) -> V3 derived
Features (`v3/features/price_features.py`) -> Targets
(`v3/targets/compute.py`) -> stack every ticker -> cross-sectional Features
(`v3/features/cross_sectional.py`) + Market context
(`v3/features/market_context.py`), which can only be computed once every
ticker's rows share one Universe-wide panel.

Reads V1's ALREADY-FETCHED Full Universe caches (READ ONLY -
`pipeline.universe_ingest.load_manifest()`, `storage.parquet_store.
load_ohlcv()`) - the same convention `v2/pipeline.py` already established
for V2; never re-fetches from providers/, never writes into any V1-owned
directory.
"""

from __future__ import annotations

import logging

import pandas as pd

from features.pipeline import compute_feature_panel
from pipeline.universe_ingest import load_manifest
from storage.parquet_store import load_ohlcv
from v3.config.loader import V3Config
from v3.features.cross_sectional import add_v3_cross_sectional_features
from v3.features.market_context import compute_market_breadth, compute_market_context_panel
from v3.features.price_features import add_v3_price_features
from v3.features.registry import CORE_FEATURE_NAMES
from v3.targets.compute import compute_v3_targets
from v3.targets.registry import TARGET_COLUMN_NAMES

logger = logging.getLogger(__name__)

MARKET_TICKER = "TOPIX"


def load_universe_tickers(config: V3Config) -> list[str]:
    manifest = load_manifest(config.source_universe_manifest)
    entries: dict = manifest.get("tickers", {})
    return sorted(t for t, e in entries.items() if e.get("included_in_universe"))


def build_ticker_panel(
    ticker: str, market_ohlcv: pd.DataFrame, config: V3Config
) -> pd.DataFrame | None:
    """One ticker's full V3 panel (V1 Feature columns + V3 derived
    Features + all 16 Target columns). Returns None if this ticker has
    no cached OHLCV (never raises - a missing ticker is simply excluded,
    the same convention v2/pipeline.py::build_ticker_panel() uses).
    """
    try:
        ohlcv = load_ohlcv(ticker, config.source_processed_dir)
    except FileNotFoundError:
        return None
    if ohlcv.empty:
        return None
    v1_panel = compute_feature_panel(ohlcv, market_df=market_ohlcv)
    v3_panel = add_v3_price_features(v1_panel)
    v3_panel = compute_v3_targets(v3_panel, market_ohlcv)
    return v3_panel


def build_universe_panel(tickers: list[str], config: V3Config) -> pd.DataFrame:
    """Every ticker's panel, stacked, PLUS the cross-sectional/market-
    context columns that only make sense once every ticker shares one
    panel. Returns an empty DataFrame if no ticker had usable data.
    """
    market_ohlcv = load_ohlcv(MARKET_TICKER, config.source_processed_dir)

    frames = []
    for ticker in tickers:
        panel = build_ticker_panel(ticker, market_ohlcv, config)
        if panel is not None and not panel.empty:
            frames.append(panel)
    if not frames:
        return pd.DataFrame()
    stacked = pd.concat(frames, ignore_index=True)

    stacked = add_v3_cross_sectional_features(stacked)

    market_context = compute_market_context_panel(market_ohlcv)
    stacked = stacked.merge(market_context, on="date", how="left")

    breadth = compute_market_breadth(stacked)
    stacked = stacked.merge(breadth, on="date", how="left")

    return stacked


def select_dataset_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Narrows the full (V1 + V3) panel down to date/ticker + the Feature
    Registry's CORE columns + all 16 Target columns - the actual "V3
    Dataset" spec section 6 asks for (the wide intermediate V1 panel is
    not the deliverable; "Feature数を無制限に増やさない").
    """
    keep = ["date", "ticker", *CORE_FEATURE_NAMES, *TARGET_COLUMN_NAMES]
    existing = [c for c in keep if c in panel.columns]
    return panel[existing]


def build_v3_dataset(tickers: list[str], config: V3Config) -> pd.DataFrame:
    logger.info("V3: building panel for %d tickers", len(tickers))
    universe_panel = build_universe_panel(tickers, config)
    logger.info("V3: dataset has %d rows", len(universe_panel))
    return select_dataset_columns(universe_panel)


# spec section 10: "ランダムtrain/test splitは禁止。必ず時間順split" -
# every horizon's Target at a TRAIN-boundary row reads Close[t+h], which
# can fall inside the TEST period for rows near the boundary. An EMBARGO
# gap between train_end and test_start (>= the largest Horizon in
# targets.registry.HORIZONS) removes this ambiguity structurally, rather
# than requiring every caller to reason about it per-row. This is the
# "Dataset APIを時間軸対応にする" surface spec section 10 asks V3-2 to
# provide, ready for V3-3's fuller Walk-Forward implementation to build on
# without redesigning the split primitive.
def time_split(
    dataset: pd.DataFrame, train_end: object, test_start: object, date_col: str = "date"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """TRAIN = rows with date <= train_end. TEST = rows with date >=
    test_start. Rows strictly between the two (the embargo) are excluded
    from both - callers should choose test_start - train_end >= the
    largest Horizon they intend to evaluate (see module docstring).
    """
    train = dataset[dataset[date_col] <= train_end].copy()
    test = dataset[dataset[date_col] >= test_start].copy()
    return train, test
