"""Auxiliary date/ticker/close/volume/sector/scale lookup (spec sections
10-12's Stock/Sector Concentration and Matched Control need raw price,
turnover proxy, and JPX segment/size classification - none of which
survive `v3.dataset.select_dataset_columns()`'s narrowing to Feature
Registry + Target Registry columns only). Reads V1's already-cached OHLCV
(`storage.parquet_store.load_ohlcv`, unmodified, same call `v3.dataset.
build_ticker_panel()` already makes) and V2-3's local JPX master cache
(`v2.causal.segment`, unmodified) - no new network fetch, no new Feature
Registry entry, purely descriptive lookups joined onto predictions after
the fact.
"""

from __future__ import annotations

import pandas as pd

from storage.parquet_store import load_ohlcv
from v2.causal.segment import attach_segment_columns, load_ticker_segment_map
from v3.config.loader import V3Config


def build_price_volume_panel(tickers: list[str], config: V3Config) -> pd.DataFrame:
    """date/ticker/close/volume, one row per ticker per trading day -
    tickers with no cached OHLCV are simply skipped (mirrors `v3.dataset.
    build_ticker_panel()`'s own missing-ticker convention).
    """
    frames = []
    for ticker in tickers:
        try:
            ohlcv = load_ohlcv(ticker, config.source_processed_dir)
        except FileNotFoundError:
            continue
        if ohlcv.empty:
            continue
        frame = ohlcv[["date", "close", "volume"]].copy()
        frame["ticker"] = ticker
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "close", "volume"])
    return pd.concat(frames, ignore_index=True)


def attach_price_volume(
    predictions: pd.DataFrame, price_volume_panel: pd.DataFrame
) -> pd.DataFrame:
    return predictions.merge(price_volume_panel, on=["date", "ticker"], how="left")


def attach_sector_and_scale(predictions: pd.DataFrame) -> pd.DataFrame:
    """Adds market_segment/sector33/scale columns via the SAME local JPX
    master cache Phase V2-3 already used - carries the same current-day-
    snapshot-projected-backward caveat V2-3 documented (see `v2.causal.
    segment`'s own module docstring), noted again in this Phase's own
    Limitations section rather than silently assumed away.
    """
    segment_map = load_ticker_segment_map()
    return attach_segment_columns(predictions, segment_map)


def attach_turnover(price_volume_predictions: pd.DataFrame) -> pd.DataFrame:
    """Close * Volume - the same turnover-as-liquidity-proxy convention
    already established by `v2.causal.segment.attach_liquidity_columns()`
    and by the V3 Feature Registry's own `turnover` Feature (`v3/features/
    registry.py`) - computed here directly from the merged close/volume
    since predictions rows don't carry the Feature Registry's `turnover`
    column (V3-3's per-window predictions only ever kept date/ticker/
    actual/prediction, see `v3/validation/train_predict.py`).
    """
    out = price_volume_predictions.copy()
    out["turnover"] = out["close"] * out["volume"]
    return out
