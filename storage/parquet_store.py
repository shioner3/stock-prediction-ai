"""Parquet read/write helpers for OHLCV, universe snapshot, and feature
panel data.

data/raw/ holds exactly what the provider returned, standardized in
column shape only, unmodified in content - kept for audit.
data/processed/ holds validated + cleaned data used by downstream feature
code. data/features/ holds the Phase 2 feature panels built from
data/processed/ (see pipeline/build_features.py).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _ticker_path(base_dir: Path, ticker: str) -> Path:
    safe_ticker = ticker.replace("/", "_").replace("^", "_")
    return Path(base_dir) / f"{safe_ticker}.parquet"


def save_ohlcv(df: pd.DataFrame, ticker: str, base_dir: Path) -> Path:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    path = _ticker_path(base_dir, ticker)
    df.to_parquet(path, engine="pyarrow", index=False)
    return path


def load_ohlcv(ticker: str, base_dir: Path) -> pd.DataFrame:
    path = _ticker_path(base_dir, ticker)
    return pd.read_parquet(path, engine="pyarrow")


def save_universe_snapshot(df: pd.DataFrame, run_date: str, base_dir: Path) -> Path:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"universe_{run_date}.parquet"
    df.to_parquet(path, engine="pyarrow", index=False)
    return path


def save_feature_panel(df: pd.DataFrame, ticker: str, base_dir: Path) -> Path:
    """Same on-disk layout as save_ohlcv (one ticker per Parquet file) -
    kept as its own named function so callers reading pipeline/
    build_features.py don't have to know that's an implementation detail.
    """
    return save_ohlcv(df, ticker, base_dir)


def load_feature_panel(ticker: str, base_dir: Path) -> pd.DataFrame:
    return load_ohlcv(ticker, base_dir)


def save_signal_records(df: pd.DataFrame, ticker: str, base_dir: Path) -> Path:
    """Same on-disk layout as save_ohlcv - one ticker per Parquet file,
    holding that ticker's triggered=True SignalRecord rows (see
    signals/pipeline.py::compute_signal_records).
    """
    return save_ohlcv(df, ticker, base_dir)


def load_signal_records(ticker: str, base_dir: Path) -> pd.DataFrame:
    return load_ohlcv(ticker, base_dir)


def save_score_records(df: pd.DataFrame, ticker: str, base_dir: Path) -> Path:
    """Same on-disk layout as save_ohlcv - one ticker per Parquet file,
    holding that ticker's ScoreRecord rows (see
    scoring/pipeline.py::compute_score_records). Only rows where a
    Signal triggered ever appear here - see that module's docstring.
    """
    return save_ohlcv(df, ticker, base_dir)


def load_score_records(ticker: str, base_dir: Path) -> pd.DataFrame:
    return load_ohlcv(ticker, base_dir)
