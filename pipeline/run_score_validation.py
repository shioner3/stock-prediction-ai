"""Phase 5 Score Validation orchestration: joins Score Records
(data/scores/*) with Forward Targets computed fresh from data/features/*
(targets/forward_returns.py - never stored as a Feature, never fed back
into Score) and runs bucket analysis independently for every
(direction, signal_name, forward_window) combination.

This module does not conclude anything about Score's validity - it only
produces the numbers (see scoring/validation.py's docstring).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest.metrics import BacktestMetrics
from config.loader import AppConfig
from scoring.validation import (
    FIXED_BUCKET_LABELS,
    assign_fixed_buckets,
    assign_quantile_buckets,
    check_monotonicity,
    compute_bucket_metrics,
    monotonicity_correlation,
)
from storage.parquet_store import load_feature_panel, load_score_records
from targets.forward_returns import FORWARD_WINDOWS, compute_forward_returns, compute_mfe_mae

logger = logging.getLogger(__name__)

QUANTILE_BUCKET_LABELS = [f"Q{i + 1}" for i in range(5)]


@dataclass
class GroupValidationResult:
    direction: str
    signal_name: str
    forward_window: int
    bucket_method: str
    n: int
    bucket_metrics: dict[str, BacktestMetrics]
    monotonic: bool | None
    monotonicity_corr: float | None


def _scored_with_targets_for_ticker(
    ticker: str, scores_dir: Path, features_dir: Path
) -> pd.DataFrame | None:
    try:
        score_records = load_score_records(ticker, scores_dir)
    except FileNotFoundError:
        # pipeline/build_scores.py deliberately writes no file for a
        # ticker with zero triggered signals (nothing to score) - at
        # Full Universe scale this is common (short-history/thinly
        # traded tickers), not an error, so it's handled here rather
        # than falling through to build_scored_with_targets' broad
        # except-and-log-ERROR below.
        return None
    if score_records.empty:
        return None
    panel = load_feature_panel(ticker, features_dir)

    forward = compute_forward_returns(panel)
    forward_with_date = panel[["date"]].join(forward)
    merged = score_records.merge(forward_with_date, on="date", how="left")

    parts = []
    for direction_value, group in merged.groupby("direction"):
        mfe_mae = compute_mfe_mae(panel, direction_value)
        mfe_mae_with_date = panel[["date"]].join(mfe_mae)
        parts.append(group.merge(mfe_mae_with_date, on="date", how="left"))

    return pd.concat(parts, ignore_index=True)


def build_scored_with_targets(
    config: AppConfig, tickers: list[str] | None = None
) -> pd.DataFrame:
    scores_dir = Path(config.data.scores_dir)
    features_dir = Path(config.data.features_dir)

    if tickers is None:
        tickers = sorted(p.stem for p in scores_dir.glob("*.parquet"))

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        try:
            frame = _scored_with_targets_for_ticker(ticker, scores_dir, features_dir)
            if frame is not None:
                frames.append(frame)
        except Exception:
            logger.exception("score validation data prep failed for %s", ticker)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_score_validation(
    config: AppConfig, tickers: list[str] | None = None, bucket_method: str = "fixed"
) -> list[GroupValidationResult]:
    start_time = time.monotonic()
    scored = build_scored_with_targets(config, tickers)
    results: list[GroupValidationResult] = []
    if scored.empty:
        return results

    if bucket_method == "fixed":
        scored = scored.assign(score_bucket=assign_fixed_buckets(scored["total_score"]))
        bucket_order = FIXED_BUCKET_LABELS
    else:
        scored = scored.assign(score_bucket=assign_quantile_buckets(scored["total_score"]))
        bucket_order = QUANTILE_BUCKET_LABELS

    for (direction, signal_name), group in scored.groupby(["direction", "signal_name"]):
        for window in FORWARD_WINDOWS:
            return_col = f"forward_return_{window}d"
            bucket_metrics = compute_bucket_metrics(group, "score_bucket", return_col)
            results.append(
                GroupValidationResult(
                    direction=direction,
                    signal_name=signal_name,
                    forward_window=window,
                    bucket_method=bucket_method,
                    n=len(group),
                    bucket_metrics=bucket_metrics,
                    monotonic=check_monotonicity(bucket_metrics, bucket_order),
                    monotonicity_corr=monotonicity_correlation(bucket_metrics, bucket_order),
                )
            )

    logger.info(
        "score validation complete: rows=%d groups=%d duration=%.1fs",
        len(scored), len(results), time.monotonic() - start_time,
    )
    return results
