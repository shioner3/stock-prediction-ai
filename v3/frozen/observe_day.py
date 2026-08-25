"""One day's V3 Forward Observation: builds today's Feature panel from
the (already-fetched, by V1's own daily step) OHLCV cache, runs all 16
Frozen Models, ranks cross-sectionally per model, and returns the
PredictionLogEntry rows to append. Reuses `v3.dataset.build_v3_dataset()`
UNCHANGED (same Feature/Target computation every V3 Phase has used),
pointed at a different `source_processed_dir` via a plain config copy -
no new Feature/Target code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime

import pandas as pd

from backtest.market_regime import compute_market_regime
from scoring.validation import assign_quantile_buckets
from v3.config.loader import V3Config
from v3.dataset import build_v3_dataset
from v3.frozen.manifest import FrozenModelSpec
from v3.frozen.observation_log import PredictionLogEntry
from v3.frozen.predict import predict_with_all_frozen_models


@dataclass(frozen=True)
class ObservationBuildResult:
    observation_date: date_type
    universe_size: int
    entries: list[PredictionLogEntry]
    feature_panel: pd.DataFrame  # kept for leakage/integrity tests - not logged verbatim


def build_observation_entries(
    tickers: list[str], v3_config: V3Config, observation_date: date_type,
    frozen_specs: list[FrozenModelSpec], market_regime_config, topix_ohlcv: pd.DataFrame,
) -> ObservationBuildResult:
    dataset = build_v3_dataset(tickers, v3_config)
    today_panel = dataset[dataset["date"] == observation_date].copy()
    if today_panel.empty:
        return ObservationBuildResult(observation_date, 0, [], today_panel)

    predictions_by_model = predict_with_all_frozen_models(frozen_specs, today_panel)

    regime_df = compute_market_regime(topix_ohlcv, market_regime_config)
    regime_today = regime_df.loc[regime_df["date"] == observation_date, "regime"]
    regime_label = str(regime_today.iloc[0]) if len(regime_today) else None

    logged_at = datetime.now().isoformat()
    entries: list[PredictionLogEntry] = []
    for spec in frozen_specs:
        preds = pd.Series(predictions_by_model[spec.model_id], index=today_panel.index)
        valid = preds.dropna()
        if valid.empty:
            continue
        buckets = assign_quantile_buckets(valid)
        ranks = valid.rank(ascending=False, method="first").astype(int)
        percentiles = valid.rank(pct=True)
        for idx in valid.index:
            ticker = today_panel.loc[idx, "ticker"]
            entries.append(
                PredictionLogEntry(
                    observation_date=observation_date.isoformat(), ticker=str(ticker),
                    model_id=spec.model_id, target_definition=spec.target_definition,
                    horizon=spec.horizon, prediction=float(valid.loc[idx]),
                    rank=int(ranks.loc[idx]), percentile=float(percentiles.loc[idx]),
                    bucket=str(buckets.loc[idx]), regime=regime_label, data_quality="OK",
                    logged_at=logged_at,
                )
            )

    return ObservationBuildResult(observation_date, len(today_panel), entries, today_panel)
