"""Phase V3-7 Frozen Model integration smoke test - end to end on a small
synthetic Universe: train -> persist -> reload -> predict -> observe ->
log (idempotently) -> leakage shock check. Mirrors the established
V3-3/V3-4/V3-5 synthetic-data test pattern.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from v3_3_test_helpers import build_v3_3_config_and_tickers

from backtest.market_regime import compute_market_regime
from storage.parquet_store import load_ohlcv
from v3.dataset import build_v3_dataset
from v3.frozen.leakage_check import run_observation_shock_checks
from v3.frozen.manifest import (
    build_manifest,
    load_manifest_raw,
    save_manifest,
    verify_frozen_models_unchanged,
)
from v3.frozen.observation_log import append_prediction_entries, load_all_entries
from v3.frozen.observe_day import build_observation_entries
from v3.frozen.predict import predict_with_frozen_model
from v3.frozen.train import train_one_frozen_model
from v3.hash import hash_dataframe
from v3.residual.reproduce import build_augmented_dataset
from v3.robustness.aux_panel import attach_sector_and_scale
from v3.validation.wfo_config import MARKET_REGIME_CONFIG


def test_frozen_model_lifecycle_end_to_end(tmp_path: Path) -> None:
    config, tickers = build_v3_3_config_and_tickers(tmp_path, n_tickers=8, n_days=400)
    dataset = build_v3_dataset(tickers, config)
    ticker_frame = pd.DataFrame({"ticker": tickers})
    sector_map = attach_sector_and_scale(ticker_frame)[["ticker", "sector33"]].drop_duplicates(
        subset=["ticker"]
    )
    augmented = build_augmented_dataset(dataset, sector_map)
    dataset_hash = hash_dataframe(dataset)
    t0 = dataset["date"].max()

    artifact_dir = tmp_path / "artifacts"
    trained_raw = train_one_frozen_model(augmented, "raw", 5, t0, dataset_hash, artifact_dir)
    trained_beta = train_one_frozen_model(
        augmented, "beta_residual", 5, t0, dataset_hash, artifact_dir
    )

    # Reproducibility: loading the saved artifact must predict IDENTICALLY
    # to the in-memory model it was saved from.
    sample = augmented.dropna(subset=["target_raw_5d"]).head(5)
    in_memory_preds = trained_raw.model.predict(sample[trained_raw.training_set.X.columns])
    reloaded_preds = predict_with_frozen_model(trained_raw.spec, sample)
    assert (abs(in_memory_preds - reloaded_preds) < 1e-9).all()

    manifest = build_manifest(t0, [trained_raw.spec, trained_beta.spec])
    manifest_path = tmp_path / "v3_frozen_models_manifest.json"
    save_manifest(manifest, manifest_path)
    saved = load_manifest_raw(manifest_path)
    unchanged, mismatches = verify_frozen_models_unchanged(saved)
    assert unchanged is True, mismatches

    # Observation for the LATEST available date.
    topix = load_ohlcv("TOPIX", config.source_processed_dir)
    result = build_observation_entries(
        tickers, config, t0, [trained_raw.spec, trained_beta.spec], MARKET_REGIME_CONFIG, topix,
    )
    assert result.universe_size == len(tickers)
    assert len(result.entries) > 0
    model_ids_seen = {e.model_id for e in result.entries}
    assert model_ids_seen == {trained_raw.spec.model_id, trained_beta.spec.model_id}

    # Append-only + idempotency: logging the same observation twice must
    # not duplicate rows.
    log_path = tmp_path / "predictions_log.jsonl"
    written_first = append_prediction_entries(log_path, result.entries)
    written_second = append_prediction_entries(log_path, result.entries)
    assert written_first == len(result.entries)
    assert written_second == 0
    assert len(load_all_entries(log_path)) == written_first

    # Leakage: use an EARLIER date (not the very last one) as the
    # observation date so there is real future data to shock.
    all_dates = sorted(dataset["date"].unique())
    mid_date = all_dates[len(all_dates) // 2]
    regime_df = compute_market_regime(topix, MARKET_REGIME_CONFIG)
    assert mid_date in set(regime_df["date"])
    shock_results = run_observation_shock_checks(
        tickers, config, mid_date, tmp_path / "shock_work", [trained_raw.spec, trained_beta.spec],
        MARKET_REGIME_CONFIG, topix,
    )
    assert len(shock_results) == 3
    assert all(r.passed for r in shock_results), shock_results
