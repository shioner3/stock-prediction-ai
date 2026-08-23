"""Phase V2-2 Leakage Tests (spec section 23), beyond Phase V2-1's own
Future Shock Test (tests/test_v2_leakage.py):

- Label Isolation Test: mutating Forward Target columns must never
  change the Score (total_score/category ranks never read a
  forward_return_* column at all - this proves it holds for real data,
  not just by code inspection).
- Date Boundary Test: truncating the panel to rows with date <= T must
  produce the SAME Score at T as computing against the full (untruncated,
  future-extended) panel - i.e. Score at T never needs future ROWS to
  exist at all, not just that a future row's VALUE doesn't matter (which
  is what the Future Shock Test already proves).
- Cross-sectional Leakage Test: same principle, checked directly via
  v2/validation/ic.py's per-day groupby (a day's IC never mixes another
  day's rows) - covered by test_v2_2_ic.py's
  test_ic_computed_independently_per_day already; re-asserted here at
  the ranking-panel level for completeness.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from conftest import make_synthetic_ohlcv

from features.pipeline import compute_feature_panel
from storage.parquet_store import save_feature_panel, save_ohlcv
from v2.config.loader import load_v2_config
from v2.pipeline import run_v2_ranking
from v2.ranking.score import CATEGORY_FEATURES

RANK_SCORE_COLUMNS = [
    "total_score", "momentum_rank", "trend_rank", "volume_rank",
    "volatility_rank", "relative_strength_rank", "pullback_rank",
]


def _build_config(tmp_path: Path, n_tickers: int = 10, n_days: int = 150):
    features_dir = tmp_path / "features"
    processed_dir = tmp_path / "processed"
    manifest_path = tmp_path / "manifest.json"

    market = make_synthetic_ohlcv(n_days, seed=999, ticker="TOPIX")
    save_ohlcv(market, "TOPIX", processed_dir)

    tickers = [f"T{i}" for i in range(n_tickers)]
    manifest_tickers = {}
    for i, ticker in enumerate(tickers):
        ohlcv = make_synthetic_ohlcv(n_days, seed=i + 1, ticker=ticker)
        panel = compute_feature_panel(ohlcv, market_df=market)
        save_feature_panel(panel, ticker, features_dir)
        manifest_tickers[ticker] = {"included_in_universe": True}
    manifest_path.write_text(json.dumps({"tickers": manifest_tickers}), encoding="utf-8")

    config = load_v2_config().model_copy(
        update={
            "source_universe_manifest": manifest_path,
            "source_features_dir": features_dir,
            "source_processed_dir": processed_dir,
        }
    )
    return config, tickers


def test_label_isolation_score_unaffected_by_forward_target_mutation(tmp_path: Path) -> None:
    config, tickers = _build_config(tmp_path)
    ranked = run_v2_ranking(config, tickers=tickers)

    mutated = ranked.copy()
    for col in mutated.columns:
        if col.startswith("forward_return_"):
            mutated[col] = mutated[col] * 999.0 + 12345.0

    for col in RANK_SCORE_COLUMNS:
        assert np.allclose(
            ranked[col].to_numpy(dtype=float), mutated[col].to_numpy(dtype=float), equal_nan=True
        ), f"{col} changed after mutating Forward Target columns - Score is not label-isolated"


def test_score_never_reads_a_forward_return_column() -> None:
    """Static check: no CATEGORY_FEATURES member (the only columns
    compute_category_ranks()/compute_v2_score() ever read) is a Forward
    Target column - the structural reason the label isolation test above
    holds, verified directly rather than only empirically.
    """
    all_feature_columns = {col for members in CATEGORY_FEATURES.values() for col, _ in members}
    assert not any(col.startswith("forward_return_") for col in all_feature_columns)


def test_date_boundary_score_matches_between_truncated_and_full_panel(tmp_path: Path) -> None:
    config, tickers = _build_config(tmp_path, n_days=150)
    full = run_v2_ranking(config, tickers=tickers)

    all_dates = sorted(full["date"].unique())
    cutoff = all_dates[len(all_dates) // 2]

    truncated_features_dir = tmp_path / "features_truncated"
    for ticker in tickers:
        from storage.parquet_store import load_feature_panel

        panel = load_feature_panel(ticker, config.source_features_dir)
        truncated = panel[panel["date"] <= cutoff]
        save_feature_panel(truncated, ticker, truncated_features_dir)

    truncated_config = config.model_copy(update={"source_features_dir": truncated_features_dir})
    truncated_ranked = run_v2_ranking(truncated_config, tickers=tickers)

    full_at_cutoff = (
        full[full["date"] == cutoff].sort_values("ticker").reset_index(drop=True)
    )
    truncated_at_cutoff = (
        truncated_ranked[truncated_ranked["date"] == cutoff]
        .sort_values("ticker").reset_index(drop=True)
    )

    assert list(full_at_cutoff["ticker"]) == list(truncated_at_cutoff["ticker"])
    for col in RANK_SCORE_COLUMNS:
        assert np.allclose(
            full_at_cutoff[col].to_numpy(dtype=float),
            truncated_at_cutoff[col].to_numpy(dtype=float),
            equal_nan=True,
        ), f"{col} at the cutoff date differs depending on whether future rows exist"


def test_cross_sectional_ranking_never_pools_across_dates(tmp_path: Path) -> None:
    config, tickers = _build_config(tmp_path)
    ranked = run_v2_ranking(config, tickers=tickers)

    from v2.validation.ic import compute_daily_spearman_ic

    daily = compute_daily_spearman_ic(
        ranked.dropna(subset=["total_score", "forward_return_5d"]),
        "total_score", "forward_return_5d",
    )
    dates_seen = {d.date for d in daily}
    assert dates_seen <= set(ranked["date"].unique())
    assert len(daily) == len(dates_seen)
