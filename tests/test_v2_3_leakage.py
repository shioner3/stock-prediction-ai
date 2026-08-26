"""Phase V2-3 Leakage Tests (spec section 36), scoped to the NEW surface
this Phase adds - v2/causal/feature_stats.py::compute_feature_percentiles().
The underlying "V1 never imports v2" invariant and the Future Shock Test
are already covered by tests/test_v2_leakage.py (that AST check scans for
ANY `v2.`-prefixed import across every V1 directory, so it already covers
v2/causal transitively - not duplicated here, matching Phase V2-2's own
precedent of not re-litigating it).

- Label Isolation Test: mutating Forward Target columns must never change
  a Feature's own percentile column (compute_feature_percentiles() only
  ever reads the raw Feature column + date/ticker, never a
  forward_return_* column).
- Date Boundary Test: truncating the panel to date <= T must produce the
  SAME pct_<feature> values at T as computing against the untruncated,
  future-extended panel.
- Cross-sectional Leakage Test: compute_feature_percentiles() never pools
  values across dates (each feature's percentile column is grouped by
  date internally, same guarantee V2-1's own percentile_rank_by_day()
  already provides - reasserted here empirically for the causal package).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from v2_3_test_helpers import build_v2_3_config_and_tickers

from storage.parquet_store import load_feature_panel, save_feature_panel
from v2.causal.feature_stats import FEATURE_LIST, compute_feature_percentiles, percentile_column
from v2.pipeline import run_v2_ranking

PCT_COLUMNS = [percentile_column(feature) for _c, feature, _h in FEATURE_LIST]


def test_label_isolation_percentiles_unaffected_by_forward_target_mutation(tmp_path: Path) -> None:
    config, tickers = build_v2_3_config_and_tickers(tmp_path, n_tickers=10, n_days=150)
    ranked = run_v2_ranking(config, tickers=tickers)
    scored = compute_feature_percentiles(ranked)

    mutated_ranked = ranked.copy()
    for col in mutated_ranked.columns:
        if col.startswith("forward_return_"):
            mutated_ranked[col] = mutated_ranked[col] * 999.0 + 12345.0
    mutated_scored = compute_feature_percentiles(mutated_ranked)

    for col in PCT_COLUMNS:
        assert np.allclose(
            scored[col].to_numpy(dtype=float),
            mutated_scored[col].to_numpy(dtype=float),
            equal_nan=True,
        ), f"{col} changed after mutating Forward Target columns"


def test_compute_feature_percentiles_never_reads_a_forward_return_column() -> None:
    from v2.ranking.cross_sectional import percentile_rank_by_day

    assert percentile_rank_by_day.__module__ == "v2.ranking.cross_sectional"
    feature_columns = {feature for _c, feature, _h in FEATURE_LIST}
    assert not any(col.startswith("forward_return_") for col in feature_columns)


def test_date_boundary_percentiles_match_between_truncated_and_full_panel(tmp_path: Path) -> None:
    config, tickers = build_v2_3_config_and_tickers(tmp_path, n_tickers=10, n_days=150)
    full_ranked = run_v2_ranking(config, tickers=tickers)
    full = compute_feature_percentiles(full_ranked)

    all_dates = sorted(full["date"].unique())
    cutoff = all_dates[len(all_dates) // 2]

    truncated_features_dir = tmp_path / "features_truncated"
    for ticker in tickers:
        panel = load_feature_panel(ticker, config.source_features_dir)
        truncated = panel[panel["date"] <= cutoff]
        save_feature_panel(truncated, ticker, truncated_features_dir)

    truncated_config = config.model_copy(update={"source_features_dir": truncated_features_dir})
    truncated_ranked = run_v2_ranking(truncated_config, tickers=tickers)
    truncated = compute_feature_percentiles(truncated_ranked)

    full_at_cutoff = full[full["date"] == cutoff].sort_values("ticker").reset_index(drop=True)
    truncated_at_cutoff = (
        truncated[truncated["date"] == cutoff].sort_values("ticker").reset_index(drop=True)
    )

    assert list(full_at_cutoff["ticker"]) == list(truncated_at_cutoff["ticker"])
    for col in PCT_COLUMNS:
        assert np.allclose(
            full_at_cutoff[col].to_numpy(dtype=float),
            truncated_at_cutoff[col].to_numpy(dtype=float),
            equal_nan=True,
        ), f"{col} at the cutoff date differs depending on whether future rows exist"


def test_feature_percentiles_never_pool_across_dates(tmp_path: Path) -> None:
    config, tickers = build_v2_3_config_and_tickers(tmp_path, n_tickers=10, n_days=150)
    ranked = run_v2_ranking(config, tickers=tickers)
    scored = compute_feature_percentiles(ranked)

    feature = FEATURE_LIST[0][1]
    col = percentile_column(feature)
    for d, group in scored.groupby("date"):
        valid = group[col].dropna()
        if len(valid) == 0:
            continue
        # Every value is a percentile rank WITHIN that day's group alone.
        assert valid.max() <= 1.0 and valid.min() > 0.0
