from __future__ import annotations

from pathlib import Path

from v2_3_test_helpers import build_v2_3_config_and_tickers

from targets.forward_returns import FORWARD_WINDOWS
from v2.causal.orchestrator import (
    PRIMARY_WINDOW_DAYS,
    build_scored_panel,
    run_v2_3_causal_analysis_on_panel,
)


def test_orchestrator_end_to_end_structure(tmp_path: Path) -> None:
    config, tickers = build_v2_3_config_and_tickers(tmp_path, n_tickers=20, n_days=300)
    ranked, scored = build_scored_panel(config, tickers=tickers)
    report = run_v2_3_causal_analysis_on_panel(config, ranked, scored)

    assert report.n_tickers == len(tickers)
    assert report.n_rows_scored <= report.n_rows_total

    assert set(report.holding_period_results.keys()) == set(FORWARD_WINDOWS)
    assert set(report.single_feature_by_window.keys()) == set(FORWARD_WINDOWS)
    assert set(report.permutation_by_window.keys()) == set(FORWARD_WINDOWS)

    assert len(report.heterogeneity.sub_bucket_stats) <= 5
    assert len(report.regime_results) == 3
    assert "full_period" in report.event_exclusion
    assert len(report.concentration_q1.ticker_contribution) == 4
    assert len(report.random_control.per_seed) == 5

    for fdr in report.fdr_results.values():
        assert 0.0 <= fdr.adjusted_p_value <= 1.0
        assert 0.0 <= fdr.raw_p_value <= 1.0


def test_orchestrator_is_deterministic(tmp_path: Path) -> None:
    config, tickers = build_v2_3_config_and_tickers(tmp_path, n_tickers=12, n_days=260)
    ranked, scored = build_scored_panel(config, tickers=tickers)

    r1 = run_v2_3_causal_analysis_on_panel(config, ranked, scored)
    r2 = run_v2_3_causal_analysis_on_panel(config, ranked, scored)

    spread1 = r1.holding_period_results[PRIMARY_WINDOW_DAYS][1]
    spread2 = r2.holding_period_results[PRIMARY_WINDOW_DAYS][1]
    assert spread1 == spread2
    assert r1.q1_day_cluster_bootstrap == r2.q1_day_cluster_bootstrap
    assert r1.q1_block_bootstrap == r2.q1_block_bootstrap
    assert [p.p_value for p in r1.permutation_by_window.values()] == [
        p.p_value for p in r2.permutation_by_window.values()
    ]
    assert r1.fdr_results.keys() == r2.fdr_results.keys()


def test_orchestrator_never_rebuilds_panel_internally(tmp_path: Path, monkeypatch) -> None:
    """spec section 35's efficiency requirement, mirrored from Phase
    V2-2's own equivalent test: run_v2_ranking (the expensive Panel/Score
    build) must be called exactly ONCE per build_scored_panel() call, and
    run_v2_3_causal_analysis_on_panel() must never call it again.
    """
    config, tickers = build_v2_3_config_and_tickers(tmp_path, n_tickers=10, n_days=220)

    import v2.causal.orchestrator as orch_module

    call_count = {"n": 0}
    original = orch_module.run_v2_ranking

    def counting_wrapper(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(orch_module, "run_v2_ranking", counting_wrapper)
    ranked, scored = build_scored_panel(config, tickers=tickers)
    assert call_count["n"] == 1

    run_v2_3_causal_analysis_on_panel(config, ranked, scored)
    assert call_count["n"] == 1
