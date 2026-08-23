"""Phase V2-3 CLI entry point: Q1 Negative Predictive Signal - Causal
Structure & Robustness Analysis (spec section 39 STEP 1-23; STEP 5/22
Leakage tests / Full test suite are run via `pytest`, not embedded here,
matching Phase V2-2's own established convention).

Usage:
    python scripts/run_v2_3_causal_analysis.py

Runs every STOP-condition preflight (V2-1 hash, V2-2 reproducibility,
Data Integrity) BEFORE the expensive Full Universe computation (spec
section 40's pre-execution checklist) - a preflight failure exits without
running anything further. The V2-1 ranked panel is built exactly ONCE and
reused for both the reproducibility check and the causal analysis battery
(spec section 35).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from common.logging_setup import setup_logging  # noqa: E402
from config.loader import load_app_config  # noqa: E402
from v2.causal.decision import V2_3DecisionInputs, classify_v2_3_decision  # noqa: E402
from v2.causal.orchestrator import (  # noqa: E402
    PRIMARY_WINDOW_DAYS,
    V2_3Report,
    build_scored_panel,
    run_v2_3_causal_analysis_on_panel,
)
from v2.causal.reproducibility import verify_v2_2_reproducibility  # noqa: E402
from v2.config.loader import load_v2_config  # noqa: E402
from v2.pipeline import load_universe_tickers  # noqa: E402
from v2.validation.data_integrity import (  # noqa: E402
    has_critical_integrity_issues,
    run_data_integrity_preflight,
)
from v2.validation.hash_check import (  # noqa: E402
    current_v2_1_code_hash,
    current_v2_1_config_hash,
    verify_v2_1_unchanged,
)

# Same pinned baseline Phase V2-2 already established (branch
# add-v2-ranking-engine, commit df88c81) - V2-1 has not changed since.
EXPECTED_V2_1_CODE_HASH = "30f62bd002d9326ec17320dceed8325ca6c0eadf239775cbf6d31371c2927925"
EXPECTED_V2_1_CONFIG_HASH = "8ed74d9a3d7436f4a9183ea855b00580a3c1371edce7d2fe0333867ec5287120"


def _json_default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index().to_dict(orient="records")
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    if hasattr(obj, "value"):
        return obj.value
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    if isinstance(obj, float) and obj in (float("inf"), float("-inf")):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


def _print_slice(label: str, n: int, spread: float | None) -> None:
    print(f"  {label:<35} n={n:>8} Q5-Q1={spread}")


def _n_tracked_changes(git_status_porcelain: str) -> int:
    return len([line for line in git_status_porcelain.splitlines() if not line.startswith("??")])


def _print_summary(report: V2_3Report) -> None:
    print(f"n_tickers: {report.n_tickers}")
    print(f"n_rows_total: {report.n_rows_total}")
    print(f"n_rows_scored: {report.n_rows_scored}")
    print(f"date_range: {report.date_range_start} .. {report.date_range_end}")

    print("\n--- Category contribution to Q1 (most negative deviation first) ---")
    ranked_categories = sorted(
        report.category_contribution_q1,
        key=lambda c: (
            c.deviation_from_population if c.deviation_from_population is not None else 1e9
        ),
    )
    for c in ranked_categories:
        print(
            f"  {c.category:<20} mean_rank={c.mean_category_rank} "
            f"deviation={c.deviation_from_population}"
        )

    print("\n--- Feature direction classification (primary 5d) ---")
    for feature, direction in sorted(report.feature_directions.items(), key=lambda kv: kv[0]):
        print(f"  {feature:<28} {direction.value}")

    print("\n--- Q1 internal heterogeneity (5d) ---")
    for b in report.heterogeneity.sub_bucket_stats:
        print(f"  {b.bucket:<8} n={b.stats.n:>7} mean={b.stats.mean_return}")

    print("\n--- Regime (5d) ---")
    for rs in report.regime_results:
        q1 = next((b for b in rs.bucket_stats if b.bucket == "Q1"), None)
        q1_mean = q1.stats.mean_return if q1 else None
        print(f"  {rs.regime:<10} n={rs.n:>8} Q5-Q1={rs.q5_q1_spread} Q1_mean={q1_mean}")

    print("\n--- Year-by-year (5d) ---")
    for year, s in sorted(report.year_results.items()):
        _print_slice(str(year), s.n, s.q5_q1_spread)

    print(f"\n--- Holding Period comparison ({len(report.holding_period_results)} windows) ---")
    for window, (_bstats, spread) in sorted(report.holding_period_results.items()):
        print(f"  {window:>3}d: spread={spread}")

    print("\n--- Market Segment (Q1) ---")
    for g in report.segment_stats:
        print(f"  {g.group_value:<15} n={g.n:>7} mean={g.stats.mean_return}")

    print("\n--- Size / scale (Q1) ---")
    for g in report.scale_stats:
        print(f"  {g.group_value:<20} n={g.n:>7} mean={g.stats.mean_return}")

    print("\n--- Liquidity (Q1 vs Q5) ---")
    l1, l5 = report.liquidity_q1, report.liquidity_q5
    print(f"  Q1 turnover_mean={l1.turnover_mean} volume_mean={l1.volume_mean}")
    print(f"  Q5 turnover_mean={l5.turnover_mean} volume_mean={l5.volume_mean}")

    print("\n--- Concentration (Q1 ticker contribution) ---")
    for share in report.concentration_q1.ticker_contribution:
        print(f"  top{share.top_k}: {share.share_of_total}")

    print("\n--- Event exclusion (5d) ---")
    for label, s in report.event_exclusion.items():
        _print_slice(label, s.n, s.q5_q1_spread)

    print("\n--- Bootstrap (Q1 alone, 5d) ---")
    tb = report.q1_trade_level_bootstrap
    print(f"  Trade-level:  {tb.point_estimate} [{tb.ci_low}, {tb.ci_high}]")
    dcb = report.q1_day_cluster_bootstrap
    print(f"  Day-cluster:  {dcb.point_estimate} [{dcb.ci_low}, {dcb.ci_high}]")
    bb = report.q1_block_bootstrap
    print(f"  Block:        {bb.point_estimate} [{bb.ci_low}, {bb.ci_high}]")

    print("\n--- Bootstrap (Q5-Q1 spread, 5d) ---")
    sdc = report.spread_day_cluster_bootstrap
    print(f"  Day-cluster:  {sdc.point_estimate} [{sdc.ci_low}, {sdc.ci_high}]")
    sb = report.spread_block_bootstrap
    print(f"  Block:        {sb.point_estimate} [{sb.ci_low}, {sb.ci_high}]")

    print("\n--- Permutation (Q1 vs population) ---")
    for window, perm in sorted(report.permutation_by_window.items()):
        print(f"  {window:>3}d: p={perm.p_value}")

    print(f"\n--- FDR ({len(report.fdr_results)} tests, top 15 by raw p) ---")
    for key, fdr in sorted(report.fdr_results.items(), key=lambda kv: kv[1].raw_p_value)[:15]:
        print(f"  {key:<28} raw_p={fdr.raw_p_value:.4f} adj_p={fdr.adjusted_p_value:.4f}")

    print("\n--- Timing Placebo (5d) ---")
    for lag_result in sorted(report.placebo_results, key=lambda r: r.lag_days):
        print(
            f"  lag={lag_result.lag_days:>4}d n={lag_result.n:>7} "
            f"mean={lag_result.stats.mean_return}"
        )

    print("\n--- Cross-sectional stability (5d) ---")
    st = report.stability
    print(
        f"  n_days={st.n_days} median={st.median_daily_return} mean={st.mean_daily_return} "
        f"positive_ratio={st.positive_day_ratio} gini={st.gini_of_below_average_days}"
    )
    print(
        f"  worst_day={st.worst_day} ({st.worst_day_return}) "
        f"best_day={st.best_day} ({st.best_day_return})"
    )

    print("\n--- Random control (5d) ---")
    for seed_result in report.random_control.per_seed:
        print(f"  seed={seed_result.seed} n={seed_result.n} mean={seed_result.stats.mean_return}")
    print(f"  pooled: mean={report.random_control.pooled_stats.mean_return}")


def _build_decision_inputs(report: V2_3Report) -> V2_3DecisionInputs:
    primary_spread = report.holding_period_results[PRIMARY_WINDOW_DAYS][1]

    holding_spreads = [
        s for _w, (_b, s) in report.holding_period_results.items() if s is not None
    ]
    holding_negative_fraction = (
        sum(1 for s in holding_spreads if s < 0) / len(holding_spreads)
        if holding_spreads
        else None
    )

    year_spreads = [
        s.q5_q1_spread for s in report.year_results.values() if s.q5_q1_spread is not None
    ]
    year_negative_fraction = (
        sum(1 for s in year_spreads if s < 0) / len(year_spreads) if year_spreads else None
    )

    regime_spreads = [
        rs.q5_q1_spread for rs in report.regime_results if rs.q5_q1_spread is not None
    ]
    regime_negative_fraction = (
        sum(1 for s in regime_spreads if s < 0) / len(regime_spreads)
        if regime_spreads
        else None
    )

    event_spreads = [
        s.q5_q1_spread
        for label, s in report.event_exclusion.items()
        if label != "full_period" and s.q5_q1_spread is not None
    ]
    survives_event_exclusion = all(s < 0 for s in event_spreads) if event_spreads else None

    fdr_key = f"holding_period:{PRIMARY_WINDOW_DAYS}d:Q1"
    fdr_result = report.fdr_results.get(fdr_key)

    return V2_3DecisionInputs(
        primary_q5_q1_spread=primary_spread,
        day_cluster_spread_ci_high=report.spread_day_cluster_bootstrap.ci_high,
        block_spread_ci_high=report.spread_block_bootstrap.ci_high,
        q1_permutation_p_value=report.permutation_by_window[PRIMARY_WINDOW_DAYS].p_value,
        fdr_significant=fdr_result.significant if fdr_result else None,
        holding_period_negative_fraction=holding_negative_fraction,
        year_negative_fraction=year_negative_fraction,
        regime_negative_fraction=regime_negative_fraction,
        survives_event_exclusion=survives_event_exclusion,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase V2-3 Causal Structure Analysis")
    parser.add_argument("--limit-tickers", type=int, default=None, help="dev/testing only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v1_config = load_app_config(Path("config/settings.yaml"))
    setup_logging(v1_config.logging.level, v1_config.logging.log_dir)
    v2_config = load_v2_config()

    # --- STEP 1: Repository confirm ---------------------------------------
    print("STEP 1: Repository confirm")
    branch = subprocess.run(
        ["git", "branch", "--show-current"], capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout
    print(f"  branch={branch}")
    print(f"  tracked changes: {_n_tracked_changes(status)}")

    # --- STEP 2: V2-1 Hash check -------------------------------------------
    print("\nSTEP 2: V2-1 hash check")
    code_hash = current_v2_1_code_hash()
    config_hash = current_v2_1_config_hash()
    unchanged, mismatches = verify_v2_1_unchanged(
        EXPECTED_V2_1_CODE_HASH, EXPECTED_V2_1_CONFIG_HASH
    )
    print(f"  code_hash={code_hash} config_hash={config_hash} unchanged={unchanged}")
    if not unchanged:
        print(f"STOP: Phase V2-1 hash mismatch: {mismatches}")
        sys.exit(1)

    tickers = load_universe_tickers(v2_config)
    if args.limit_tickers:
        tickers = tickers[: args.limit_tickers]

    print(
        f"\nBuilding V2-1 ranked panel + Feature percentiles for {len(tickers)} tickers "
        "(feeds STEP 3/6-21)..."
    )
    ranked, scored = build_scored_panel(v2_config, tickers=tickers)

    # --- STEP 3: V2-2 result reproduction check ----------------------------
    print("\nSTEP 3: V2-2 reproducibility check")
    repro = verify_v2_2_reproducibility(ranked, scored)
    print(
        f"  passed={repro.passed} recomputed_spread={repro.recomputed_spread} "
        f"saved_spread={repro.saved_spread}"
    )
    if not repro.passed:
        print(f"STOP: V2-2 results did not reproduce: {repro.mismatches}")
        sys.exit(1)

    # --- STEP 4: Data Integrity preflight -----------------------------------
    print("\nSTEP 4: Data Integrity preflight")
    summary = run_data_integrity_preflight(
        tickers, v2_config.source_processed_dir, datetime.date(2018, 1, 1), datetime.date.today()
    )
    print(
        f"  checked={summary.n_tickers_checked} missing={len(summary.n_tickers_missing)} "
        f"duplicate_dates={summary.total_duplicate_dates} "
        f"invalid_ohlc={summary.total_invalid_ohlc_rows} "
        f"negative_volume={summary.total_negative_volume_rows} "
        f"nan_rows={summary.total_nan_rows}"
    )
    if has_critical_integrity_issues(summary):
        print("STOP: critical Data Integrity issue detected")
        sys.exit(1)

    # --- STEP 5: Leakage tests (verified via pytest, not embedded here) ---
    print("\nSTEP 5: Leakage tests -> see `pytest tests/test_v2_3_leakage.py`")

    # --- STEP 40: pre-execution checklist -----------------------------------
    print("\n=== STEP 40: Pre-execution checklist ===")
    print(f"  V2-1 Hash: unchanged={unchanged}")
    print(f"  V2-2 reproducibility: passed={repro.passed}")
    print(f"  Git tracked changes: {_n_tracked_changes(status)}")
    print(f"  Data period: {ranked['date'].min()} .. {ranked['date'].max()}")
    print(f"  Universe count: {ranked['ticker'].nunique()}")
    w = v2_config.score_weights
    print(
        f"  Score weights: momentum={w.momentum} trend={w.trend} volume={w.volume} "
        f"relative_strength={w.relative_strength} pullback={w.pullback} volatility={w.volatility}"
    )
    print(
        "  Target definitions: forward_return_{1,3,5,7,10,15,20}d "
        "(targets.forward_returns, unchanged)"
    )
    print("  Data Integrity: OK (see STEP 4)")
    print("All preflight checks passed - proceeding to Full Universe causal analysis.\n")

    # --- STEP 6-21: causal analysis battery ---------------------------------
    print("STEP 6-21: running Full Universe causal analysis...")
    report = run_v2_3_causal_analysis_on_panel(v2_config, ranked, scored)
    _print_summary(report)

    decision_inputs = _build_decision_inputs(report)
    decision = classify_v2_3_decision(decision_inputs)
    print(f"\n=== Decision: {decision.decision.value} ===")
    for reason in decision.reasons:
        print(f"  - {reason}")

    save_path = Path("data/v2/reports/v2_3_causal_analysis_report.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "report": dataclasses.asdict(report),
        "decision_inputs": dataclasses.asdict(decision_inputs),
        "decision": dataclasses.asdict(decision),
        "v2_1_code_hash": code_hash,
        "v2_1_config_hash": config_hash,
        "v2_2_reproducibility": dataclasses.asdict(repro),
    }
    save_path.write_text(
        json.dumps(payload, default=_json_default, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nsaved: {save_path}")


if __name__ == "__main__":
    main()
