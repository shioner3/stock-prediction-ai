"""Phase V2-2 CLI entry point: Full Universe OOS Validation of the
Phase V2-1 Swing Candidate Ranking Engine's Score (spec section 33
STEP 1-19; STEPs 4/20 - Leakage tests / Full test suite - are run via
`pytest`, not embedded in this script, matching every earlier Phase's
convention in this project).

Usage:
    python scripts/run_v2_2_oos_validation.py

Runs the STOP-condition preflight (V2-1 hash check, Data Integrity)
BEFORE the expensive Full Universe computation (spec section 34) - a
preflight failure exits without running anything further.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.logging_setup import setup_logging  # noqa: E402
from config.loader import load_app_config  # noqa: E402
from v2.config.loader import load_v2_config  # noqa: E402
from v2.pipeline import load_universe_tickers  # noqa: E402
from v2.validation.data_integrity import (  # noqa: E402
    has_critical_integrity_issues,
    run_data_integrity_preflight,
)
from v2.validation.decision import V2DecisionInputs, classify_v2_decision  # noqa: E402
from v2.validation.hash_check import (  # noqa: E402
    current_v2_1_code_hash,
    current_v2_1_config_hash,
    verify_v2_1_unchanged,
)
from v2.validation.orchestrator import (  # noqa: E402,E501
    PRIMARY_WINDOW_DAYS,
    V2_2Report,
    run_v2_2_validation,
)

# Pinned at the start of Phase V2-2, computed from the actual committed
# Phase V2-1 state (branch add-v2-ranking-engine, commit df88c81) via
# `current_v2_1_code_hash()`/`current_v2_1_config_hash()` BEFORE any
# Phase V2-2 code was written - see research/phase_v2_2_report.md
# section "V1 Independence"/"V2-1仕様確認" for the verification record.
EXPECTED_V2_1_CODE_HASH = "30f62bd002d9326ec17320dceed8325ca6c0eadf239775cbf6d31371c2927925"
EXPECTED_V2_1_CONFIG_HASH = "8ed74d9a3d7436f4a9183ea855b00580a3c1371edce7d2fe0333867ec5287120"


def _json_default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
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
    print(f"  {label:<30} n={n:>8} Q5-Q1={spread}")


def _print_summary(report: V2_2Report) -> None:
    print(f"n_tickers: {report.n_tickers}")
    print(f"n_rows_total: {report.n_rows_total}")
    print(f"n_rows_scored: {report.n_rows_scored}")
    print(f"date_range: {report.date_range_start} .. {report.date_range_end}")

    print(f"\n--- Primary window ({report.primary.window_days}d) ---")
    print(f"  Q5-Q1 spread: {report.primary.light.q5_q1_spread}")
    print(f"  monotonic: {report.primary.light.monotonicity.is_monotonic_nondecreasing}")
    print(f"  Spearman: {report.primary.light.monotonicity.spearman}")
    print(f"  Kendall: {report.primary.light.monotonicity.kendall}")
    print(
        f"  IC mean={report.primary.light.ic_summary.mean_ic} "
        f"IR={report.primary.light.ic_summary.information_ratio} "
        f"positive_ratio={report.primary.light.ic_summary.positive_ic_ratio}"
    )

    tb = report.primary.trade_level_bootstrap
    print(f"\n  Trade-level bootstrap: {tb.point_estimate} [{tb.ci_low}, {tb.ci_high}]")
    dcb = report.primary.day_cluster_bootstrap
    print(f"  Day-cluster bootstrap: {dcb.point_estimate} [{dcb.ci_low}, {dcb.ci_high}]")
    bb = report.primary.block_bootstrap
    print(f"  Block bootstrap:       {bb.point_estimate} [{bb.ci_low}, {bb.ci_high}]")

    print("\n  Permutation:")
    for p in report.primary.permutation:
        print(f"    {p.bucket_label}: p={p.result.p_value}")

    print("\n  Cost sensitivity (Q5):")
    for cost_result in report.primary.cost_sensitivity:
        print(
            f"    {cost_result.tier_name} ({cost_result.round_trip_bps}bps): "
            f"net_mean={cost_result.net_stats.mean_return}"
        )

    print("\n  Event exclusion:")
    for label, s in report.primary.event_exclusion.items():
        _print_slice(label, s.n, s.q5_q1_spread)

    print("\n  Year-by-year:")
    for year, s in sorted(report.primary.year_by_year.items()):
        _print_slice(str(year), s.n, s.q5_q1_spread)

    print("\n  Regime:")
    for rs in report.primary.regime:
        _print_slice(rs.regime, rs.n, rs.q5_q1_spread)

    print("\n  Concentration (Q5 ticker contribution):")
    for share in report.primary.concentration.ticker_contribution:
        print(f"    top{share.top_k}: {share.share_of_total}")

    print(f"\n--- Holding Period comparison ({len(report.holding_period_comparison)} windows) ---")
    for window, lr in sorted(report.holding_period_comparison.items()):
        print(
            f"  {window:>3}d: spread={lr.q5_q1_spread} "
            f"mono={lr.monotonicity.is_monotonic_nondecreasing} "
            f"IC={lr.ic_summary.mean_ic}"
        )

    print(f"\n--- FDR ({len(report.fdr_results)} tests) ---")
    for key, fdr in sorted(report.fdr_results.items(), key=lambda kv: kv[1].raw_p_value):
        print(f"  {key:<28} raw_p={fdr.raw_p_value:.4f} adj_p={fdr.adjusted_p_value:.4f}")


def _build_decision_inputs(report: V2_2Report) -> V2DecisionInputs:
    primary = report.primary
    core_fdr_key = f"holding_period:{PRIMARY_WINDOW_DAYS}d:Q5"
    fdr_result = report.fdr_results.get(core_fdr_key)

    windows = list(report.holding_period_comparison.values())
    holding_positive = [w.q5_q1_spread for w in windows if w.q5_q1_spread is not None]
    holding_fraction = (
        sum(1 for s in holding_positive if s > 0) / len(holding_positive)
        if holding_positive else None
    )

    years = list(primary.year_by_year.values())
    year_positive = [y.q5_q1_spread for y in years if y.q5_q1_spread is not None]
    year_fraction = (
        sum(1 for s in year_positive if s > 0) / len(year_positive) if year_positive else None
    )

    survives_event = all(
        s.q5_q1_spread is not None and s.q5_q1_spread > 0
        for s in primary.event_exclusion.values()
    ) if primary.event_exclusion else None

    top5 = next((t for t in primary.light.topn_results if t.n == 5), None)
    topn_reproduces = (
        top5.stats.mean_return is not None and top5.stats.mean_return > 0 if top5 else None
    )

    low_cost = next((c for c in primary.cost_sensitivity if c.tier_name == "low"), None)
    survives_low_cost = (
        low_cost.net_stats.mean_return is not None and low_cost.net_stats.mean_return > 0
        if low_cost else None
    )

    regime_spreads = {rs.regime: rs.q5_q1_spread for rs in primary.regime}
    known_regime_spreads = [s for s in regime_spreads.values() if s is not None]
    regime_dependent = (
        not all(s > 0 for s in known_regime_spreads) if known_regime_spreads else None
    )

    return V2DecisionInputs(
        q5_q1_spread=primary.light.q5_q1_spread,
        block_bootstrap_ci_low=primary.block_bootstrap.ci_low,
        permutation_p_value=next(
            (p.result.p_value for p in primary.permutation if p.bucket_label == "Q5"), None
        ),
        fdr_significant=fdr_result.significant if fdr_result else None,
        holding_period_positive_fraction=holding_fraction,
        year_positive_fraction=year_fraction,
        survives_event_exclusion=survives_event,
        topn_reproduces=topn_reproduces,
        survives_low_cost=survives_low_cost,
        regime_dependent=regime_dependent,
        holding_period_dependent=(holding_fraction is not None and holding_fraction < 1.0),
        segment_dependent=None,  # spec section 14 "if possible" - not implemented this run
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase V2-2 Full Universe OOS Validation")
    parser.add_argument("--limit-tickers", type=int, default=None, help="dev/testing only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v1_config = load_app_config(Path("config/settings.yaml"))
    setup_logging(v1_config.logging.level, v1_config.logging.log_dir)
    v2_config = load_v2_config()

    # --- STEP 2: V2-1 hash check -----------------------------------------
    print("STEP 2: V2-1 hash check")
    code_hash = current_v2_1_code_hash()
    config_hash = current_v2_1_config_hash()
    unchanged, mismatches = verify_v2_1_unchanged(
        EXPECTED_V2_1_CODE_HASH, EXPECTED_V2_1_CONFIG_HASH
    )
    print(f"  code_hash={code_hash} config_hash={config_hash} unchanged={unchanged}")
    if not unchanged:
        print(f"STOP: Phase V2-1 hash mismatch: {mismatches}")
        sys.exit(1)

    # --- STEP 3: Data Integrity preflight --------------------------------
    print("\nSTEP 3: Data Integrity preflight")
    tickers = load_universe_tickers(v2_config)
    if args.limit_tickers:
        tickers = tickers[: args.limit_tickers]
    print(f"  Universe size: {len(tickers)}")
    summary = run_data_integrity_preflight(
        tickers, v2_config.source_processed_dir,
        datetime.date(2018, 1, 1), datetime.date.today(),
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

    # --- STEP 6-19: run the validation battery ---------------------------
    print("\nSTEP 6-19: running Full Universe validation...")
    report = run_v2_2_validation(v2_config, tickers=tickers)
    _print_summary(report)

    decision_inputs = _build_decision_inputs(report)
    decision = classify_v2_decision(decision_inputs)
    print(f"\n=== Decision: {decision.decision.value} ===")
    for reason in decision.reasons:
        print(f"  - {reason}")

    save_path = Path("data/v2/reports/v2_2_validation_report.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "report": dataclasses.asdict(report),
        "decision_inputs": dataclasses.asdict(decision_inputs),
        "decision": dataclasses.asdict(decision),
        "v2_1_code_hash": code_hash,
        "v2_1_config_hash": config_hash,
    }
    save_path.write_text(
        json.dumps(payload, default=_json_default, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nsaved: {save_path}")


if __name__ == "__main__":
    main()
