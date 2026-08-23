"""Phase V3-4 CLI entry point: Market Timing vs Stock-Specific Edge
Decomposition on Phase V3-3's FROZEN pipeline (spec section 26 - the goal
is deciding WHERE V3-3's +0.383% Q5-Q1 spread comes from, not improving
it; runs to completion, then stops - never proceeds to hyperparameter
tuning, V1 integration, or a UI).

Usage:
    python scripts/run_v3_4_robustness.py [--limit-tickers N]
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
from config.loader import TransactionCostConfig, load_app_config  # noqa: E402
from storage.parquet_store import load_ohlcv  # noqa: E402
from v3.config.loader import load_v3_config  # noqa: E402
from v3.dataset import load_universe_tickers  # noqa: E402
from v3.leakage.availability_check import check_v3_features_no_forward_reads  # noqa: E402
from v3.robustness.orchestrator import V3_4Report, run_v3_4_analysis  # noqa: E402
from v3.robustness.reproduce import (  # noqa: E402
    build_frozen_dataset,
    load_v3_3_reference_hashes,
    reproduce_predictions,
    save_predictions,
    verify_against_v3_3,
)
from v3.robustness.v3_3_reference import (  # noqa: E402
    PRIMARY_Q5_Q1_SPREAD_REFERENCE,
    check_primary_spread_reproduction,
)
from v3.validation.leakage_check import run_full_universe_shock_checks  # noqa: E402
from v3.validation.ranking_metrics import evaluate_ranking  # noqa: E402
from v3.validation.wfo_config import MARKET_REGIME_CONFIG, PRIMARY_TARGET_COL  # noqa: E402
from v3.validation.windows import get_v3_3_windows  # noqa: E402

LEAKAGE_SHOCK_CUTOFF = datetime.date(2024, 6, 1)


def _json_default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index().to_dict(orient="records")
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    if hasattr(obj, "value"):
        return obj.value
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    if isinstance(obj, float) and obj in (float("inf"), float("-inf")):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


def _print_summary(report: V3_4Report) -> None:
    print(f"\nprimary Q5-Q1 spread (reproduced): {report.primary_ranking.q5_q1_spread}")
    print(f"primary Rank IC (reproduced): {report.primary_ranking.ic_summary.mean_ic}")

    print("\n--- Market Timing vs Stock Selection (section 3) ---")
    for variant, result in report.market_decomposition.items():
        print(
            f"  {variant:<18} spread={result.q5_q1_spread} q5={result.q5_mean} q1={result.q1_mean} "
            f"IC={result.ic_summary.mean_ic}"
        )

    print("\n--- Cross-sectional structural invariance (section 4/5) ---")
    inv = report.structural_invariance
    print(f"  rank_ic_identical={inv.rank_ic_identical} top5={inv.top5_selection_identical} "
          f"top10={inv.top10_selection_identical} top20={inv.top20_selection_identical}")
    print(f"  spread original={inv.q5_q1_spread_original} demeaned={inv.q5_q1_spread_demeaned} "
          f"demedianed={inv.q5_q1_spread_demedianed}")
    print(f"  daily market-component correlation: {report.market_component_correlation}")

    print("\n--- Regime robustness (section 6/9) ---")
    for label, s in report.regime_robustness.breakdown.items():
        print(f"  {label}: n={s.n} spread={s.ranking.q5_q1_spread}")
    for label, s in report.regime_robustness.leave_one_out.items():
        print(f"  {label}: n={s.n} spread={s.ranking.q5_q1_spread}")
    print(f"  regime_dependent={report.regime_robustness.regime_dependent}")

    print("\n--- Day concentration (section 7) ---")
    print(f"  gini={report.day_concentration.gini_day_contribution} "
          f"top1%={report.day_concentration.top_1pct_n_days} days")
    for label, s in report.day_concentration.top_k_exclusion.items():
        print(f"  excl_{label}: n={s.n} spread={s.ranking.q5_q1_spread}")

    print("\n--- Year robustness (section 8) ---")
    for year, s in report.year_robustness.leave_one_out.items():
        print(f"  excl_{year}: n={s.n} spread={s.ranking.q5_q1_spread}")

    print("\n--- Stock concentration (section 10) ---")
    print(f"  gini={report.stock_concentration.gini_ticker_contribution}")
    for label, s in report.stock_concentration.top_k_exclusion.items():
        print(f"  excl_{label}: n={s.n} spread={s.ranking.q5_q1_spread}")

    print("\n--- Sector concentration (section 11) ---")
    print(f"  gini={report.sector_concentration.gini_sector_contribution}")
    for share in report.sector_concentration.sector_shares[:10]:
        print(f"  {share.sector33}: n={share.n} pnl_share={share.pnl_share}")
    llo = report.sector_concentration.leave_largest_sector_out
    print(f"  leave_largest_out: n={llo.n} spread={llo.ranking.q5_q1_spread}")

    print("\n--- Matched control (section 12) ---")
    mc = report.matched_control
    print(f"  n_q5_rows={mc.n_q5_rows} n_matched={mc.n_matched} n_unmatched={mc.n_unmatched}")
    print(f"  tiers={mc.match_tier_counts}")
    for col, comp in mc.comparisons.items():
        print(
            f"  {col}: treatment_mean={comp.treatment_stats.mean_return} "
            f"control_mean={comp.control_stats.mean_return} "
            f"diff_ci=[{comp.diff_bootstrap.ci_low}, {comp.diff_bootstrap.ci_high}]"
        )

    print("\n--- Holding period (section 15) ---")
    for horizon, ranking in sorted(report.holding_period.items()):
        print(f"  {horizon}d: spread={ranking.q5_q1_spread} IC={ranking.ic_summary.mean_ic}")

    print("\n--- Cost sensitivity (section 16) ---")
    for name, r in report.cost_sensitivity.items():
        print(f"  {name} ({r.round_trip_bps}bps): spread={r.evaluation.ranking.q5_q1_spread}")

    print("\n--- Economic significance (section 17) ---")
    for n, econ in sorted(report.economic_significance.items()):
        print(
            f"  top{n}: expectancy={econ.expectancy} win_rate={econ.win_rate} "
            f"pf={econ.profit_factor} maxdd={econ.max_drawdown} "
            f"max_losing_streak={econ.max_losing_streak_days} "
            f"annualized={econ.annualized_return_pct} sharpe={econ.sharpe}"
        )

    print(f"\n--- Variant FDR ({len(report.variant_fdr_results)} tests, section 13) ---")
    for key, fdr in sorted(report.variant_fdr_results.items(), key=lambda kv: kv[1].raw_p_value):
        print(f"  {key:<24} raw_p={fdr.raw_p_value:.4f} adj_p={fdr.adjusted_p_value:.4f}")

    print(f"\n=== V3-3 Decision (reapplied, section 18): {report.v3_3_decision.value} ===")
    for reason in report.v3_3_decision_reasons:
        print(f"  - {reason}")

    print(
        f"\n=== Edge Classification (section 19): "
        f"{report.edge_classification.classification.value} ==="
    )
    for reason in report.edge_classification.reasons:
        print(f"  - {reason}")
    ec = report.edge_classification
    print(
        f"  orig_positive={ec.orig_positive} beta_survives={ec.beta_survives} "
        f"topix_rel_survives={ec.topix_rel_survives} bear_excl_survives={ec.bear_excl_survives} "
        f"day_top20_survives={ec.day_top20_survives}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase V3-4 Robustness/Decomposition analysis")
    parser.add_argument("--limit-tickers", type=int, default=None, help="dev/testing only")
    parser.add_argument(
        "--skip-leakage-shock", action="store_true",
        help="dev/testing only - skip the slow shock check",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v1_config = load_app_config(Path("config/settings.yaml"))
    setup_logging(v1_config.logging.level, v1_config.logging.log_dir)
    v3_config = load_v3_config()

    print("STEP 1: Repository confirm")
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout
    allowed_modified_roots = {".gitignore", "README.md", "pyproject.toml"}
    allowed_modified_prefixes = (
        "v3/", "tests/test_v3", "scripts/train_v3", "scripts/build_v3", "scripts/run_v3",
        "research/phase_v3_4_report.md",
    )
    tracked_changes = [
        line
        for line in status.splitlines()
        if not line.startswith("??")
        and line[3:].strip() not in allowed_modified_roots
        and not line[3:].strip().startswith(allowed_modified_prefixes)
    ]
    print(f"  tracked changes to V1/V2 files (should be 0): {len(tracked_changes)}")
    if tracked_changes:
        print("STOP: tracked V1/V2 files were modified")
        for line in tracked_changes:
            print(f"  {line}")
        sys.exit(1)

    print("\nSTEP 2: Mechanical Feature leakage check (v3/features/*.py, AST scan)")
    findings = check_v3_features_no_forward_reads()
    print(f"  findings: {len(findings)}")
    if findings:
        for f in findings:
            print(f"  LEAKAGE_FOUND: {f.file}:{f.line} - {f.reason}")
        sys.exit(1)

    print("\nSTEP 3: building Full Universe dataset (identical to V3-3's STEP 3)")
    dataset = build_frozen_dataset(v3_config, limit_tickers=args.limit_tickers)
    tickers = load_universe_tickers(v3_config)
    if args.limit_tickers:
        tickers = tickers[: args.limit_tickers]
    print(f"  dataset rows: {len(dataset)}, columns: {len(dataset.columns)}")
    print(f"  date range: {dataset['date'].min()} .. {dataset['date'].max()}")

    print("\nSTEP 3b: verifying hash match against Phase V3-3's own saved report")
    reference = load_v3_3_reference_hashes()
    verification = verify_against_v3_3(dataset, reference)
    print(f"  code_hash_match={verification.code_hash_match} (informational only - see")
    print("    HashVerification's docstring: code_hash covers this Phase's OWN new")
    print("    v3/robustness/ package too, so it is expected NOT to match V3-3's)")
    print(f"  config_hash_match={verification.config_hash_match}")
    print(f"  feature_hash_match={verification.feature_hash_match}")
    print(f"  dataset_hash_match={verification.dataset_hash_match}")
    if not verification.all_match:
        print("STOP: V3-3 hash mismatch - frozen spec has drifted, per spec section 1/25")
        print(f"  current:   {verification.current}")
        print(f"  reference: {verification.reference}")
        sys.exit(1)

    windows = get_v3_3_windows()
    print(f"\nSTEP 4: WFO windows ({len(windows)}, identical to V3-3)")
    if len(windows) < 3:
        print("STOP: fewer than 3 OOS windows - WFO structure does not hold")
        sys.exit(1)

    if not args.skip_leakage_shock:
        print("\nSTEP 5: Full Universe Leakage re-verification (4 shock types, reused from V3-3)")
        shock_results = run_full_universe_shock_checks(
            tickers, v3_config, LEAKAGE_SHOCK_CUTOFF, Path("data/v3/tmp_shock_check_v3_4"), dataset
        )
        for r in shock_results:
            print(
                f"  {r.label}: compared={r.n_rows_compared} mismatches={r.n_mismatches} "
                f"passed={r.passed}"
            )
        if not all(r.passed for r in shock_results):
            print("STOP: LEAKAGE_FOUND in Full Universe shock check")
            sys.exit(1)
    else:
        print("\nSTEP 5: Full Universe Leakage re-verification SKIPPED (--skip-leakage-shock)")

    print(
        "\nSTEP 6: reproducing frozen Model A predictions "
        "(5d/10d/15d/20d raw) - this will take a while..."
    )
    reproduced = reproduce_predictions(dataset, windows)
    for target_col, df in reproduced.items():
        print(f"  {target_col}: {len(df)} pooled OOS rows")
    save_predictions(reproduced, Path("data/v3/robustness/predictions"))

    print("\nSTEP 6b: verifying reproduced Primary spread matches V3-3's own recorded value")
    primary_ranking = evaluate_ranking(reproduced[PRIMARY_TARGET_COL], 5)
    print(f"  reproduced primary Q5-Q1 spread: {primary_ranking.q5_q1_spread}")
    print(f"  V3-3 recorded value: {PRIMARY_Q5_Q1_SPREAD_REFERENCE}")
    if not check_primary_spread_reproduction(primary_ranking.q5_q1_spread):
        print("STOP: reproduced Primary spread does not match V3-3's recorded value - the")
        print("      substitute for spec section 1's model_hash reproducibility check failed")
        sys.exit(1)

    topix = load_ohlcv("TOPIX", v3_config.source_processed_dir)
    cost_tiers = TransactionCostConfig().tiers

    print("\nSTEP 7: running the decomposition/robustness battery...")
    report = run_v3_4_analysis(
        dataset=dataset, reproduced_predictions=reproduced, tickers=tickers, topix_ohlcv=topix,
        market_regime_config=MARKET_REGIME_CONFIG, cost_tiers=cost_tiers, v3_config=v3_config,
    )
    _print_summary(report)

    save_path = Path("data/v3/reports/v3_4_robustness_report.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "report": dataclasses.asdict(report),
        "hash_verification": dataclasses.asdict(verification),
    }
    save_path.write_text(
        json.dumps(payload, default=_json_default, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nsaved: {save_path}")
    print("\nPhase V3-4 complete — stopped after Robustness/Decomposition analysis.")


if __name__ == "__main__":
    main()
