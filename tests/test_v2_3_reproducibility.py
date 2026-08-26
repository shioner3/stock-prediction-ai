from __future__ import annotations

import json
from pathlib import Path

from v2_3_test_helpers import build_scored_panel_for_tests

from v2.causal.reproducibility import verify_v2_2_reproducibility
from v2.stats import (
    compute_q5_q1_spread,
    compute_quantile_bucket_stats,
    exclude_implausible_returns,
)
from v2.validation.ic import compute_daily_spearman_ic, summarize_ic
from v2.validation.monotonicity import compute_monotonicity


def _save_matching_json(path: Path, ranked, scored) -> None:
    return_col = "forward_return_5d"
    clean = exclude_implausible_returns(scored.dropna(subset=[return_col]), return_col)
    bucket_stats = compute_quantile_bucket_stats(clean, "score_bucket", return_col, 5)
    spread = compute_q5_q1_spread(bucket_stats)
    bucket_means = {b.bucket: b.stats.mean_return for b in bucket_stats}
    mono = compute_monotonicity(bucket_means, 5)
    daily_ic = compute_daily_spearman_ic(clean, "total_score", return_col)
    ic_summary = summarize_ic(daily_ic, 5)

    payload = {
        "report": {
            "n_tickers": int(ranked["ticker"].nunique()),
            "primary": {
                "light": {
                    "q5_q1_spread": spread,
                    "bucket_stats": [
                        {
                            "bucket": b.bucket,
                            "stats": {"n": b.stats.n, "mean_return": b.stats.mean_return},
                        }
                        for b in bucket_stats
                    ],
                    "monotonicity": {"spearman": mono.spearman},
                    "ic_summary": {"mean_ic": ic_summary.mean_ic},
                }
            },
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_reproducibility_passes_against_matching_saved_report(tmp_path: Path) -> None:
    _config, _tickers, ranked, scored = build_scored_panel_for_tests(
        tmp_path / "data", n_tickers=15, n_days=300
    )
    saved_path = tmp_path / "saved_v2_2.json"
    _save_matching_json(saved_path, ranked, scored)

    result = verify_v2_2_reproducibility(ranked, scored, saved_report_path=saved_path)
    assert result.passed
    assert result.mismatches == []


def test_reproducibility_fails_against_perturbed_saved_report(tmp_path: Path) -> None:
    _config, _tickers, ranked, scored = build_scored_panel_for_tests(
        tmp_path / "data", n_tickers=15, n_days=300
    )
    saved_path = tmp_path / "saved_v2_2.json"
    _save_matching_json(saved_path, ranked, scored)

    payload = json.loads(saved_path.read_text(encoding="utf-8"))
    payload["report"]["primary"]["light"]["q5_q1_spread"] += 0.5  # deliberately wrong
    saved_path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_v2_2_reproducibility(ranked, scored, saved_report_path=saved_path)
    assert not result.passed
    assert any("q5_q1_spread mismatch" in m for m in result.mismatches)
