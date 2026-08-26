"""Phase V3-5 Leakage tests (spec section 28):

- V1/V2/V3-1..V3-4 isolation: `v3/residual/*.py` never imports V1's
  decision-making modules (same corrected blocklist `tests/test_v3_3_
  leakage.py`/`tests/test_v3_4_leakage.py` established).
- Beta/residual-target shock-check smoke test: `v3/residual/leakage_
  check.py::run_residual_shock_checks()` runs end to end against a real
  (small synthetic) ticker set and reports all 4 shock types PASS -
  proving `compute_rolling_beta()`'s trailing window and the residual
  Targets' market_forward/sector-mean terms never read a future row.
"""

from __future__ import annotations

import ast
import datetime
from pathlib import Path

import pandas as pd
from v3_3_test_helpers import build_v3_3_config_and_tickers

from v3.dataset import build_v3_dataset
from v3.residual.leakage_check import run_residual_shock_checks
from v3.residual.targets import compute_residual_targets
from v3.robustness.aux_panel import attach_sector_and_scale
from v3.robustness.beta import compute_rolling_beta

REPO_ROOT = Path(__file__).resolve().parent.parent
V1_DECISION_MODULES = [
    "signals", "scoring.scorer", "scoring.pipeline", "backtest.engine", "backtest.decision",
    "forward_test", "ensemble",
]


def test_v3_residual_never_imports_v1_decision_layers() -> None:
    offending = []
    for path in (REPO_ROOT / "v3" / "residual").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if any(name == d or name.startswith(f"{d}.") for d in V1_DECISION_MODULES):
                    offending.append(f"{path.relative_to(REPO_ROOT)}: imports {name}")
    assert not offending, offending


def test_residual_shock_checks_pass_on_real_leak_free_pipeline(tmp_path: Path) -> None:
    config, tickers = build_v3_3_config_and_tickers(tmp_path, n_tickers=6, n_days=300)
    dataset = build_v3_dataset(tickers, config)
    ticker_frame = pd.DataFrame({"ticker": tickers})
    sector_map = attach_sector_and_scale(ticker_frame)[["ticker", "sector33"]]
    beta_panel = compute_rolling_beta(dataset)
    baseline_with_residuals = compute_residual_targets(dataset, beta_panel, sector_map)

    dates = sorted(dataset["date"].unique())
    cutoff = dates[len(dates) // 2]
    if isinstance(cutoff, datetime.datetime):
        cutoff = cutoff.date()

    results = run_residual_shock_checks(
        tickers, config, cutoff, tmp_path / "residual_shock_work", baseline_with_residuals
    )
    assert len(results) == 4
    assert all(r.passed for r in results), results
