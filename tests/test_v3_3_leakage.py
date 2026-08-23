"""Phase V3-3 Leakage tests (spec section 22):

- V1/V2 isolation: `v3/validation/*.py` never imports V1's actual
  decision-making modules (signals/scoring.scorer/scoring.pipeline/
  backtest.engine/backtest.decision/forward_test/ensemble) - see
  tests/test_v3_leakage.py's own comment for why "scoring"/"backtest"
  are not blanket-blocked (both packages are mostly generic, already-
  approved-for-reuse statistics utilities V2 itself already imports).
- Full Universe shock check smoke test: `v3/validation/leakage_check.py::
  run_full_universe_shock_checks()` runs end to end against a real
  (small) ticker set and reports all 4 shock types PASS - the same
  future-shock mechanics `tests/test_v3_leakage.py` already exhaustively
  validated at the individual-shock-function level, re-exercised here
  through this Phase's own orchestration wrapper.
"""

from __future__ import annotations

import ast
from pathlib import Path

from v3_3_test_helpers import build_v3_3_config_and_tickers

from v3.dataset import build_v3_dataset
from v3.validation.leakage_check import run_full_universe_shock_checks

REPO_ROOT = Path(__file__).resolve().parent.parent
V1_DECISION_MODULES = [
    "signals", "scoring.scorer", "scoring.pipeline", "backtest.engine", "backtest.decision",
    "forward_test", "ensemble",
]


def test_v3_validation_never_imports_v1_decision_layers() -> None:
    offending = []
    for path in (REPO_ROOT / "v3" / "validation").rglob("*.py"):
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


def test_shock_checks_pass_on_real_leak_free_pipeline(tmp_path: Path) -> None:
    config, tickers = build_v3_3_config_and_tickers(tmp_path, n_tickers=5, n_days=500)
    baseline = build_v3_dataset(tickers, config)
    dates = sorted(baseline["date"].unique())
    cutoff = dates[len(dates) // 2]

    results = run_full_universe_shock_checks(
        tickers, config, cutoff, tmp_path / "shock_work", baseline
    )
    assert len(results) == 4
    assert all(r.passed for r in results), results
