"""Phase V3-4 Leakage check (spec section 22): `v3/robustness/*.py` never
imports V1's actual decision-making modules - the SAME corrected
blocklist `tests/test_v3_3_leakage.py` established (see that file's own
comment for why "scoring"/"backtest" are not blanket-blocked: both are
mostly generic, already-approved-for-reuse statistics utilities V2/V3-3
themselves already import - `backtest.market_regime`/`backtest.bootstrap`/
`backtest.permutation`/`backtest.multiple_testing`/`backtest.costs` are
all legitimate reuse here too).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V1_DECISION_MODULES = [
    "signals", "scoring.scorer", "scoring.pipeline", "backtest.engine", "backtest.decision",
    "forward_test", "ensemble",
]


def test_v3_robustness_never_imports_v1_decision_layers() -> None:
    offending = []
    for path in (REPO_ROOT / "v3" / "robustness").rglob("*.py"):
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
