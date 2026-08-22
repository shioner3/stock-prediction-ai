"""Phase 6.5 section 14: multiple testing correction, INFORMATIONAL ONLY.

Evaluating 12 Signals (times windows/horizons/metrics) simultaneously
inflates the chance that at least one shows p < 0.05 purely by chance.
This module computes the standard Benjamini-Hochberg False Discovery
Rate correction over a set of p-values, but its output is NEVER read by
backtest/decision.py's classify() - Phase 6's ACCEPT_CANDIDATE/REJECT/
INSUFFICIENT_EVIDENCE decision rule is fixed and uses only the raw,
per-signal permutation p-value (see spec section 14 and 22: the Decision
criteria itself must not be loosened or tightened by this analysis).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FDRResult:
    signal_key: str
    raw_p_value: float
    adjusted_p_value: float  # Benjamini-Hochberg q-value
    rank: int  # 1 = smallest raw p-value
    n_tests: int
    significant: bool  # adjusted_p_value <= alpha


def benjamini_hochberg_correction(
    p_values: dict[str, float], alpha: float = 0.05
) -> dict[str, FDRResult]:
    """p_values: e.g. {"LONG:breakout": 0.031, "SHORT:pullback": 0.44, ...}
    (one raw p-value per Signal, typically the permutation test's
    p_value). NaN/None entries are dropped - they carry no information
    about family-wise error and would corrupt the rank ordering.
    """
    items = [
        (key, p) for key, p in p_values.items()
        if p is not None and not (isinstance(p, float) and math.isnan(p))
    ]
    n = len(items)
    if n == 0:
        return {}

    items_sorted = sorted(items, key=lambda kv: kv[1])
    raw_sorted = [p for _, p in items_sorted]

    # BH adjusted p-value: q(i) = min_{j>=i}( p(j) * n / j ), computed as a
    # running minimum from the largest rank down to keep it monotone
    # non-decreasing as rank increases (the standard BH step-up procedure).
    adjusted = [0.0] * n
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        candidate = raw_sorted[i] * n / rank
        running_min = min(running_min, candidate)
        adjusted[i] = min(running_min, 1.0)

    return {
        key: FDRResult(
            signal_key=key,
            raw_p_value=raw_p,
            adjusted_p_value=adjusted[i],
            rank=i + 1,
            n_tests=n,
            significant=adjusted[i] <= alpha,
        )
        for i, (key, raw_p) in enumerate(items_sorted)
    }
