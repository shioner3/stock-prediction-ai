"""Window/Year/Regime/Holding-Period stability analysis (spec section
24): summarizes the SPREAD of an already-computed per-slice metric
(Rank IC mean, Q5-Q1 spread) across slices - is one slice doing all the
work, or is the pattern broadly consistent? A small, new, generic
computation (no existing V1/V2 primitive summarizes "spread across
already-computed slice results" in this form).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StabilitySummary:
    label: str
    n_slices: int
    values: list[float]
    mean: float | None
    std: float | None
    min: float | None
    max: float | None
    positive_fraction: float | None


def summarize_stability(label: str, values: list[float | None]) -> StabilitySummary:
    clean = np.array([v for v in values if v is not None], dtype=float)
    n = len(clean)
    if n == 0:
        return StabilitySummary(label, 0, [], None, None, None, None, None)
    return StabilitySummary(
        label=label,
        n_slices=n,
        values=[float(v) for v in clean],
        mean=float(clean.mean()),
        std=float(clean.std(ddof=1)) if n > 1 else None,
        min=float(clean.min()),
        max=float(clean.max()),
        positive_fraction=float((clean > 0).mean()),
    )
