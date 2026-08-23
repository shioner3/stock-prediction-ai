"""Q1-Q5 Monotonicity check (spec section 6/7): does Forward Return
increase (roughly) monotonically from Q1 to Q5, not just "Q5 > Q1"?

Spearman/Kendall rank correlation here are computed on the 5 BUCKET-MEAN
points (quantile rank 1..5 vs each bucket's mean Forward Return) - a
tiny (n=5, 10-pair) computation, deliberately NOT the same thing as
v2/validation/ic.py's per-ticker daily cross-sectional IC (that one
correlates thousands of (score, return) pairs per day; this one
correlates 5 SUMMARY points). Kendall's tau-b is implemented directly
(pure numpy, O(n^2) - trivially fast at n=5) rather than via scipy
(not a project dependency - see v2/validation/ic.py's own docstring for
why), using the standard tie-corrected tau-b formula.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Q_ORDER = ["Q1", "Q2", "Q3", "Q4", "Q5"]


def spearman_on_points(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    rank_x = np.argsort(np.argsort(x)).astype(float)
    rank_y = np.argsort(np.argsort(y)).astype(float)
    corr = np.corrcoef(rank_x, rank_y)[0, 1]
    return float(corr) if corr == corr else None


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float | None:
    """Standard tie-corrected Kendall tau-b over all C(n,2) pairs."""
    n = len(x)
    if n < 2:
        return None
    concordant = discordant = tied_x = tied_y = tied_both = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            if dx == 0 and dy == 0:
                tied_both += 1
            elif dx == 0:
                tied_x += 1
            elif dy == 0:
                tied_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    n0 = n * (n - 1) / 2
    denom = np.sqrt((n0 - tied_x - tied_both) * (n0 - tied_y - tied_both))
    if denom == 0:
        return None
    return float((concordant - discordant) / denom)


@dataclass(frozen=True)
class MonotonicityResult:
    window_days: int
    bucket_means: dict[str, float | None]  # Q1..Q5 -> mean Forward Return
    is_monotonic_nondecreasing: bool | None
    spearman: float | None
    kendall: float | None
    q5_q1_spread: float | None
    non_monotonic_pattern: str | None  # human-readable flag, e.g. "Q5だけ高い"


def _describe_non_monotonic_pattern(means: list[float]) -> str | None:
    """Flags the specific non-monotonic shapes spec section 7 explicitly
    calls out - purely descriptive labeling, not a statistical test.
    """
    if all(b >= a for a, b in zip(means, means[1:])):
        return None  # already monotonic non-decreasing
    max_idx = int(np.argmax(means))
    if max_idx == 0:
        return "Q1が最も高い(逆行)"
    if max_idx == len(means) - 1:
        return "Q5は最高だが途中で非単調"
    mid = len(means) // 2
    if max_idx == mid:
        return f"{Q_ORDER[max_idx]}だけ高い(中間ピーク)"
    return "非単調(詳細はbucket_means参照)"


def compute_monotonicity(
    bucket_mean_returns: dict[str, float | None], window_days: int
) -> MonotonicityResult:
    """bucket_mean_returns: {"Q1": mean, ..., "Q5": mean} (from
    v2/stats.py::compute_quantile_bucket_stats() - callers extract
    .stats.mean_return per bucket before calling this).
    """
    means_list = [bucket_mean_returns.get(q) for q in Q_ORDER]
    if any(m is None for m in means_list):
        return MonotonicityResult(
            window_days=window_days, bucket_means=bucket_mean_returns,
            is_monotonic_nondecreasing=None, spearman=None, kendall=None,
            q5_q1_spread=None, non_monotonic_pattern=None,
        )

    means = np.array(means_list, dtype=float)
    ranks = np.arange(1, len(means) + 1, dtype=float)
    is_monotonic = bool(all(b >= a for a, b in zip(means, means[1:])))

    return MonotonicityResult(
        window_days=window_days,
        bucket_means=bucket_mean_returns,
        is_monotonic_nondecreasing=is_monotonic,
        spearman=spearman_on_points(ranks, means),
        kendall=kendall_tau_b(ranks, means),
        q5_q1_spread=float(means[-1] - means[0]),
        non_monotonic_pattern=(
            None if is_monotonic else _describe_non_monotonic_pattern(means.tolist())
        ),
    )
