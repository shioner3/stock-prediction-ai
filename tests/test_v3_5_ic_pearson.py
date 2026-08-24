"""v3/residual/ic_pearson.py - Pearson IC, verified to differ from
Spearman when the relationship is monotonic but non-linear (Pearson
sensitive to the actual VALUES, not just rank order).
"""

from __future__ import annotations

import pandas as pd

from v2.validation.ic import compute_daily_spearman_ic, summarize_ic
from v3.residual.ic_pearson import compute_daily_pearson_ic


def test_perfect_linear_relationship_gives_pearson_one() -> None:
    panel = pd.DataFrame({
        "date": ["2023-01-02"] * 5, "score": [1, 2, 3, 4, 5], "ret": [0.01, 0.02, 0.03, 0.04, 0.05],
    })
    daily = compute_daily_pearson_ic(panel, "score", "ret")
    assert daily[0].ic is not None
    assert abs(daily[0].ic - 1.0) < 1e-9


def test_monotonic_nonlinear_relationship_differs_from_spearman() -> None:
    # ret = score^3 - monotonic (Spearman = 1) but NOT linear (Pearson < 1).
    panel = pd.DataFrame({
        "date": ["2023-01-02"] * 5, "score": [1, 2, 3, 4, 5], "ret": [1, 8, 27, 64, 125],
    })
    pearson_daily = compute_daily_pearson_ic(panel, "score", "ret")
    spearman_daily = compute_daily_spearman_ic(panel, "score", "ret")
    assert spearman_daily[0].ic is not None and abs(spearman_daily[0].ic - 1.0) < 1e-9
    assert pearson_daily[0].ic is not None and pearson_daily[0].ic < 0.99


def test_summarize_reuses_v2_ic_summary_shape() -> None:
    panel = pd.DataFrame({
        "date": ["2023-01-02"] * 5 + ["2023-01-03"] * 5,
        "score": [1, 2, 3, 4, 5] * 2, "ret": [0.01, 0.02, 0.03, 0.04, 0.05] * 2,
    })
    daily = compute_daily_pearson_ic(panel, "score", "ret")
    summary = summarize_ic(daily, window_days=5)
    assert summary.n_days_with_ic == 2
    assert summary.mean_ic is not None and abs(summary.mean_ic - 1.0) < 1e-9
