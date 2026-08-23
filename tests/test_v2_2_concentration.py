from __future__ import annotations

from datetime import date

import pandas as pd

from v2.validation.concentration import compute_concentration


def test_single_ticker_dominance_detected() -> None:
    rows = [
        (date(2024, 1, 1), "DOMINANT", "Q5", 1.0),
        (date(2024, 1, 2), "T1", "Q5", 0.01),
        (date(2024, 1, 3), "T2", "Q5", 0.01),
        (date(2024, 1, 4), "T3", "Q5", 0.01),
    ]
    scored = pd.DataFrame(
        {
            "date": [r[0] for r in rows], "ticker": [r[1] for r in rows],
            "score_bucket": [r[2] for r in rows], "ret": [r[3] for r in rows],
        }
    )
    result = compute_concentration(scored, "ret", window_days=5)
    top1 = next(c for c in result.ticker_contribution if c.top_k == 1)
    assert top1.share_of_total > 0.9


def test_evenly_spread_contribution() -> None:
    rows = [(date(2024, 1, 1 + i), f"T{i}", "Q5", 0.1) for i in range(10)]
    scored = pd.DataFrame(
        {
            "date": [r[0] for r in rows], "ticker": [r[1] for r in rows],
            "score_bucket": [r[2] for r in rows], "ret": [r[3] for r in rows],
        }
    )
    result = compute_concentration(scored, "ret", window_days=5)
    top1 = next(c for c in result.ticker_contribution if c.top_k == 1)
    assert abs(top1.share_of_total - 0.1) < 1e-9


def test_empty_bucket_gives_none_shares() -> None:
    scored = pd.DataFrame({"date": [], "ticker": [], "score_bucket": [], "ret": []})
    result = compute_concentration(scored, "ret", window_days=5)
    assert all(c.share_of_total is None for c in result.ticker_contribution)


def test_day_contribution_computed_separately_from_ticker() -> None:
    rows = [
        (date(2024, 1, 1), "T0", "Q5", 0.5),
        (date(2024, 1, 1), "T1", "Q5", 0.5),
        (date(2024, 1, 2), "T2", "Q5", 0.01),
    ]
    scored = pd.DataFrame(
        {
            "date": [r[0] for r in rows], "ticker": [r[1] for r in rows],
            "score_bucket": [r[2] for r in rows], "ret": [r[3] for r in rows],
        }
    )
    result = compute_concentration(scored, "ret", window_days=5)
    day_top1 = next(c for c in result.day_contribution if c.top_k == 1)
    assert day_top1.share_of_total > 0.9  # 2024-01-01 dominates
