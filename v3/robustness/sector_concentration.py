"""Spec section 11: Sector/Industry Concentration. Trade share and PnL
contribution per JPX sector33 (`v2.causal.segment`, unmodified - same
local cache Phase V2-3 already used), plus leave-largest-sector-out.
"Sector-neutralized performance" is NOT a new computation: it is exactly
`market_decomposition.py`'s `VARIANT_SECTOR_RELATIVE` result (raw return
minus that day's same-sector mean, evaluated on the ORIGINAL prediction's
Q1-Q5 buckets) - reused directly here rather than recomputed a second way.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from scoring.validation import assign_quantile_buckets
from v3.robustness.common_eval import SliceEvaluation, evaluate_slice
from v3.robustness.gini import compute_contribution_ranking, gini_coefficient

TOP_K_VALUES = (1, 3, 5)


@dataclass(frozen=True)
class SectorTradeShare:
    sector33: str
    n: int
    trade_share: float
    pnl_share: float | None


@dataclass(frozen=True)
class SectorConcentrationResult:
    sector_shares: list[SectorTradeShare]
    gini_sector_contribution: float | None
    leave_largest_sector_out: SliceEvaluation


def run_sector_concentration(
    predictions_with_sector: pd.DataFrame, window_days: int, prediction_col: str = "prediction",
    actual_col: str = "actual", sector_col: str = "sector33",
) -> SectorConcentrationResult:
    valid = predictions_with_sector.dropna(subset=[prediction_col, actual_col]).copy()
    valid["_bucket"] = assign_quantile_buckets(valid[prediction_col])
    q5 = valid[valid["_bucket"] == "Q5"].dropna(subset=[sector_col])

    n_total_q5 = len(q5)
    contribution = compute_contribution_ranking(q5, sector_col, actual_col)
    gini = gini_coefficient(contribution.to_numpy())
    counts = q5.groupby(sector_col).size()
    total_pnl = contribution.sum()

    sector_shares = [
        SectorTradeShare(
            sector33=str(sector), n=int(counts.get(sector, 0)),
            trade_share=float(counts.get(sector, 0) / n_total_q5) if n_total_q5 else 0.0,
            pnl_share=float(pnl / total_pnl) if total_pnl != 0 else None,
        )
        for sector, pnl in contribution.items()
    ]
    sector_shares.sort(key=lambda s: s.pnl_share or 0.0, reverse=True)

    largest_sector = contribution.index[0] if len(contribution) else None
    remainder = (
        valid[valid[sector_col] != largest_sector] if largest_sector is not None else valid
    )
    leave_largest_out = evaluate_slice(
        remainder, f"excl_largest_sector_{largest_sector}", window_days, prediction_col, actual_col
    )

    return SectorConcentrationResult(
        sector_shares=sector_shares, gini_sector_contribution=gini,
        leave_largest_sector_out=leave_largest_out,
    )
