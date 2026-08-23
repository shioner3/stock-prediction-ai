"""V3 Target Registry (spec section 3): 4 Horizons x 4 Target variants =
16 target columns, all generated - WHICH variant/horizon is eventually
adopted for the Candidate Ranking Engine is an explicit future decision
(spec: "どのTargetを最終採用するかは事前固定した評価基準で決定する"),
made in a later V3 Phase against pre-registered OOS evaluation criteria,
NOT chosen here or after seeing any result.

Targets are the FUTURE-LOOKING side of the Feature(t) -> FutureReturn(t+h)
boundary (spec section 2) - unlike v3/features/*.py, it is CORRECT and
expected for target formulas to read rows > t (Close[t+h], forward
High/Low). `v3/leakage/availability_check.py`'s AST scan is therefore
scoped to `v3/features/` only, never `v3/targets/` - the boundary itself,
not "no future reads anywhere," is what must never be crossed.
"""

from __future__ import annotations

from dataclasses import dataclass

HORIZONS: tuple[int, ...] = (5, 10, 15, 20)

VARIANT_RAW = "raw"
VARIANT_TOPIX_RELATIVE = "topix_relative"
VARIANT_VOL_ADJUSTED = "vol_adjusted"
VARIANT_RISK_ADJUSTED = "risk_adjusted"

VARIANT_FORMULAS: dict[str, str] = {
    VARIANT_RAW: "Close[t+h]/Close[t] - 1",
    VARIANT_TOPIX_RELATIVE: "raw(stock, h) - raw(TOPIX Proxy, h), aligned by date",
    VARIANT_VOL_ADJUSTED: "raw(h) / volatility_20d[t]  (ex-ante: t-time-known risk denominator)",
    VARIANT_RISK_ADJUSTED: "raw(h) / abs(mae_h)  (ex-post: realized worst-case-during-holding "
    "denominator, mae_h from targets.forward_returns.compute_mfe_mae(direction='LONG'))",
}


def target_column_name(variant: str, horizon: int) -> str:
    return f"target_{variant}_{horizon}d"


@dataclass(frozen=True)
class TargetSpec:
    name: str
    variant: str
    horizon_days: int
    formula: str
    uses_future_data: bool  # always True - see module docstring


TARGET_REGISTRY: list[TargetSpec] = [
    TargetSpec(
        name=target_column_name(variant, horizon),
        variant=variant,
        horizon_days=horizon,
        formula=VARIANT_FORMULAS[variant],
        uses_future_data=True,
    )
    for variant in (
        VARIANT_RAW, VARIANT_TOPIX_RELATIVE, VARIANT_VOL_ADJUSTED, VARIANT_RISK_ADJUSTED,
    )
    for horizon in HORIZONS
]

TARGET_COLUMN_NAMES: list[str] = [t.name for t in TARGET_REGISTRY]
