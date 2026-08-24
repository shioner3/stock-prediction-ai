"""Spec section 12: Random Matched Control. For each day's Q5-selected
tickers, draws one same-day control ticker from the rest of the Universe
sharing similar size (JPX `scale` classification - `v2.causal.segment`,
unmodified), liquidity (`turnover` = Close*Volume tercile, same proxy
`v2.causal.segment.attach_liquidity_columns()`/the V3 Feature Registry's
own `turnover` Feature already use), and price level (Close tercile) -
isolating whether Q5's edge is explained by "these are just big/liquid/
expensive stocks that happened to do well" rather than genuine selection.

Matching falls back through 3 tiers (documented, never silently dropped):
  1. scale + turnover tercile + price tercile (all 3, same day)
  2. scale + turnover tercile (relaxed if tier 1 has no candidate)
  3. scale only (relaxed further if tier 2 has no candidate)
A ticker with literally no same-day, same-scale candidate is excluded
from the comparison (tracked in `n_unmatched`, never silently imputed).

Comparison outcomes reuse ALREADY-COMPUTED Target Registry columns
(raw 5/10/15/20d, TOPIX-relative 5d, Vol-adjusted 5d) rather than
re-deriving returns - the SAME matched (ticker, control_ticker, date)
triple, computed once, is evaluated across all of them.

`build_full_day_panel()`/`attach_outcomes()`/`run_matched_control_
analysis()` all accept an optional `outcome_cols` override (default =
`MATCH_OUTCOME_COLS`, so every V3-4 call site and test is byte-for-byte
unaffected) - added for Phase V3-5's reuse of this exact matching
machinery against its own new residual Target columns, per that Phase's
own spec section 25 ("V3-4で実施したMatched Controlを再利用").

**Bug found and fixed during this Phase's real Full Universe run**
(same failure class as `market_decomposition.py`'s own "Bug found" note):
`build_full_day_panel()` pulls MATCH_OUTCOME_COLS straight from the FULL
(unfiltered) dataset, which still contains the known raw-data-artifact
and TOPIX-Proxy-artifact-date rows. A single such row is enough to drag
a ~400K-row treatment/control MEAN up by tens of points (observed:
target_raw_10d/15d/20d treatment means of ~54-56, i.e. 5,400-5,600%,
versus target_raw_5d's own plausible +0.77% on the same real run) even
though only ONE artifact row is involved. Masked to NaN with the SAME
`MAX_PLAUSIBLE_FORWARD_RETURN` bound for every RETURN-scaled outcome
column (raw 5/10/15/20d + TOPIX-relative 5d) - `target_vol_adjusted_5d`
is deliberately left unfiltered here, since it is a RATIO (return /
volatility), not itself bounded by a "forward return" plausibility
check (same distinction `v3/targets/compute.py`'s own Target Registry
draws between Raw/TOPIX-relative and Vol-adjusted/Risk-adjusted).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.bootstrap import BootstrapDiffResult, bootstrap_diff_ci
from v2.stats import MAX_PLAUSIBLE_FORWARD_RETURN, ReturnStats, compute_return_stats
from v3.validation.wfo_config import TRADE_LEVEL_BOOTSTRAP_CONFIG

MATCH_SEED = 733
N_TERCILES = 3
TERCILE_LABELS = ["low", "mid", "high"]
MATCH_OUTCOME_COLS = (
    "target_raw_5d", "target_raw_10d", "target_raw_15d", "target_raw_20d",
    "target_topix_relative_5d", "target_vol_adjusted_5d",
)
RETURN_SCALED_OUTCOME_COLS = (
    "target_raw_5d", "target_raw_10d", "target_raw_15d", "target_raw_20d",
    "target_topix_relative_5d",
)


def _add_match_bins(day_panel: pd.DataFrame) -> pd.DataFrame:
    out = day_panel.copy()
    try:
        out["_turnover_bin"] = pd.qcut(
            out["turnover"], N_TERCILES, labels=TERCILE_LABELS, duplicates="drop"
        )
    except ValueError:
        out["_turnover_bin"] = "low"
    try:
        out["_price_bin"] = pd.qcut(
            out["close"], N_TERCILES, labels=TERCILE_LABELS, duplicates="drop"
        )
    except ValueError:
        out["_price_bin"] = "low"
    return out


def build_full_day_panel(
    dataset: pd.DataFrame, price_volume_panel: pd.DataFrame, sector_map: pd.DataFrame,
    outcome_cols: tuple[str, ...] = MATCH_OUTCOME_COLS,
    return_scaled_cols: tuple[str, ...] = RETURN_SCALED_OUTCOME_COLS,
) -> pd.DataFrame:
    """date/ticker/scale/turnover/close + outcome_cols, one row per
    (date, ticker) across the ENTIRE dataset (the matching POOL, not just
    Q5-selected rows).
    """
    cols = ["date", "ticker", *outcome_cols]
    base = dataset[[c for c in cols if c in dataset.columns]].copy()
    for col in return_scaled_cols:
        if col in base.columns:
            base[col] = base[col].where(base[col].abs() <= MAX_PLAUSIBLE_FORWARD_RETURN)
    base = base.merge(price_volume_panel, on=["date", "ticker"], how="inner")
    base["turnover"] = base["close"] * base["volume"]
    base = base.merge(
        sector_map[["ticker", "scale"]].drop_duplicates(subset=["ticker"]), on="ticker", how="left"
    )
    return base.dropna(subset=["scale", "turnover", "close"])


def build_matched_control_pairs(
    q5_predictions: pd.DataFrame, full_day_panel: pd.DataFrame, seed: int = MATCH_SEED,
) -> pd.DataFrame:
    """q5_predictions: date/ticker rows already restricted to the
    original Primary ranking's Q5 bucket. Returns one row per matched
    Q5 ticker: date/ticker/control_ticker/match_tier.
    """
    rng = np.random.default_rng(seed)
    binned_by_date = {
        d: _add_match_bins(g) for d, g in full_day_panel.groupby("date", sort=False)
    }
    q5_by_date: dict = {}
    for d, g in q5_predictions.groupby("date", sort=False):
        q5_by_date[d] = sorted(g["ticker"].unique())

    rows = []
    for day in sorted(q5_by_date.keys()):
        day_panel = binned_by_date.get(day)
        if day_panel is None:
            continue
        q5_tickers = q5_by_date[day]
        pool = day_panel[~day_panel["ticker"].isin(q5_tickers)]
        by_ticker = day_panel.set_index("ticker")
        for ticker in q5_tickers:
            if ticker not in by_ticker.index:
                continue
            row = by_ticker.loc[ticker]
            scale, tbin, pbin = row["scale"], row["_turnover_bin"], row["_price_bin"]

            tier = "scale_turnover_price"
            candidates = pool[
                (pool["scale"] == scale)
                & (pool["_turnover_bin"] == tbin)
                & (pool["_price_bin"] == pbin)
            ]
            if candidates.empty:
                tier = "scale_turnover"
                candidates = pool[(pool["scale"] == scale) & (pool["_turnover_bin"] == tbin)]
            if candidates.empty:
                tier = "scale_only"
                candidates = pool[pool["scale"] == scale]
            if candidates.empty:
                continue

            chosen_idx = int(rng.integers(len(candidates)))
            control_ticker = candidates.iloc[chosen_idx]["ticker"]
            rows.append({
                "date": day, "ticker": ticker, "control_ticker": control_ticker, "match_tier": tier,
            })

    return pd.DataFrame(rows, columns=["date", "ticker", "control_ticker", "match_tier"])


def attach_outcomes(
    pairs: pd.DataFrame, full_day_panel: pd.DataFrame,
    outcome_cols: tuple[str, ...] = MATCH_OUTCOME_COLS,
) -> pd.DataFrame:
    """Merges each pair's treatment (Q5 ticker) and control ticker's
    outcome_cols onto `pairs` - suffixed `_treatment`/`_control`.
    """
    treatment_cols = ["date", "ticker", *outcome_cols]
    treatment = full_day_panel[treatment_cols].rename(
        columns={c: f"{c}_treatment" for c in outcome_cols}
    )
    out = pairs.merge(treatment, on=["date", "ticker"], how="left")

    control_cols = ["date", "ticker", *outcome_cols]
    control = full_day_panel[control_cols].rename(
        columns={"ticker": "control_ticker", **{c: f"{c}_control" for c in outcome_cols}}
    )
    out = out.merge(control, on=["date", "control_ticker"], how="left")
    return out


@dataclass(frozen=True)
class MatchedControlComparison:
    outcome_col: str
    treatment_stats: ReturnStats
    control_stats: ReturnStats
    diff_bootstrap: BootstrapDiffResult


@dataclass(frozen=True)
class MatchedControlResult:
    n_q5_rows: int
    n_matched: int
    n_unmatched: int
    match_tier_counts: dict[str, int]
    comparisons: dict[str, MatchedControlComparison]


def run_matched_control_analysis(
    q5_predictions: pd.DataFrame, full_day_panel: pd.DataFrame, seed: int = MATCH_SEED,
    outcome_cols: tuple[str, ...] = MATCH_OUTCOME_COLS,
) -> MatchedControlResult:
    n_q5_rows = len(q5_predictions[["date", "ticker"]].drop_duplicates())
    pairs = build_matched_control_pairs(q5_predictions, full_day_panel, seed)
    with_outcomes = attach_outcomes(pairs, full_day_panel, outcome_cols)

    comparisons: dict[str, MatchedControlComparison] = {}
    for col in outcome_cols:
        treatment = with_outcomes[f"{col}_treatment"].dropna()
        control = with_outcomes[f"{col}_control"].dropna()
        comparisons[col] = MatchedControlComparison(
            outcome_col=col,
            treatment_stats=compute_return_stats(treatment),
            control_stats=compute_return_stats(control),
            diff_bootstrap=bootstrap_diff_ci(
                treatment.to_numpy(), control.to_numpy(), TRADE_LEVEL_BOOTSTRAP_CONFIG
            ),
        )

    tier_counts = pairs["match_tier"].value_counts().to_dict() if len(pairs) else {}
    return MatchedControlResult(
        n_q5_rows=n_q5_rows, n_matched=len(pairs), n_unmatched=n_q5_rows - len(pairs),
        match_tier_counts={str(k): int(v) for k, v in tier_counts.items()},
        comparisons=comparisons,
    )
