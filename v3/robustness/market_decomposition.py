"""Spec section 3: Market Timing vs Stock Selection - Return Definition
decomposition. Holds the Primary Model A prediction's Q1-Q5 bucket
assignment FIXED (exactly as V3-3 computed it - a single global quantile
split of the pooled OOS prediction column, `scoring.validation.
assign_quantile_buckets()`, unmodified) and varies only the RETURN used
to evaluate that fixed ranking. Answers: "even holding the model's
ranking constant, does the measured edge survive once market-wide /
sector-wide movement is stripped out of the OUTCOME?" - a different
question from `cross_sectional_decomp.py` (sections 4/5), which instead
varies the RANKING itself.

5 variants (spec section 3.A-E):
  A. Raw            - `target_raw_5d` as-is (V3-3's own Primary outcome).
  B. TOPIX-relative - `target_topix_relative_5d`, an ALREADY-COMPUTED,
                       frozen Target Registry column (`v3/targets/
                       compute.py`, beta=1 assumption) - no new formula.
  C. Beta-adjusted  - raw - beta_i * market_forward_return, where
                       market_forward_return is recovered ALGEBRAICALLY
                       from A and B (market_forward = raw - topix_relative,
                       identical for every ticker on a given date by
                       construction) rather than re-deriving TOPIX's own
                       forward return from scratch, and beta_i is
                       `beta.compute_rolling_beta()`'s trailing-window
                       estimate (the one genuinely new quantity this
                       Phase computes - everything else here is reuse).
  D. Sector-relative - raw - sector_day_mean(raw), sector_day_mean computed
                       across the FULL Universe (every ticker trading that
                       day, not just this ranking's selected rows) grouped
                       by JPX sector33 (`v2.causal.segment`, unmodified).
  E. Market-neutralized - raw - day_mean(raw), same idea as D without the
                       sector grouping (the whole Universe's same-day mean).

For each variant: Q1-Q5 spread, Rank IC (of the ORIGINAL prediction vs
this variant's return - Rank IC is unaffected by A vs E's differencing
choice only insofar as the differenced column's OWN cross-sectional rank
that day differs from the raw column's rank, which it generally will,
since day_mean/sector_mean subtraction is a per-GROUP not a per-ROW
constant), Q5 mean, Q1 mean, Top-N return/PF/Expectancy (Top-N selected
by the ORIGINAL prediction, evaluated on the variant's return), and
Bootstrap CI on the Q5-Q1 spread - all via already-existing V1/V2/V3-3
primitives, swapping only which column plays "actual".

**Bug found and fixed during this Phase's real Full Universe run**
(`research/phase_v3_4_report.md`'s own "Bugs Discovered" section): the
Sector-relative and Market-neutralized variants' day/sector MEAN is
computed across the FULL Universe dataset (by design - every ticker
trading that day, not just the Primary ranking's own OOS rows), but the
full dataset's own `target_raw_5d` column still contains the SAME known
raw-data artifact class already documented in V2-1/Phase V3-3 (a
handful of rows with a multi-million-percent "return" from an
unadjusted-looking price jump). A mean is extremely outlier-sensitive -
a single such row poisons an entire day's or sector's mean, producing a
physically nonsensical variant value for every OTHER ticker that
day/sector (observed: Sector-relative/Market-neutralized spreads of
-35.5/+21.1 on real data, versus a plausible few-percent range).
`build_return_variant_columns()` now applies the SAME already-established
`v2.stats.exclude_implausible_returns()`/`MAX_PLAUSIBLE_FORWARD_RETURN`
filter to the full dataset's `raw_target_col` BEFORE computing either
mean - never to `out`/the OOS rows themselves (those already went
through this exact filter once, in `v3/models/data_prep.py`).

A second, related instance of the same failure class was found on the
SAME real run: `target_topix_relative_5d` itself (a frozen V3-1 Target
column never previously stress-tested against the Full Universe) carries
a TOPIX Proxy data artifact around 2026-03-30/31 making TOPIX's own
forward return ~+930% on those 2 dates - since every ticker's
topix_relative subtracts that SAME market-wide value, it corrupted
EVERY ticker's topix_relative (and, downstream, `market_forward` and
thus Beta-adjusted) on those 2 dates, even though their own
`target_raw_5d` stayed perfectly plausible. Fixed the same way: masked
to NaN with the same bound, never dropped/clipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from scoring.validation import assign_quantile_buckets
from v2.stats import (
    MAX_PLAUSIBLE_FORWARD_RETURN,
    QuantileBucketStats,
    compute_q5_q1_spread,
    compute_quantile_bucket_stats,
    exclude_implausible_returns,
)
from v2.validation.ic import ICSummary, compute_daily_spearman_ic, summarize_ic
from v3.validation.robustness import SpreadBootstrapBattery, bootstrap_q5_q1_spread
from v3.validation.topn_portfolio import TopNPortfolioMetrics, compute_topn_portfolio_metrics
from v3.validation.wfo_config import TOP_N_VALUES

VARIANT_RAW = "raw"
VARIANT_TOPIX_RELATIVE = "topix_relative"
VARIANT_BETA_ADJUSTED = "beta_adjusted"
VARIANT_SECTOR_RELATIVE = "sector_relative"
VARIANT_MARKET_NEUTRALIZED = "market_neutralized"
RETURN_VARIANTS = (
    VARIANT_RAW, VARIANT_TOPIX_RELATIVE, VARIANT_BETA_ADJUSTED,
    VARIANT_SECTOR_RELATIVE, VARIANT_MARKET_NEUTRALIZED,
)


def build_return_variant_columns(
    primary_predictions: pd.DataFrame,
    dataset: pd.DataFrame,
    beta_panel: pd.DataFrame,
    sector_map: pd.DataFrame,
    raw_target_col: str = "target_raw_5d",
    topix_relative_target_col: str = "target_topix_relative_5d",
) -> pd.DataFrame:
    """primary_predictions: date/ticker/actual/prediction (actual ==
    dataset[raw_target_col] for these rows already). Returns
    primary_predictions with 5 new columns actual_<variant> appended.
    """
    out = primary_predictions.copy()

    target_cols = ["date", "ticker", raw_target_col, topix_relative_target_col]
    targets = dataset[target_cols].drop_duplicates(subset=["date", "ticker"])
    out = out.merge(targets, on=["date", "ticker"], how="left")
    out[f"actual_{VARIANT_RAW}"] = out["actual"]
    # `target_topix_relative_5d` is a FROZEN Target Registry column
    # (v3/targets/compute.py, unmodified) never previously validated for
    # implausible values on the Full Universe - a TOPIX Proxy data
    # artifact around 2026-03-30/31 (real Full Universe run) makes
    # TOPIX's own forward return ~+930% for those 2 dates, which - since
    # target_topix_relative_5d subtracts the SAME market-wide forward
    # return from every ticker that day - corrupts EVERY ticker's
    # topix_relative value on those 2 dates (~5,720 rows), even though
    # their own target_raw_5d stays perfectly plausible. Masked to NaN
    # (never dropped/clipped) using the SAME MAX_PLAUSIBLE_FORWARD_RETURN
    # bound already established for target_raw_5d, so the affected rows
    # are simply excluded downstream (evaluate_return_variant's own
    # dropna()) rather than silently distorting the mean.
    plausible_topix_relative = out[topix_relative_target_col].where(
        out[topix_relative_target_col].abs() <= MAX_PLAUSIBLE_FORWARD_RETURN
    )
    out[f"actual_{VARIANT_TOPIX_RELATIVE}"] = plausible_topix_relative

    # market_forward(date) = raw - topix_relative, identical for every
    # ticker on a given date by target_topix_relative_5d's own definition
    # (v3/targets/compute.py) - recovered algebraically rather than
    # re-deriving TOPIX's own forward return a second time. Computed from
    # the ALREADY-masked topix_relative column above, so a corrupted
    # market_forward on an artifact date propagates NaN into
    # actual_beta_adjusted too, instead of silently inheriting the same
    # ~+930% distortion.
    market_forward = out[raw_target_col] - plausible_topix_relative
    out = out.merge(beta_panel, on=["date", "ticker"], how="left")
    out[f"actual_{VARIANT_BETA_ADJUSTED}"] = out[raw_target_col] - out["beta"] * market_forward

    plausible_full = exclude_implausible_returns(
        dataset[["date", "ticker", raw_target_col]], raw_target_col, MAX_PLAUSIBLE_FORWARD_RETURN
    )

    sector_lookup = sector_map[["ticker", "sector33"]].drop_duplicates(subset=["ticker"])
    full_with_sector = plausible_full.merge(sector_lookup, on="ticker", how="left")
    sector_day_mean = (
        full_with_sector.groupby(["date", "sector33"])[raw_target_col].transform("mean")
    )
    full_with_sector = full_with_sector.assign(_sector_day_mean=sector_day_mean)
    out = out.merge(
        full_with_sector[["date", "ticker", "_sector_day_mean"]], on=["date", "ticker"], how="left"
    )
    out[f"actual_{VARIANT_SECTOR_RELATIVE}"] = out[raw_target_col] - out["_sector_day_mean"]

    day_mean = plausible_full.groupby("date")[raw_target_col].transform("mean")
    day_mean_lookup = plausible_full[["date", "ticker"]].assign(_day_mean=day_mean)
    out = out.merge(day_mean_lookup, on=["date", "ticker"], how="left")
    out[f"actual_{VARIANT_MARKET_NEUTRALIZED}"] = out[raw_target_col] - out["_day_mean"]

    return out


@dataclass(frozen=True)
class ReturnVariantResult:
    variant: str
    n: int
    bucket_stats: list[QuantileBucketStats]
    q5_q1_spread: float | None
    q5_mean: float | None
    q1_mean: float | None
    ic_summary: ICSummary
    topn: dict[int, TopNPortfolioMetrics]
    spread_bootstrap: SpreadBootstrapBattery


def evaluate_return_variant(
    predictions_with_variants: pd.DataFrame,
    variant: str,
    window_days: int,
    prediction_col: str = "prediction",
) -> ReturnVariantResult:
    actual_col = f"actual_{variant}"
    valid = predictions_with_variants.dropna(subset=[prediction_col, actual_col]).copy()
    # bucket is fixed from the ORIGINAL prediction, exactly as V3-3
    # computed it - never re-derived from the variant's return.
    valid["_bucket"] = assign_quantile_buckets(valid[prediction_col])

    bucket_stats = compute_quantile_bucket_stats(valid, "_bucket", actual_col, window_days)
    by_bucket = {b.bucket: b.stats for b in bucket_stats}
    q5_mean = by_bucket["Q5"].mean_return if "Q5" in by_bucket else None
    q1_mean = by_bucket["Q1"].mean_return if "Q1" in by_bucket else None

    daily_ic = compute_daily_spearman_ic(valid, prediction_col, actual_col)
    ic_summary = summarize_ic(daily_ic, window_days)

    topn = {
        n: compute_topn_portfolio_metrics(
            valid, n, actual_col, window_days, score_col=prediction_col
        )
        for n in TOP_N_VALUES
    }
    spread_bootstrap = bootstrap_q5_q1_spread(
        valid, prediction_col=prediction_col, actual_col=actual_col
    )

    return ReturnVariantResult(
        variant=variant, n=len(valid), bucket_stats=bucket_stats,
        q5_q1_spread=compute_q5_q1_spread(bucket_stats), q5_mean=q5_mean, q1_mean=q1_mean,
        ic_summary=ic_summary, topn=topn, spread_bootstrap=spread_bootstrap,
    )


def run_market_decomposition(
    predictions_with_variants: pd.DataFrame, window_days: int, prediction_col: str = "prediction",
) -> dict[str, ReturnVariantResult]:
    return {
        variant: evaluate_return_variant(
            predictions_with_variants, variant, window_days, prediction_col
        )
        for variant in RETURN_VARIANTS
    }
