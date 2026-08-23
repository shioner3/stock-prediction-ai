"""Phase V2-3: Q1 Negative Predictive Signal — Causal Structure & Robustness
Analysis.

This package decomposes the asymmetric finding from Phase V2-2 (`v2/
validation/`, `research/phase_v2_2_report.md`): the V2 Initial Research
Score's lowest-score bucket (Q1) shows a statistically robust NEGATIVE
relationship with future returns (relatively higher forward returns than
Q5, the opposite of the Score's design intent), while Q5 shows no
reliable edge. Phase V2-3 asks WHY Q1 behaves this way, not whether the
Score should be changed - Rule 1/2 (V2-1/V2-2 frozen) mean nothing here
ever mutates Score/Feature/Weight/Threshold definitions, and Rule 3
(no new Risk Filter) means this Phase only characterizes Q1's structure,
never adopts an exclusion rule.

Every genuinely new analysis primitive lives here (feature percentile
decomposition, single-feature quantile buckets, feature interaction
cross-tabs, Q1 internal heterogeneity, sector/segment/liquidity profiling,
timing placebo, random control, cross-sectional stability/Gini). Anything
with a direct V1 or Phase V2-2 analogue is imported unmodified instead of
being reimplemented here:
  - `v2.pipeline.run_v2_ranking` (V2-1, frozen) builds the ranked panel
    ONCE - every raw Feature column, all 6 category ranks, total_score,
    and all 7 forward_return_{n}d columns survive into its output, which
    is why no separate Feature/Score recomputation exists in this package.
  - `scoring.validation.assign_quantile_buckets` (V1, frozen) is reused
    both for the Score's own Q1-Q5 split (already computed identically in
    Phase V2-2) and for every SINGLE FEATURE's own Q1-Q5 split (section 8) -
    the same generic function, never a bespoke bucketing rule per feature.
  - `backtest.bootstrap.bootstrap_ci`, `backtest.day_cluster_bootstrap.
    day_cluster_bootstrap`, `backtest.block_bootstrap.block_bootstrap` (V1,
    frozen) directly compute Q1's OWN (single-group, not spread) CI -
    unlike Phase V2-2's `v2/validation/spread_bootstrap.py` (built for a
    Q5-Q1 DIFFERENCE), section 22 here only needs Q1's own mean CI, which
    these V1 functions already do natively.
  - `backtest.permutation.permutation_test`, `backtest.multiple_testing.
    benjamini_hochberg_correction` (V1, frozen) are reused unmodified, as
    in Phase V2-2.
  - `v2.validation.regime.analyze_by_regime`, `v2.validation.
    event_year_analysis.analyze_year_by_year`/`analyze_event_exclusion`,
    `v2.validation.concentration.compute_concentration` (Phase V2-2,
    frozen) already compute per-bucket (including Q1) breakdowns by
    Regime/Year/Event/concentration - reused directly rather than
    reimplemented for Q1's sake.

Interpretation note carried over from `v2/validation/__init__.py`: "V2 to
V1 import" is fine (V2-1/V2-2/V2-3 already import V1's frozen primitives
throughout this project); the actual invariant, verified via `git status`
every run, is that V1 never imports anything from `v2/`.
"""
