---
signal_name: long_pullback
direction: LONG
status: candidate (not yet backtested/evaluated - Phase 6)
---

# Signal Name

LONG Pullback

# Direction

LONG

# Hypothesis

Within an established uptrend, a moderate, bounded pullback may represent
a lower-risk entry point than chasing new highs - if (and this is the
part left for the backtest to answer) the uptrend tends to resume rather
than reverse. A pullback that is too shallow may just be noise; one that
is too deep may indicate the uptrend has already broken.

# Conditions

```
SMA_fast[t] > SMA_slow[t]                          (uptrend intact)
AND Close[t] > SMA_fast[t]                          (price still above the fast MA)
AND min_depth <= pullback_depth[t] <= max_depth     (a bounded dip)
```

Default parameters (`config/settings.yaml: signals.long.pullback`):
`sma_fast = 20`, `sma_slow = 50`, `min_depth = 0.03`, `max_depth = 0.15`.

`pullback_depth` (`features/pullback.py`) is the fractional distance
below the rolling `recent_high_window`-day high (default 20 days,
`config: features.pullback.recent_high_window`).

# Expected Behavior

- Should trigger more often in choppy uptrends (frequent shallow dips)
  than in smooth, low-volatility uptrends (few dips deep enough to clear
  `min_depth`).
- Should almost never co-trigger with `long_momentum_continuation` on the
  same day for the same stock, since that Signal requires strong positive
  5-day return while this one requires an active pullback - though this
  is a plausibility expectation, not a hard constraint enforced in code.

# Known Risks

- `min_depth`/`max_depth` are arbitrary round numbers (3%/15%), not fit
  to any data - a real "meaningful but not broken" pullback range likely
  varies by volatility regime and was not modeled here.
- The definition requires Close to still be above SMA_fast even during
  the pullback, which excludes deeper, MA-crossing pullbacks by
  construction - `long_ma_rebound` is meant to cover that case instead
  (they are deliberately different hypotheses, not variations of the
  same one).

## Phase 6 OOS Evaluation (2026-08-19)

- config_hash: `a9b34ccb6e6e1e1a...`
- data_hash: `05d9d41dbe397257...`
- data range: 2022-01-04 .. 2024-06-28 (4 tickers: 7203/6758/9984/8951)
- walk-forward windows evaluated: 5 (TRAIN=12mo/VAL=3mo/OOS=3mo/STEP=3mo)

### Window-by-window (TRAIN -> VAL -> OOS expectancy, base cost)

| Window | OOS period | TRAIN n / exp | VAL n / exp | OOS n / exp | OOS PF |
|---|---|---|---|---|---|
| 0 | 2023-04-04..2023-07-04 | 18 / -0.71% | 2 / -3.36% | 6 / 1.70% | inf |
| 1 | 2023-07-04..2023-10-04 | 20 / -0.97% | 6 / 1.70% | 8 / -0.06% | 0.9331 |
| 2 | 2023-10-04..2024-01-04 | 17 / -0.38% | 8 / -0.06% | 8 / 0.10% | 1.0724 |
| 3 | 2024-01-04..2024-04-04 | 20 / -0.31% | 8 / 0.10% | 10 / 1.30% | 3.3582 |
| 4 | 2024-04-04..2024-06-28 | 24 / 0.16% | 9 / 0.93% | 4 / 0.91% | 1.5391 |

### Aggregate OOS (all windows combined, by transaction cost tier)

| Cost tier | n | win_rate | PF | expectancy |
|---|---|---|---|---|
| zero | 35 | 60.00% | 2.2461 | 0.94% |
| low | 35 | 60.00% | 2.0582 | 0.84% |
| base | 35 | 60.00% | 1.7337 | 0.64% |
| high | 35 | 45.71% | 1.1315 | 0.14% |

### Bootstrap (95% CI, base cost, n_resamples=10000, seed=42)

- expectancy: point=0.64%, CI=[-0.33%, 1.69%]
- profit_factor: point=1.7337, CI=[0.7241, 4.1210]

### Permutation Test (forward_return_5d, n_permutations=10000, seed=43, two-sided)

- observed_mean=0.50%, p_value=0.6378, n_signal=77, n_population=1200

### Consistency and Regime

- windows with OOS PF>1: 4/5
- windows with OOS expectancy>0: 4/5
- by market regime: BULL: n=24, expectancy=1.06%, NEUTRAL: n=11, expectancy=0.68%

### Decision

**INSUFFICIENT_EVIDENCE** (candidate only - see README "Signal Decision"; not an automatic accept/reject)

Rationale (mechanical, from backtest/decision.py - not a narrative justification): this label reflects whether OOS trade count, aggregate expectancy/PF, the bootstrap CI on expectancy, the permutation p-value, window consistency, and high-cost-tier expectancy jointly cleared the fixed thresholds in backtest/decision.py, none of which were adjusted after seeing this or any other Signal's OOS result.
