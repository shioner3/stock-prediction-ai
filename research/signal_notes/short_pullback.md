---
signal_name: short_pullback
direction: SHORT
status: candidate (not yet backtested/evaluated - Phase 6)
---

# Signal Name

SHORT Pullback ("戻り売り")

# Direction

SHORT

# Hypothesis

Mirror of `long_pullback`: within an established downtrend, a moderate,
bounded bounce ("戻り") may represent a lower-risk short entry than
chasing new lows - if the downtrend tends to resume rather than reverse.

# Conditions

```
SMA_fast[t] < SMA_slow[t]                       (downtrend intact)
AND Close[t] < SMA_fast[t]                       (price still below the fast MA)
AND min_depth <= bounce_depth[t] <= max_depth    (a bounded bounce)
```

Default parameters (`config/settings.yaml: signals.short.pullback`):
`sma_fast = 20`, `sma_slow = 50`, `min_depth = 0.03`, `max_depth = 0.15`.

`bounce_depth` (`features/pullback.py`, added in Phase 4) is the
fractional rise above the rolling `recent_high_window`-day low - the
structural mirror of `pullback_depth`, using the same lookback window
config value.

# Expected Behavior

Symmetric to `long_pullback`'s expected behavior, mirrored in direction.

# Known Risks

Same structural risks as `long_pullback` (arbitrary depth bounds,
deliberately excludes deeper MA-crossing bounces which `short_ma_rejection`
covers instead), mirrored in direction. Downtrends in individual equities
can also be driven by company-specific deterioration (not just technical
mean-reversion dynamics) in ways that make "bounce then resume falling"
a fundamentally different regime from the LONG uptrend case - this
asymmetry is not modeled by treating the two Signals as pure mirrors.

## Phase 6 OOS Evaluation (2026-08-19)

- config_hash: `a9b34ccb6e6e1e1a...`
- data_hash: `05d9d41dbe397257...`
- data range: 2022-01-04 .. 2024-06-28 (4 tickers: 7203/6758/9984/8951)
- walk-forward windows evaluated: 5 (TRAIN=12mo/VAL=3mo/OOS=3mo/STEP=3mo)

### Window-by-window (TRAIN -> VAL -> OOS expectancy, base cost)

| Window | OOS period | TRAIN n / exp | VAL n / exp | OOS n / exp | OOS PF |
|---|---|---|---|---|---|
| 0 | 2023-04-04..2023-07-04 | 18 / -2.62% | 5 / 0.60% | 3 / -0.71% | 0.4942 |
| 1 | 2023-07-04..2023-10-04 | 22 / -1.39% | 3 / -0.71% | 5 / 1.56% | 4.5616 |
| 2 | 2023-10-04..2024-01-04 | 19 / -0.43% | 5 / 1.56% | 6 / -0.55% | 0.6633 |
| 3 | 2024-01-04..2024-04-04 | 20 / 0.35% | 6 / -0.55% | 0 / N/A | N/A |
| 4 | 2024-04-04..2024-06-28 | 19 / 0.28% | 0 / N/A | 4 / -3.14% | 0.0034 |

### Aggregate OOS (all windows combined, by transaction cost tier)

| Cost tier | n | win_rate | PF | expectancy |
|---|---|---|---|---|
| zero | 18 | 44.44% | 0.8154 | -0.26% |
| low | 18 | 44.44% | 0.7551 | -0.36% |
| base | 18 | 44.44% | 0.6471 | -0.56% |
| high | 18 | 38.89% | 0.4406 | -1.06% |

### Bootstrap (95% CI, base cost, n_resamples=10000, seed=42)

- expectancy: point=-0.56%, CI=[-2.03%, 0.90%]
- profit_factor: point=0.6471, CI=[0.1683, 2.0839]

### Permutation Test (forward_return_5d, n_permutations=10000, seed=43, two-sided)

- observed_mean=-0.19%, p_value=0.8491, n_signal=41, n_population=1200

### Consistency and Regime

- windows with OOS PF>1: 1/4
- windows with OOS expectancy>0: 1/4
- by market regime: BULL: n=7, expectancy=-0.54%, NEUTRAL: n=11, expectancy=-0.09%

### Decision

**INSUFFICIENT_EVIDENCE** (candidate only - see README "Signal Decision"; not an automatic accept/reject)

Rationale (mechanical, from backtest/decision.py - not a narrative justification): this label reflects whether OOS trade count, aggregate expectancy/PF, the bootstrap CI on expectancy, the permutation p-value, window consistency, and high-cost-tier expectancy jointly cleared the fixed thresholds in backtest/decision.py, none of which were adjusted after seeing this or any other Signal's OOS result.
