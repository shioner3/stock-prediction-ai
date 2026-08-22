---
signal_name: short_volume_breakdown
direction: SHORT
status: candidate (not yet backtested/evaluated - Phase 6)
---

# Signal Name

SHORT Volume Breakdown

# Direction

SHORT

# Hypothesis

Mirror of `long_volume_breakout`: a single-day price drop accompanied by
unusually high volume may reflect new information or a shift in
participation. Independent of any multi-day low comparison - that is
`short_breakdown`'s hypothesis, not this one's.

# Conditions

```
return_1d[t] < return_1d_max
AND volume_ratio_20d[t] > volume_ratio_min
```

Default parameters
(`config/settings.yaml: signals.short.volume_breakdown`):
`return_1d_max = -0.03`, `volume_ratio_min = 2.0`.

# Expected Behavior

Symmetric to `long_volume_breakout`'s expected behavior, mirrored in
direction.

# Known Risks

Same structural risks as `long_volume_breakout` (cannot distinguish a
durable news-driven move from a one-off large order or data anomaly;
unvalidated, unnormalized thresholds), mirrored in direction. Sharp
single-day drops can also reflect halted-then-reopened trading or other
data artifacts that Phase 1's validation layer does not fully rule out -
see README's known issues.

## Phase 6 OOS Evaluation (2026-08-19)

- config_hash: `a9b34ccb6e6e1e1a...`
- data_hash: `05d9d41dbe397257...`
- data range: 2022-01-04 .. 2024-06-28 (4 tickers: 7203/6758/9984/8951)
- walk-forward windows evaluated: 5 (TRAIN=12mo/VAL=3mo/OOS=3mo/STEP=3mo)

### Window-by-window (TRAIN -> VAL -> OOS expectancy, base cost)

| Window | OOS period | TRAIN n / exp | VAL n / exp | OOS n / exp | OOS PF |
|---|---|---|---|---|---|
| 0 | 2023-04-04..2023-07-04 | 9 / -2.41% | 2 / 4.36% | 1 / -5.14% | 0.0000 |
| 1 | 2023-07-04..2023-10-04 | 10 / -0.89% | 1 / -5.14% | 1 / 2.92% | inf |
| 2 | 2023-10-04..2024-01-04 | 9 / -1.64% | 1 / 2.92% | 1 / -5.19% | 0.0000 |
| 3 | 2024-01-04..2024-04-04 | 6 / 0.68% | 1 / -5.19% | 1 / 2.30% | inf |
| 4 | 2024-04-04..2024-06-28 | 5 / 0.26% | 1 / 2.30% | 1 / -7.28% | 0.0000 |

### Aggregate OOS (all windows combined, by transaction cost tier)

| Cost tier | n | win_rate | PF | expectancy |
|---|---|---|---|---|
| zero | 5 | 40.00% | 0.3484 | -2.18% |
| low | 5 | 40.00% | 0.3305 | -2.28% |
| base | 5 | 40.00% | 0.2965 | -2.48% |
| high | 5 | 40.00% | 0.2209 | -2.98% |

### Bootstrap (95% CI, base cost, n_resamples=10000, seed=42)

- expectancy: point=-2.48%, CI=[-6.01%, 1.06%]
- profit_factor: point=0.2965, CI=[0.0000, 2.0318]

### Permutation Test (forward_return_5d, n_permutations=10000, seed=43, two-sided)

- observed_mean=1.94%, p_value=0.2682, n_signal=5, n_population=1200

### Consistency and Regime

- windows with OOS PF>1: 2/5
- windows with OOS expectancy>0: 2/5
- by market regime: BULL: n=4, expectancy=-1.50%, NEUTRAL: n=1, expectancy=-4.89%

### Decision

**INSUFFICIENT_EVIDENCE** (candidate only - see README "Signal Decision"; not an automatic accept/reject)

Rationale (mechanical, from backtest/decision.py - not a narrative justification): this label reflects whether OOS trade count, aggregate expectancy/PF, the bootstrap CI on expectancy, the permutation p-value, window consistency, and high-cost-tier expectancy jointly cleared the fixed thresholds in backtest/decision.py, none of which were adjusted after seeing this or any other Signal's OOS result.
