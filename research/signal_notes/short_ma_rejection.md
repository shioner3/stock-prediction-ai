---
signal_name: short_ma_rejection
direction: SHORT
status: candidate (not yet backtested/evaluated - Phase 6)
---

# Signal Name

SHORT MA Rejection

# Direction

SHORT

# Hypothesis

Mirror of `long_ma_rebound`: within a downtrend, a fast moving average
sometimes acts as dynamic resistance - price rallies to or above it and
then closes back below it the next session ("rejected").

# Conditions

```
SMA_fast[t] < SMA_slow[t]              (downtrend intact)
AND Close[t-1] >= SMA_fast[t-1]        (was at/above the MA yesterday)
AND Close[t]   <  SMA_fast[t]          (closed back below it today)
```

Default parameters (`config/settings.yaml: signals.short.ma_rejection`):
`sma_fast = 20`, `sma_slow = 50`.

# Expected Behavior

Symmetric to `long_ma_rebound`'s expected behavior, mirrored in direction.

# Known Risks

Same structural risks as `long_ma_rebound` (single-day confirmation only,
sensitive to exact MA choice), mirrored in direction.

## Phase 6 OOS Evaluation (2026-08-19)

- config_hash: `a9b34ccb6e6e1e1a...`
- data_hash: `05d9d41dbe397257...`
- data range: 2022-01-04 .. 2024-06-28 (4 tickers: 7203/6758/9984/8951)
- walk-forward windows evaluated: 5 (TRAIN=12mo/VAL=3mo/OOS=3mo/STEP=3mo)

### Window-by-window (TRAIN -> VAL -> OOS expectancy, base cost)

| Window | OOS period | TRAIN n / exp | VAL n / exp | OOS n / exp | OOS PF |
|---|---|---|---|---|---|
| 0 | 2023-04-04..2023-07-04 | 10 / -1.67% | 5 / -1.04% | 9 / -0.97% | 0.2388 |
| 1 | 2023-07-04..2023-10-04 | 15 / -1.46% | 9 / -0.97% | 3 / 0.86% | 6.0957 |
| 2 | 2023-10-04..2024-01-04 | 21 / -1.10% | 3 / 0.86% | 4 / 0.65% | 1.5370 |
| 3 | 2024-01-04..2024-04-04 | 21 / -0.45% | 4 / 0.65% | 2 / 0.84% | 7.3877 |
| 4 | 2024-04-04..2024-06-28 | 21 / -0.42% | 2 / 0.84% | 5 / -2.39% | 0.2036 |

### Aggregate OOS (all windows combined, by transaction cost tier)

| Cost tier | n | win_rate | PF | expectancy |
|---|---|---|---|---|
| zero | 23 | 52.17% | 0.7582 | -0.30% |
| low | 23 | 47.83% | 0.6901 | -0.40% |
| base | 23 | 47.83% | 0.5699 | -0.60% |
| high | 23 | 43.48% | 0.3433 | -1.10% |

### Bootstrap (95% CI, base cost, n_resamples=10000, seed=42)

- expectancy: point=-0.60%, CI=[-1.84%, 0.51%]
- profit_factor: point=0.5699, CI=[0.1798, 1.7791]

### Permutation Test (forward_return_5d, n_permutations=10000, seed=43, two-sided)

- observed_mean=0.05%, p_value=0.9622, n_signal=31, n_population=1200

### Consistency and Regime

- windows with OOS PF>1: 3/5
- windows with OOS expectancy>0: 3/5
- by market regime: BULL: n=12, expectancy=-0.64%, NEUTRAL: n=11, expectancy=0.07%

### Decision

**INSUFFICIENT_EVIDENCE** (candidate only - see README "Signal Decision"; not an automatic accept/reject)

Rationale (mechanical, from backtest/decision.py - not a narrative justification): this label reflects whether OOS trade count, aggregate expectancy/PF, the bootstrap CI on expectancy, the permutation p-value, window consistency, and high-cost-tier expectancy jointly cleared the fixed thresholds in backtest/decision.py, none of which were adjusted after seeing this or any other Signal's OOS result.
