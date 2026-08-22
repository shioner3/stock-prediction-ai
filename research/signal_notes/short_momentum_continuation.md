---
signal_name: short_momentum_continuation
direction: SHORT
status: candidate (not yet backtested/evaluated - Phase 6)
---

# Signal Name

SHORT Momentum Continuation

# Direction

SHORT

# Hypothesis

Mirror of `long_momentum_continuation`: tests "the decline continues,"
requiring both short-term and medium-term negative momentum plus price
below its 20-day MA, rather than reacting to a single day's move.

# Conditions

```
return_5d[t]  < return_5d_max
AND return_20d[t] < return_20d_max
AND Close[t] < SMA_period[t]
```

Default parameters
(`config/settings.yaml: signals.short.momentum_continuation`):
`return_5d_max = -0.03`, `return_20d_max = 0.0`, `sma_period = 20`.

# Expected Behavior

Symmetric to `long_momentum_continuation`'s expected behavior, mirrored
in direction.

# Known Risks

Same structural risks as `long_momentum_continuation` (round-number,
unvalidated thresholds not normalized by per-stock volatility), mirrored
in direction.

## Phase 6 OOS Evaluation (2026-08-19)

- config_hash: `a9b34ccb6e6e1e1a...`
- data_hash: `05d9d41dbe397257...`
- data range: 2022-01-04 .. 2024-06-28 (4 tickers: 7203/6758/9984/8951)
- walk-forward windows evaluated: 5 (TRAIN=12mo/VAL=3mo/OOS=3mo/STEP=3mo)

### Window-by-window (TRAIN -> VAL -> OOS expectancy, base cost)

| Window | OOS period | TRAIN n / exp | VAL n / exp | OOS n / exp | OOS PF |
|---|---|---|---|---|---|
| 0 | 2023-04-04..2023-07-04 | 49 / -1.43% | 9 / 0.83% | 5 / -1.61% | 0.1665 |
| 1 | 2023-07-04..2023-10-04 | 46 / -0.69% | 5 / -1.61% | 11 / -1.78% | 0.3245 |
| 2 | 2023-10-04..2024-01-04 | 40 / -0.52% | 8 / -0.32% | 12 / -1.94% | 0.3450 |
| 3 | 2024-01-04..2024-04-04 | 31 / -0.29% | 12 / -1.94% | 7 / -1.06% | 0.4019 |
| 4 | 2024-04-04..2024-06-28 | 34 / -0.78% | 7 / -1.06% | 11 / -0.74% | 0.6302 |

### Aggregate OOS (all windows combined, by transaction cost tier)

| Cost tier | n | win_rate | PF | expectancy |
|---|---|---|---|---|
| zero | 43 | 37.21% | 0.5619 | -0.85% |
| low | 43 | 37.21% | 0.5257 | -0.95% |
| base | 43 | 34.88% | 0.4602 | -1.15% |
| high | 43 | 34.88% | 0.3282 | -1.65% |

### Bootstrap (95% CI, base cost, n_resamples=10000, seed=42)

- expectancy: point=-1.15%, CI=[-2.32%, 0.00%]
- profit_factor: point=0.4602, CI=[0.1882, 1.0025]

### Permutation Test (forward_return_5d, n_permutations=10000, seed=43, two-sided)

- observed_mean=1.56%, p_value=0.0056, n_signal=132, n_population=1200

### Consistency and Regime

- windows with OOS PF>1: 0/5
- windows with OOS expectancy>0: 0/5
- by market regime: BULL: n=18, expectancy=-0.28%, NEUTRAL: n=25, expectancy=-1.26%

### Decision

**REJECT** (candidate only - see README "Signal Decision"; not an automatic accept/reject)

Rationale (mechanical, from backtest/decision.py - not a narrative justification): this label reflects whether OOS trade count, aggregate expectancy/PF, the bootstrap CI on expectancy, the permutation p-value, window consistency, and high-cost-tier expectancy jointly cleared the fixed thresholds in backtest/decision.py, none of which were adjusted after seeing this or any other Signal's OOS result.
