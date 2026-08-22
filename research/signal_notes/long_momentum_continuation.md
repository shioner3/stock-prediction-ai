---
signal_name: long_momentum_continuation
direction: LONG
status: candidate (not yet backtested/evaluated - Phase 6)
---

# Signal Name

LONG Momentum Continuation

# Direction

LONG

# Hypothesis

This Signal tests "the trend continues," not "the stock just jumped" -
it requires both short-term (5-day) and medium-term (20-day) momentum to
be positive simultaneously, plus price above its 20-day MA, rather than
reacting to any single day's move. Whether multi-horizon-confirmed
momentum tends to persist further is the question left to the backtest.

# Conditions

```
return_5d[t]  > return_5d_min
AND return_20d[t] > return_20d_min
AND Close[t] > SMA_period[t]
```

Default parameters
(`config/settings.yaml: signals.long.momentum_continuation`):
`return_5d_min = 0.03`, `return_20d_min = 0.0`, `sma_period = 20`.

# Expected Behavior

- Should trigger in clusters (consecutive days) more often than as
  isolated single-day events, since `return_5d`/`return_20d` change
  gradually.
- Should overlap more with `long_breakout` than with `long_pullback`
  (this Signal and Breakout both describe upward-moving conditions; this
  one and Pullback describe opposite price states, so simultaneous
  triggers on the same day for the same stock should be rare, though not
  impossible depending on exact thresholds).

# Known Risks

- Two return-based conditions plus an MA filter can all be satisfied by
  a stock that is simply in a long, gradual, low-volatility rise -
  "continuation" here is a structural condition, not a claim that the
  move is unusual or exploitable.
- `return_5d_min = 0.03` and `return_20d_min = 0.0` are round-number
  initial guesses, not fit to this universe's typical volatility - a
  low-volatility large-cap and a high-volatility small-cap will clear
  this bar at very different frequencies.

## Phase 6 OOS Evaluation (2026-08-19)

- config_hash: `a9b34ccb6e6e1e1a...`
- data_hash: `05d9d41dbe397257...`
- data range: 2022-01-04 .. 2024-06-28 (4 tickers: 7203/6758/9984/8951)
- walk-forward windows evaluated: 5 (TRAIN=12mo/VAL=3mo/OOS=3mo/STEP=3mo)

### Window-by-window (TRAIN -> VAL -> OOS expectancy, base cost)

| Window | OOS period | TRAIN n / exp | VAL n / exp | OOS n / exp | OOS PF |
|---|---|---|---|---|---|
| 0 | 2023-04-04..2023-07-04 | 40 / -0.32% | 10 / -2.27% | 17 / 2.94% | 5.2450 |
| 1 | 2023-07-04..2023-10-04 | 43 / -1.05% | 17 / 2.94% | 13 / -0.70% | 0.5872 |
| 2 | 2023-10-04..2024-01-04 | 49 / 0.63% | 13 / -0.70% | 11 / -1.45% | 0.3399 |
| 3 | 2024-01-04..2024-04-04 | 51 / 0.47% | 11 / -1.45% | 19 / -0.45% | 0.7164 |
| 4 | 2024-04-04..2024-06-28 | 51 / 0.04% | 19 / -0.45% | 9 / 0.83% | 1.5637 |

### Aggregate OOS (all windows combined, by transaction cost tier)

| Cost tier | n | win_rate | PF | expectancy |
|---|---|---|---|---|
| zero | 69 | 53.62% | 1.4909 | 0.65% |
| low | 69 | 52.17% | 1.4007 | 0.55% |
| base | 69 | 49.28% | 1.2373 | 0.35% |
| high | 69 | 47.83% | 0.9117 | -0.15% |

### Bootstrap (95% CI, base cost, n_resamples=10000, seed=42)

- expectancy: point=0.35%, CI=[-0.65%, 1.37%]
- profit_factor: point=1.2373, CI=[0.6575, 2.2868]

### Permutation Test (forward_return_5d, n_permutations=10000, seed=43, two-sided)

- observed_mean=0.39%, p_value=0.877, n_signal=233, n_population=1200

### Consistency and Regime

- windows with OOS PF>1: 2/5
- windows with OOS expectancy>0: 2/5
- by market regime: BULL: n=49, expectancy=0.60%, NEUTRAL: n=20, expectancy=0.77%

### Decision

**REJECT** (candidate only - see README "Signal Decision"; not an automatic accept/reject)

Rationale (mechanical, from backtest/decision.py - not a narrative justification): this label reflects whether OOS trade count, aggregate expectancy/PF, the bootstrap CI on expectancy, the permutation p-value, window consistency, and high-cost-tier expectancy jointly cleared the fixed thresholds in backtest/decision.py, none of which were adjusted after seeing this or any other Signal's OOS result.
