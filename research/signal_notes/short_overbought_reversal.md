---
signal_name: short_overbought_reversal
direction: SHORT
status: candidate (not yet backtested/evaluated - Phase 6)
---

# Signal Name

SHORT Overbought Reversal

# Direction

SHORT

# Hypothesis

Mirror of `long_oversold_rebound`: after a period registering as
"overbought" by RSI, a down day may mark the start of a short-term
mean-reversion pullback. "RSI is high" is explicitly NOT treated as
"sell" by this project - it is a testable condition, not a conclusion.

# Conditions

```
RSI_period[t] > rsi_min
AND Close[t] < Close[t-1]
```

Default parameters
(`config/settings.yaml: signals.short.overbought_reversal`):
`rsi_period = 14`, `rsi_min = 70.0`.

# Expected Behavior

Symmetric to `long_oversold_rebound`'s expected behavior, mirrored in
direction.

# Known Risks

Same structural risks as `long_oversold_rebound` (high RSI can persist
through a strong sustained uptrend; a single down day is weak
confirmation), mirrored in direction. Historically, momentum-driven
uptrends (which tend to produce sustained high RSI) may behave
differently from momentum-driven downtrends in equities - this asymmetry
is not modeled by treating the two Signals as pure mirrors.

## Phase 6 OOS Evaluation (2026-08-19)

- config_hash: `a9b34ccb6e6e1e1a...`
- data_hash: `05d9d41dbe397257...`
- data range: 2022-01-04 .. 2024-06-28 (4 tickers: 7203/6758/9984/8951)
- walk-forward windows evaluated: 5 (TRAIN=12mo/VAL=3mo/OOS=3mo/STEP=3mo)

### Window-by-window (TRAIN -> VAL -> OOS expectancy, base cost)

| Window | OOS period | TRAIN n / exp | VAL n / exp | OOS n / exp | OOS PF |
|---|---|---|---|---|---|
| 0 | 2023-04-04..2023-07-04 | 2 / -5.67% | 0 / N/A | 4 / -2.57% | 0.3279 |
| 1 | 2023-07-04..2023-10-04 | 2 / -5.67% | 4 / -2.57% | 2 / -0.35% | 0.6863 |
| 2 | 2023-10-04..2024-01-04 | 6 / -3.61% | 2 / -0.35% | 0 / N/A | N/A |
| 3 | 2024-01-04..2024-04-04 | 8 / -2.79% | 0 / N/A | 6 / -0.98% | 0.4491 |
| 4 | 2024-04-04..2024-06-28 | 6 / -1.83% | 6 / -0.98% | 1 / -0.05% | 0.0000 |

### Aggregate OOS (all windows combined, by transaction cost tier)

| Cost tier | n | win_rate | PF | expectancy |
|---|---|---|---|---|
| zero | 13 | 46.15% | 0.5011 | -1.00% |
| low | 13 | 46.15% | 0.4656 | -1.10% |
| base | 13 | 38.46% | 0.4011 | -1.30% |
| high | 13 | 30.77% | 0.2825 | -1.80% |

### Bootstrap (95% CI, base cost, n_resamples=10000, seed=42)

- expectancy: point=-1.30%, CI=[-3.56%, 0.75%]
- profit_factor: point=0.4011, CI=[0.0320, 1.8165]

### Permutation Test (forward_return_5d, n_permutations=10000, seed=43, two-sided)

- observed_mean=1.99%, p_value=0.0548, n_signal=25, n_population=1200

### Consistency and Regime

- windows with OOS PF>1: 0/4
- windows with OOS expectancy>0: 0/4
- by market regime: BULL: n=10, expectancy=-1.32%, NEUTRAL: n=3, expectancy=0.05%

### Decision

**INSUFFICIENT_EVIDENCE** (candidate only - see README "Signal Decision"; not an automatic accept/reject)

Rationale (mechanical, from backtest/decision.py - not a narrative justification): this label reflects whether OOS trade count, aggregate expectancy/PF, the bootstrap CI on expectancy, the permutation p-value, window consistency, and high-cost-tier expectancy jointly cleared the fixed thresholds in backtest/decision.py, none of which were adjusted after seeing this or any other Signal's OOS result.
