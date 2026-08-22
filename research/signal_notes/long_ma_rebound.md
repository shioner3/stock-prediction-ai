---
signal_name: long_ma_rebound
direction: LONG
status: candidate (not yet backtested/evaluated - Phase 6)
---

# Signal Name

LONG MA Rebound

# Direction

LONG

# Hypothesis

Within an uptrend, a fast moving average sometimes acts as dynamic
support: price dips to or below it and then closes back above it the
next session. Whether that transition day is followed by continuation
of the uptrend (rather than a further breakdown) is the open question.

# Conditions

```
SMA_fast[t] > SMA_slow[t]              (uptrend intact)
AND Close[t-1] <= SMA_fast[t-1]        (was at/below the MA yesterday)
AND Close[t]   >  SMA_fast[t]          (closed back above it today)
```

Default parameters (`config/settings.yaml: signals.long.ma_rebound`):
`sma_fast = 20`, `sma_slow = 50`.

"Rebound" is defined strictly as this yesterday-below / today-above
transition, deliberately narrower than "price is above its MA" (which
would match every day of an uptrend, not just the rebound day itself).

# Expected Behavior

- Should trigger on isolated, single days (by construction: the
  yesterday<=MA condition means it cannot re-trigger on the day right
  after it already fired, unless price dips back below the MA again
  first).
- Frequency should be inversely related to trend smoothness - a very
  smooth uptrend that never touches its 20-day MA will never trigger
  this Signal.

# Known Risks

- A single close back above the MA does not distinguish a genuine
  rebound from a brief overshoot that reverses the next day - this
  Signal has no confirmation window.
- Highly sensitive to the exact `sma_fast` choice; a stock oscillating
  right around its 20-day MA could trigger this Signal repeatedly over a
  short span, none of which may be meaningful moves.

## Phase 6 OOS Evaluation (2026-08-19)

- config_hash: `a9b34ccb6e6e1e1a...`
- data_hash: `05d9d41dbe397257...`
- data range: 2022-01-04 .. 2024-06-28 (4 tickers: 7203/6758/9984/8951)
- walk-forward windows evaluated: 5 (TRAIN=12mo/VAL=3mo/OOS=3mo/STEP=3mo)

### Window-by-window (TRAIN -> VAL -> OOS expectancy, base cost)

| Window | OOS period | TRAIN n / exp | VAL n / exp | OOS n / exp | OOS PF |
|---|---|---|---|---|---|
| 0 | 2023-04-04..2023-07-04 | 20 / -1.79% | 5 / -1.91% | 3 / 4.88% | 6.6238 |
| 1 | 2023-07-04..2023-10-04 | 25 / -1.81% | 3 / 4.88% | 6 / -0.35% | 0.7580 |
| 2 | 2023-10-04..2024-01-04 | 20 / -0.91% | 6 / -0.35% | 11 / -0.05% | 0.9614 |
| 3 | 2024-01-04..2024-04-04 | 17 / -0.43% | 11 / -0.05% | 7 / 4.55% | 3.4031 |
| 4 | 2024-04-04..2024-06-28 | 25 / 0.10% | 6 / 4.55% | 5 / -0.66% | 0.5830 |

### Aggregate OOS (all windows combined, by transaction cost tier)

| Cost tier | n | win_rate | PF | expectancy |
|---|---|---|---|---|
| zero | 31 | 48.39% | 2.0865 | 1.46% |
| low | 31 | 48.39% | 1.9748 | 1.36% |
| base | 31 | 48.39% | 1.7744 | 1.16% |
| high | 31 | 45.16% | 1.3754 | 0.66% |

### Bootstrap (95% CI, base cost, n_resamples=10000, seed=42)

- expectancy: point=1.16%, CI=[-0.95%, 3.69%]
- profit_factor: point=1.7744, CI=[0.5369, 5.0868]

### Permutation Test (forward_return_5d, n_permutations=10000, seed=43, two-sided)

- observed_mean=1.02%, p_value=0.2764, n_signal=42, n_population=1200

### Consistency and Regime

- windows with OOS PF>1: 2/5
- windows with OOS expectancy>0: 2/5
- by market regime: BULL: n=19, expectancy=1.73%, NEUTRAL: n=12, expectancy=1.03%

### Decision

**INSUFFICIENT_EVIDENCE** (candidate only - see README "Signal Decision"; not an automatic accept/reject)

Rationale (mechanical, from backtest/decision.py - not a narrative justification): this label reflects whether OOS trade count, aggregate expectancy/PF, the bootstrap CI on expectancy, the permutation p-value, window consistency, and high-cost-tier expectancy jointly cleared the fixed thresholds in backtest/decision.py, none of which were adjusted after seeing this or any other Signal's OOS result.
