---
signal_name: long_breakout
direction: LONG
status: candidate (not yet backtested/evaluated - Phase 6)
---

# Signal Name

LONG Breakout

# Direction

LONG

# Hypothesis

A stock that closes above its own N-day price ceiling, on volume well
above its recent average, may be starting a new advance rather than
continuing to trade in a range. This is a well-known pattern in
technical analysis literature; whether it holds up out-of-sample for
this universe and period is an open question this project exists to
test, not an assumption.

# Conditions

```
Close[t] > high_Nd[t]                  (high_Nd = highest Close over the
                                         N sessions strictly before t)
AND volume_ratio_20d[t] > volume_multiple
```

Default parameters (`config/settings.yaml: signals.long.breakout`):
`lookback = 20`, `volume_multiple = 1.5`.

Implementation: `signals/long/breakout.py`. Reuses `features/breakout.py`'s
`high_Nd` and `features/volume.py`'s `volume_ratio_20d` - no rolling
calculation is reimplemented in the Signal layer.

# Expected Behavior

- Should trigger relatively rarely (both a price and a volume condition
  must hold simultaneously) and should be more common in trending/volatile
  markets than in quiet, range-bound periods.
- Should almost never trigger on two consecutive days for the same stock
  unless the breakout continues with sustained volume (each day the price
  ceiling itself moves up, since `high_Nd` excludes the trigger day).

# Known Risks

- Breakouts are a widely-known pattern; if many participants react
  simultaneously, the historically favorable follow-through (if any) may
  not persist going forward (crowding).
- Volume spikes can be driven by one-off news events unrelated to a
  durable trend change - the Signal cannot distinguish "breakout on
  accumulation" from "breakout on a one-day news spike."
- `high_Nd` is Close-based (not High-based), per the Feature layer's
  definition; a stock with a large intraday wick above a prior high but a
  Close below it will not trigger here.

## Phase 6 OOS Evaluation (2026-08-19)

- config_hash: `a9b34ccb6e6e1e1a...`
- data_hash: `05d9d41dbe397257...`
- data range: 2022-01-04 .. 2024-06-28 (4 tickers: 7203/6758/9984/8951)
- walk-forward windows evaluated: 5 (TRAIN=12mo/VAL=3mo/OOS=3mo/STEP=3mo)

### Window-by-window (TRAIN -> VAL -> OOS expectancy, base cost)

| Window | OOS period | TRAIN n / exp | VAL n / exp | OOS n / exp | OOS PF |
|---|---|---|---|---|---|
| 0 | 2023-04-04..2023-07-04 | 9 / -1.29% | 1 / -3.23% | 10 / -0.36% | 0.7824 |
| 1 | 2023-07-04..2023-10-04 | 8 / -1.91% | 10 / -0.36% | 4 / 0.28% | 1.1791 |
| 2 | 2023-10-04..2024-01-04 | 17 / -1.06% | 4 / 0.28% | 5 / -1.40% | 0.0000 |
| 3 | 2024-01-04..2024-04-04 | 19 / -0.53% | 5 / -1.40% | 5 / 2.48% | inf |
| 4 | 2024-04-04..2024-06-28 | 20 / -0.64% | 5 / 2.48% | 4 / 0.81% | 1.9937 |

### Aggregate OOS (all windows combined, by transaction cost tier)

| Cost tier | n | win_rate | PF | expectancy |
|---|---|---|---|---|
| zero | 28 | 46.43% | 1.5144 | 0.52% |
| low | 28 | 42.86% | 1.3943 | 0.42% |
| base | 28 | 39.29% | 1.1860 | 0.22% |
| high | 28 | 39.29% | 0.8121 | -0.28% |

### Bootstrap (95% CI, base cost, n_resamples=10000, seed=42)

- expectancy: point=0.22%, CI=[-0.96%, 1.50%]
- profit_factor: point=1.1860, CI=[0.4168, 3.0021]

### Permutation Test (forward_return_5d, n_permutations=10000, seed=43, two-sided)

- observed_mean=0.82%, p_value=0.3813, n_signal=51, n_population=1200

### Consistency and Regime

- windows with OOS PF>1: 3/5
- windows with OOS expectancy>0: 3/5
- by market regime: BULL: n=16, expectancy=0.88%, NEUTRAL: n=12, expectancy=0.04%

### Decision

**INSUFFICIENT_EVIDENCE** (candidate only - see README "Signal Decision"; not an automatic accept/reject)

Rationale (mechanical, from backtest/decision.py - not a narrative justification): this label reflects whether OOS trade count, aggregate expectancy/PF, the bootstrap CI on expectancy, the permutation p-value, window consistency, and high-cost-tier expectancy jointly cleared the fixed thresholds in backtest/decision.py, none of which were adjusted after seeing this or any other Signal's OOS result.
