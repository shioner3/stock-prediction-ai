---
signal_name: long_volume_breakout
direction: LONG
status: candidate (not yet backtested/evaluated - Phase 6)
---

# Signal Name

LONG Volume Breakout

# Direction

LONG

# Hypothesis

A single-day price move accompanied by unusually high volume may reflect
new information or a shift in participation that a "quiet" price move of
the same size would not. This is deliberately a DIFFERENT hypothesis
from `long_breakout`: it has no multi-day price-high condition at all, so
it can trigger on a day nowhere near a 20-day high, purely on the
day's own move + volume.

# Conditions

```
return_1d[t] > return_1d_min
AND volume_ratio_20d[t] > volume_ratio_min
```

Default parameters
(`config/settings.yaml: signals.long.volume_breakout`):
`return_1d_min = 0.03`, `volume_ratio_min = 2.0`.

# Expected Behavior

- Should trigger as isolated single-day events far more often than
  `long_breakout`, since it requires no multi-day price structure.
- Should sometimes co-trigger with `long_breakout` (a breakout day is
  often also a big single-day move) but should also trigger on many days
  that `long_breakout` does not (see
  `tests/test_signals_long.py::test_volume_breakout_is_not_identical_condition_to_breakout`).

# Known Risks

- Cannot distinguish a volume spike driven by durable new information
  from one driven by a single large order, an index-rebalance flow, or a
  data anomaly that survived Phase 1's validation.
- `return_1d_min`/`volume_ratio_min` are round-number initial guesses
  and are not normalized by the stock's own typical volatility/volume -
  a stock with persistently high volume variance will trigger this
  Signal more often than one with stable volume, independent of any
  actual "event."

## Phase 6 OOS Evaluation (2026-08-19)

- config_hash: `a9b34ccb6e6e1e1a...`
- data_hash: `05d9d41dbe397257...`
- data range: 2022-01-04 .. 2024-06-28 (4 tickers: 7203/6758/9984/8951)
- walk-forward windows evaluated: 5 (TRAIN=12mo/VAL=3mo/OOS=3mo/STEP=3mo)

### Window-by-window (TRAIN -> VAL -> OOS expectancy, base cost)

| Window | OOS period | TRAIN n / exp | VAL n / exp | OOS n / exp | OOS PF |
|---|---|---|---|---|---|
| 0 | 2023-04-04..2023-07-04 | 5 / 3.99% | 1 / -3.23% | 3 / 1.26% | 1.7391 |
| 1 | 2023-07-04..2023-10-04 | 5 / 0.82% | 3 / 1.26% | 0 / N/A | N/A |
| 2 | 2023-10-04..2024-01-04 | 7 / 0.63% | 0 / N/A | 2 / -0.98% | 0.0000 |
| 3 | 2024-01-04..2024-04-04 | 6 / 0.45% | 2 / -0.98% | 2 / 4.51% | inf |
| 4 | 2024-04-04..2024-06-28 | 6 / -0.23% | 2 / 4.51% | 3 / -2.05% | 0.0000 |

### Aggregate OOS (all windows combined, by transaction cost tier)

| Cost tier | n | win_rate | PF | expectancy |
|---|---|---|---|---|
| zero | 10 | 40.00% | 1.6792 | 0.77% |
| low | 10 | 40.00% | 1.5612 | 0.67% |
| base | 10 | 30.00% | 1.3546 | 0.47% |
| high | 10 | 30.00% | 0.9815 | -0.03% |

### Bootstrap (95% CI, base cost, n_resamples=10000, seed=42)

- expectancy: point=0.47%, CI=[-1.80%, 3.13%]
- profit_factor: point=1.3546, CI=[0.0000, 6.5043]

### Permutation Test (forward_return_5d, n_permutations=10000, seed=43, two-sided)

- observed_mean=2.33%, p_value=0.0548, n_signal=16, n_population=1200

### Consistency and Regime

- windows with OOS PF>1: 2/4
- windows with OOS expectancy>0: 2/4
- by market regime: BULL: n=7, expectancy=1.34%, NEUTRAL: n=3, expectancy=-0.56%

### Decision

**INSUFFICIENT_EVIDENCE** (candidate only - see README "Signal Decision"; not an automatic accept/reject)

Rationale (mechanical, from backtest/decision.py - not a narrative justification): this label reflects whether OOS trade count, aggregate expectancy/PF, the bootstrap CI on expectancy, the permutation p-value, window consistency, and high-cost-tier expectancy jointly cleared the fixed thresholds in backtest/decision.py, none of which were adjusted after seeing this or any other Signal's OOS result.
