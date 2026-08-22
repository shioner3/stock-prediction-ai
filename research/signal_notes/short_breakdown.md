---
signal_name: short_breakdown
direction: SHORT
status: candidate (not yet backtested/evaluated - Phase 6)
---

# Signal Name

SHORT Breakdown

# Direction

SHORT

# Hypothesis

Mirror of `long_breakout`: a stock that closes below its own N-day price
floor, on volume well above its recent average, may be starting a new
decline rather than continuing to trade in a range.

# Conditions

```
Close[t] < low_Nd[t]                   (low_Nd = lowest Close over the
                                         N sessions strictly before t)
AND volume_ratio_20d[t] > volume_multiple
```

Default parameters (`config/settings.yaml: signals.short.breakdown`):
`lookback = 20`, `volume_multiple = 1.5`.

Implementation note: `low_Nd` did not exist before Phase 4 - Phase 2 only
built `high_Nd` for LONG Breakout. It was added to
`features/breakout.py` in Phase 4 specifically so this Signal reads from
the Feature layer instead of recomputing a rolling min inline - see that
module's docstring and README's Phase 4 notes.

# Expected Behavior

Symmetric to `long_breakout`'s expected behavior, mirrored in direction.

# Known Risks

Same structural risks as `long_breakout` (crowding, news-driven volume
spikes, Close-based rather than Low-based comparison), mirrored in
direction. Additionally: short selling in this market carries borrow-
availability and cost considerations this project does not model at all
(Phase 4 has no position sizing or execution-cost model of any kind).

## Phase 6 OOS Evaluation (2026-08-19)

- config_hash: `a9b34ccb6e6e1e1a...`
- data_hash: `05d9d41dbe397257...`
- data range: 2022-01-04 .. 2024-06-28 (4 tickers: 7203/6758/9984/8951)
- walk-forward windows evaluated: 5 (TRAIN=12mo/VAL=3mo/OOS=3mo/STEP=3mo)

### Window-by-window (TRAIN -> VAL -> OOS expectancy, base cost)

| Window | OOS period | TRAIN n / exp | VAL n / exp | OOS n / exp | OOS PF |
|---|---|---|---|---|---|
| 0 | 2023-04-04..2023-07-04 | 19 / -1.49% | 4 / 1.78% | 1 / -2.39% | 0.0000 |
| 1 | 2023-07-04..2023-10-04 | 21 / -0.29% | 1 / -2.39% | 2 / 1.94% | inf |
| 2 | 2023-10-04..2024-01-04 | 16 / 0.35% | 2 / 1.94% | 3 / -1.17% | 0.4946 |
| 3 | 2024-01-04..2024-04-04 | 10 / 0.81% | 3 / -1.17% | 4 / -0.50% | 0.5367 |
| 4 | 2024-04-04..2024-06-28 | 10 / 0.51% | 4 / -0.50% | 5 / -2.95% | 0.0000 |

### Aggregate OOS (all windows combined, by transaction cost tier)

| Cost tier | n | win_rate | PF | expectancy |
|---|---|---|---|---|
| zero | 15 | 26.67% | 0.4307 | -0.95% |
| low | 15 | 26.67% | 0.3973 | -1.05% |
| base | 15 | 26.67% | 0.3383 | -1.25% |
| high | 15 | 26.67% | 0.2242 | -1.75% |

### Bootstrap (95% CI, base cost, n_resamples=10000, seed=42)

- expectancy: point=-1.25%, CI=[-2.71%, 0.14%]
- profit_factor: point=0.3383, CI=[0.0527, 1.1372]

### Permutation Test (forward_return_5d, n_permutations=10000, seed=43, two-sided)

- observed_mean=1.75%, p_value=0.1256, n_signal=18, n_population=1200

### Consistency and Regime

- windows with OOS PF>1: 1/5
- windows with OOS expectancy>0: 1/5
- by market regime: BULL: n=6, expectancy=-1.39%, NEUTRAL: n=9, expectancy=-0.66%

### Decision

**INSUFFICIENT_EVIDENCE** (candidate only - see README "Signal Decision"; not an automatic accept/reject)

Rationale (mechanical, from backtest/decision.py - not a narrative justification): this label reflects whether OOS trade count, aggregate expectancy/PF, the bootstrap CI on expectancy, the permutation p-value, window consistency, and high-cost-tier expectancy jointly cleared the fixed thresholds in backtest/decision.py, none of which were adjusted after seeing this or any other Signal's OOS result.
