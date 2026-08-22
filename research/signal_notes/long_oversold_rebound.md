---
signal_name: long_oversold_rebound
direction: LONG
status: candidate (not yet backtested/evaluated - Phase 6)
---

# Signal Name

LONG Oversold Rebound

# Direction

LONG

# Hypothesis

After a period registering as "oversold" by RSI, an up day may mark the
start of a short-term mean-reversion bounce. This is stated here purely
as a testable condition - "RSI is low" is explicitly NOT treated as
"buy" by this project; whether a rebound on a low-RSI day tends to
continue is exactly what Phase 6's backtest exists to check.

# Conditions

```
RSI_period[t] < rsi_max
AND Close[t] > Close[t-1]
```

Default parameters
(`config/settings.yaml: signals.long.oversold_rebound`):
`rsi_period = 14`, `rsi_max = 30.0`.

# Expected Behavior

- Should trigger more often in declining or high-volatility markets,
  since RSI < 30 requires a recent run of losses to have dominated gains
  (see `features/indicators.py`'s Wilder RSI definition).
- The up-day requirement means this Signal fires on the FIRST up day
  after (or during) an oversold reading, not on every day RSI stays low.

# Known Risks

- Low RSI in a strong, sustained downtrend does not mean a reversal is
  near - "oversold" can remain oversold for a long time (the classic
  mean-reversion trap).
- A single up day is a very weak confirmation signal; this Signal has no
  mechanism to distinguish a genuine reversal from one random up-tick in
  an ongoing decline.

## Phase 6 OOS Evaluation (2026-08-19)

- config_hash: `a9b34ccb6e6e1e1a...`
- data_hash: `05d9d41dbe397257...`
- data range: 2022-01-04 .. 2024-06-28 (4 tickers: 7203/6758/9984/8951)
- walk-forward windows evaluated: 5 (TRAIN=12mo/VAL=3mo/OOS=3mo/STEP=3mo)

### Window-by-window (TRAIN -> VAL -> OOS expectancy, base cost)

| Window | OOS period | TRAIN n / exp | VAL n / exp | OOS n / exp | OOS PF |
|---|---|---|---|---|---|
| 0 | 2023-04-04..2023-07-04 | 7 / 3.20% | 2 / 2.76% | 0 / N/A | N/A |
| 1 | 2023-07-04..2023-10-04 | 7 / 2.46% | 0 / N/A | 0 / N/A | N/A |
| 2 | 2023-10-04..2024-01-04 | 7 / 2.46% | 0 / N/A | 0 / N/A | N/A |
| 3 | 2024-01-04..2024-04-04 | 5 / 1.76% | 0 / N/A | 0 / N/A | N/A |
| 4 | 2024-04-04..2024-06-28 | 2 / 2.76% | 0 / N/A | 2 / 4.42% | inf |

### Aggregate OOS (all windows combined, by transaction cost tier)

| Cost tier | n | win_rate | PF | expectancy |
|---|---|---|---|---|
| zero | 2 | 100.00% | inf | 4.72% |
| low | 2 | 100.00% | inf | 4.62% |
| base | 2 | 100.00% | inf | 4.42% |
| high | 2 | 100.00% | inf | 3.92% |

### Bootstrap (95% CI, base cost, n_resamples=10000, seed=42)

- expectancy: point=4.42%, CI=[1.98%, 6.85%]
- profit_factor: point=inf, CI=[N/A, N/A]

### Permutation Test (forward_return_5d, n_permutations=10000, seed=43, two-sided)

- observed_mean=5.80%, p_value=0.0513, n_signal=2, n_population=1200

### Consistency and Regime

- windows with OOS PF>1: 1/1
- windows with OOS expectancy>0: 1/1
- by market regime: BULL: n=1, expectancy=2.28%, NEUTRAL: n=1, expectancy=7.15%

### Decision

**INSUFFICIENT_EVIDENCE** (candidate only - see README "Signal Decision"; not an automatic accept/reject)

Rationale (mechanical, from backtest/decision.py - not a narrative justification): this label reflects whether OOS trade count, aggregate expectancy/PF, the bootstrap CI on expectancy, the permutation p-value, window consistency, and high-cost-tier expectancy jointly cleared the fixed thresholds in backtest/decision.py, none of which were adjusted after seeing this or any other Signal's OOS result.
