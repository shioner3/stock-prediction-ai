"""Strategy Version 2 (V2): Independent Research Ranking Engine.

V2 is a SEPARATE research track from V1 (the frozen `long_oversold_rebound`
Strategy Version 1, its 12 Signals, Score, Backtest, Walk Forward
Validation, and Forward Test Engine). Nothing under this package may
import FROM v1's mutable state (Forward Test logs/portfolio/manifest) or
be imported BY any V1 module - V1 must never depend on V2, and this
package must never write to any V1-owned directory (`data/raw`,
`data/processed`, `data/features`, `data/signals`, `data/scores`,
`data/forward_test`, `data/walk_forward`, `data/phase7`).

V2 DOES reuse several V1 modules by plain import - this is deliberate
(see PhaseV2-1 spec section 3: "importするだけ / wrapperを作る /
adapterを作る") and never modifies the imported V1 code itself:

- features.pipeline.compute_feature_panel() - Momentum/Trend/Volatility/
  Volume/RSI/MACD/Breakout/Pullback/Relative Strength, unmodified.
- targets.forward_returns.compute_forward_returns() - the same Forward
  Return definition Phase 6+ Score Validation already uses, unmodified.
- pipeline.universe_ingest.load_manifest() / storage.parquet_store's
  load_ohlcv/load_feature_panel - reads V1's already-fetched Full
  Universe OHLCV/Feature caches (READ ONLY - V2 never writes into
  V1's data/phase7/* directories).
- scoring.validation.assign_quantile_buckets() - the same Q1-Q5 bucketing
  convention V1's own Score Validation uses, unmodified.

V2 is rule-based / statistical, not ML, and does not place any real
order, connect to a broker, or touch V1's Forward Test state. See
`research/phase_v2_1_report.md` for the full design writeup.
"""
