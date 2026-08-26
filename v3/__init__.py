"""V3: ML Expected-Value Ranking Engine.

A THIRD, fully independent research line - separate from V1 (hand-crafted
Signals -> Score -> Backtest -> Forward Test, frozen) and V2 (rule-based
cross-sectional Score, `v2/`, frozen). Goal: for every trading day t,
predict a risk-adjusted expected return over 5/10/15/20 trading days for
every stock in the JPX Prime/Standard/Growth Universe using ML, then rank
the Universe cross-sectionally - NOT a signal generator, a ranking engine.

Independence invariants (spec section 1), enforced structurally and
verified every run via `git status` + `v3/hash.py`:
  - V3 never imports anything from `signals/`, `scoring/`, `backtest/`,
    `forward_test/`, `ensemble/` (V1's decision-making layers) or writes
    into any V1-owned directory.
  - V3 MAY import from V1's pure, already-reused-by-V2 computation layers
    (`features/`, `targets/forward_returns.py`, `universe/`) - the same
    "reuse a tested formula via plain import, never re-derive it" pattern
    V2 already established, and the same layers V2 itself imports from.
  - V3 MAY import from V2's already-frozen, generic utilities
    (`v2/ranking/cross_sectional.py`'s day-grouped percentile rank) where
    doing so avoids reimplementing an identical, already-leak-tested
    formula - this is READING a pure function, never writing into `v2/`
    or depending on any V2-3-SPECIFIC finding (spec section 1 rule 9
    explicitly forbids treating V2-3's "short momentum weak + long trend
    intact" pattern specially - V3's Feature Registry includes the
    underlying raw Features that COULD express that pattern, exactly as
    it includes every other raw Feature, and lets the model decide,
    rather than hand-coding that interaction).
  - V3 has its OWN config/code/feature/dataset/model hash namespace
    (`v3/hash.py`), independent of V1's Strategy Hash (`forward_test/
    manifest.py`) and V2's manifest (`v2/manifest.py`).

Phase V3-1 scope (this package, as first built): Dataset construction,
Feature Registry, Target Registry, and the Leakage framework only - NO
model training, NO Full-Universe OOS run. See research/phase_v3_1_report.md.
"""
