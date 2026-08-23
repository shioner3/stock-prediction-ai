"""Phase V2-2: Full Universe OOS Validation of the Phase V2-1 Swing
Candidate Ranking Engine's Score.

Every module here is NEW code added on top of the frozen Phase V2-1
engine (v2/__init__.py, v2/candidate.py, v2/config/, v2/features_adapter.py,
v2/manifest.py, v2/pipeline.py, v2/ranking/, v2/stats.py,
v2/targets_adapter.py - see FROZEN_V2_1_FILES in
v2/validation/hash_check.py for the exact file list this package treats
as immutable). Nothing here modifies a V2-1 file; V2-1's Score
composition, weights, Feature definitions, Candidate thresholds,
Universe filter, and outlier-exclusion rule are all imported and reused
UNCHANGED (spec section 2's "single source of truth" instruction).

Interpretation note on the spec's Rule 1 ("V2からV1へのimportは禁止"):
read literally this would contradict spec section 12 ("既存のV2/V1で
定義済みのMarket Regimeがある場合は、それをそのまま使用する") and
Phase V2-1's own already-accepted architecture (which imports V1's
compute_feature_panel/compute_forward_returns/assign_quantile_buckets).
This package follows V2-1's precedent: importing V1's PURE, stateless
statistical/classification primitives (Market Regime, Day Cluster/Block
Bootstrap, Permutation, FDR, Transaction Cost, Data Integrity coverage
checks) is treated as intended reuse - never a V1 Signal/Score/Backtest
change, and V1 is never modified or imported BY. The invariant that
actually matters (V1 completely untouched) is verified directly via
`git status`/`git diff` at the end of this Phase, not by refusing every
V1 import.
"""
