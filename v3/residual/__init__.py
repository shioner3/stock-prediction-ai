"""Phase V3-5: Stock-Specific Residual ML Validation.

Answers a DIFFERENT question from Phase V3-4: V3-4 held the Primary
Model A's (trained on `target_raw_5d`) ranking FIXED and only varied
which return column evaluated it. V3-5 instead RETRAINS Model A (frozen
V3-2 Hyperparameters/seed, frozen V3-3 WFO structure - `v3.validation.
train_predict.run_model_a_on_window()`, unmodified) on 3 market-
neutralized TARGETS - so the model's own cross-sectional RANKING, not
just the outcome it's scored against, changes. If a model trained
directly to predict market-neutral returns can still rank stocks with a
robust positive edge, that is stronger evidence of genuine stock-
selection ability than anything V3-4 alone could show.

4 Target definitions, kept side by side (never collapsed to "the best
one"): A=Raw (`target_raw_*d`, reused UNCHANGED from V3-3/V3-4's own
saved predictions - no retraining), B=TOPIX-relative (`target_topix_
relative_*d`, an ALREADY-COMPUTED frozen V3-1 Target Registry column for
all 4 horizons - no new formula), C=Beta-adjusted residual (NEW, this
Phase's own `targets.py`, reusing V3-4's `beta.compute_rolling_beta()`
unchanged), D=Sector-relative (NEW, same file, same day/sector-mean
pattern V3-4's `market_decomposition.py` already established and fixed).

Every statistical primitive is reused unmodified from V1/V2/V3-3/V3-4
(Q1-Q5, Rank IC, Top-N, Bootstrap, Permutation, FDR, Regime/Event/Year/
Concentration, Matched Control - the last extended with a
backward-compatible `outcome_cols` parameter, not rewritten). Only the
Target construction, the leakage re-check for that construction, and the
new 4-way Edge Classification are genuinely new code, isolated entirely
under this package - V1/V2/V3-1/V3-2/V3-3/V3-4 are not modified.
"""
