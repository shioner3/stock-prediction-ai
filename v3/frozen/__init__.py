"""Phase V3-7: Frozen Model Creation + Forward Observation.

Unlike V3-3/V3-4/V3-5 (Walk-Forward research: train fresh per rolling
window, predict a KNOWN historical OOS period, discard the model), this
package trains and PERSISTS 16 models exactly once - 4 Target definitions
(Raw / TOPIX-relative / Beta-adjusted Residual / Sector-relative, all 4
already-frozen V3-1/V3-5 definitions) x 4 Horizons (5/10/15/20d), each
trained on ALL available data through T0 = 2026-08-20 - so they can be
applied, unchanged, to genuinely NEW data arriving after T0.

No canonical/primary model is designated among the 16 (explicit user
decision for this Phase): every downstream consumer (observation logging,
leakage tests, GitHub Actions) treats all 16 identically. Reuses V1/V3-1
through V3-5's frozen Feature Registry, Target definitions, Hyperparameters,
random seed, and beta/residual-target computation code UNCHANGED - only
the "train once, persist, never retrain" packaging and the daily
Observation pipeline are new.
"""
