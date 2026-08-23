"""Phase V3-4: Market Timing vs Stock-Specific Edge Decomposition.

This package does NOT run a new independent OOS validation. It re-derives
robustness/decomposition analyses on top of Phase V3-3's FROZEN pipeline
(same Feature Registry, Target Registry, Model A/B/C, Hyperparameters,
WFO structure, Q1-Q5/Top-N/Decision Framework - see `v3/validation/
wfo_config.py`, imported here unmodified). The one exception is Model A
on the Primary target (`target_raw_5d`) and the 3 secondary Horizon
targets (10/15/20d raw): `reproduce.py` re-runs `v3.validation.
train_predict.run_model_a_on_window()` (unmodified V3-3 code) against
the SAME Full Universe dataset/windows to regenerate raw per-row OOS
predictions, since Phase V3-3's own JSON report (containing those rows)
is ~4GB and impractical to re-parse. This re-run is verified bit-
identical via `v3.hash`'s code/config/feature/dataset hashes AND an
empirical match against V3-3's own reported pooled Q5-Q1 spread - a
reproducibility check, not a new experiment.

Every statistical primitive with a direct V1/V2/V3-3 analogue (Q1-Q5
bucketing, Rank IC, Top-N, 3-method Bootstrap, Permutation, FDR, Regime/
Year/Event slicing, Concentration, Cost tiers, Sector/Liquidity profiling)
is imported unmodified. Only genuinely new decomposition logic lives here:
market-beta computation, cross-sectional demeaning/requantization,
leave-one-out slicing, Gini/Lorenz concentration, matched control
generation, and the extended Economic Significance / Decision-with-
Robustness-Evidence framework spec sections 3-19 ask for.
"""
