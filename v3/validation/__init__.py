"""Phase V3-3: Full Universe Walk-Forward OOS Validation of the FROZEN
Phase V3-1/V3-2 Feature Registry, Target Registry, and Model A/B/C
structures. This package only EVALUATES those frozen components against
Full Universe data across multiple time-ordered OOS windows - it never
adds a Feature, changes a Target formula, or adjusts a Hyperparameter
(spec section 1/28).

Reuses V1's Walk Forward window generator (`backtest/walk_forward.py::
generate_windows()`, unmodified) rather than re-deriving window
arithmetic - the same TRAIN/VALIDATION/OOS three-segment, calendar-month,
rolling-window structure V1's own Phase 6 WFO already uses (spec section
7's explicit "既存V1/V2のWFO思想と整合する"). The middle "VALIDATION"
segment plays the role of this Phase's EMBARGO (data used for neither
training nor testing) - V3-3 does no hyperparameter search, so there is
nothing else for that segment to do; this interpretation is documented
here rather than silently assumed.
"""
