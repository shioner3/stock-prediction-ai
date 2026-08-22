"""Phase 10: Frozen Strategy Forward Test - Paper Portfolio infrastructure
for long_oversold_rebound, running forward from T0 with no strategy
tuning allowed. See forward_test/manifest.py, forward_test/portfolio.py,
forward_test/integrity.py, and pipeline/run_forward_test.py.
"""

from pathlib import Path

# Default working root for all Forward Test artifacts - deliberately
# distinct from data/ (Phase 1-6.5) and data/phase7/ (Phase 7-9), so
# Forward Test never reads or overwrites any historical phase's data
# (spec section 20 item 10: historical result isolation).
FORWARD_TEST_ROOT = Path("data/forward_test")
