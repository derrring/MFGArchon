Corrected `_picard_verdict`'s docstring in `scripts/capability_matrix.py`, which still opened with
"Deliberately NOT part of any cell's `ok`". That has been false since #1893 routed **five**
`_solved(art)` conjuncts through the verdicts — four `ok = ` lines plus `fvm_vs_fdm/agreement`'s
inline conditional, the cell that gets forgotten. Struck in place, along with two further copies of
the same claim: `tests/unit/test_capability_matrix.py`'s "recorded, not gated … this test is the
only oracle over the field", every clause of which is now false, and the unreleased
`1871-capability-picard-verdict` fragment, which said the opposite of its own sibling. Also fixed a
number that had rotted: `sl_linear_2d` converges at **iteration 32**, not "35 sweeps" — 35 is the
budget. No line numbers are cited anywhere; a docstring above the lines it names renumbers them
whenever it is edited, so it carries a grep instead (#1871).
