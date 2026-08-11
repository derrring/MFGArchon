- **A capability cell now requires the coupled iteration to have converged** (Issue #1891).
  `picard_converged` was recorded but left out of the verdict by #1871, on the stated ground that
  gating it "turns three long-standing greens red" — a transition cost, not a correctness argument.
  Two things settle it now.
  First, #1865 had already required it in as many words: *"any restatement of the 2-D fixture must
  assert convergence, or the cell measures whether the FP time-stepper conserves mass (which it does
  regardless) rather than whether the configuration solves."*
  Second: a solve that has not reached a fixed point has not solved, which is what a cell claims to
  certify.
  *An earlier version of this entry argued that `picard_converged` is "the only recorded field
  measured to depend on the coupling", with `mass_t0`, `max_rel_drift` and `min_density` "not moving
  at all". That was asserted from measuring one field. A full-artifact diff over the mutation family
  shows `min_density` moving in **four** cells against this field's two, and `max_drift`, `mass_max`,
  `max_rel_drift`, `rel_l2_*` and `worst` moving as well. The narrower true statement: of the fields
  a verdict reads, only `worst` — in the agreement cell, at 10x — ever crosses its own threshold, so
  four of the five coupled cells had a verdict no coupling mutation could move. Gating does not make
  those four coupling-sensitive; it makes three of them honestly red. The coupling-sensitivity
  argument was wrong and is withdrawn; the correctness argument stands on its own.*
- **Three cells go PASS -> FAIL, and they are honestly red.** `fdm_upwind` (5 sweeps),
  `sl_linear` (5) and `sl_linear_2d` (3, its budget) all record `picard_converged: false`. Each
  carries an `intended` note with what is known: `fdm_upwind` does not converge at 100, 200 or 400
  sweeps either and its inner half is #1878; `sl_linear` fails on all four criterion arms with the
  density arm not descending (#1880); `sl_linear_2d` exhausts a 3-sweep budget, which is also why it
  is the one cell whose convergence flag cannot move under the coupling mutations. `--check-baseline`
  records the change in the same commit and reports **0 of 9 non-PASS cells unexplained**.
- **The known-inert list falls from four to one.** Only `fvm_muscl/mass_conservation` remains: it
  converges (18 sweeps) and its verdict still survives every member of the family. The three that
  left are no longer PASS, so they are never mutated and cannot be judged — reported as such by the
  self-test rather than folded into a verdict, which is the `recovered` defect #1892 fixed, working
  on a real case. They are deliberately **not** kept listed against their return: a cell coming back
  green should have to prove its oracle can see the coupling, and staying listed would let it arrive
  as `inert (known)` instead.
- One helper, `_solved`, owns the new conjunct across all four verdicts. Absent means not applicable:
  a cell that runs no coupled iteration carries no such field and is unaffected.
- **The fifth coupled cell was left out and is now included.** `fvm_vs_fdm/agreement` runs two
  coupled iterations and records both flags, and its verdict was `worst < AGREEMENT_RTOL` alone — so
  it could certify PASS with neither iteration converged, demonstrated by cutting its budget to 2
  (`PASS`, `worst=0.0343`, both flags `false`). It now carries the conjunct too. This also makes
  `_solved`'s prefixed-field branches reachable: before, all four call sites passed the unprefixed
  name and the `fdm_picard_converged` fallback and `fvm_picard_converged` conjunct were dead code
  written for an artifact shape nothing sent them.
- **The self-test's "not judged this run" line never printed.** It computed
  `COUPLING_INERT_BASELINE - judged`, and pruning the waiver list to one entry removed exactly the
  cells the prose said it named. It is now computed over `MASS_ORACLE_CELLS - judged`, which is the
  right set anyway: a cell that is not PASS has no coupling verdict this run whether or not it was
  ever waived. Eight cells now appear there.
- **The `sl_linear_2d` note said "unmeasured" about something this file already answers.** It
  converges at iteration 32 given a budget of 35 or 60, and `capability_matrix.py`'s module docstring
  says so twenty lines above the helper that gates the cell. The note is corrected: the cell is red
  on its 3-sweep budget rather than on the scheme.
