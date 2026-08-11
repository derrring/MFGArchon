- **A capability cell now requires the coupled iteration to have converged** (Issue #1891).
  `picard_converged` was recorded but left out of the verdict by #1871, on the stated ground that
  gating it "turns three long-standing greens red" — a transition cost, not a correctness argument.
  Two things settle it now.
  First, #1865 had already required it in as many words: *"any restatement of the 2-D fixture must
  assert convergence, or the cell measures whether the FP time-stepper conserves mass (which it does
  regardless) rather than whether the configuration solves."*
  Second, and new: it is **the only recorded field measured to depend on the coupling**. Under the
  mutation family added in #1892, deleting `f(m)` makes `fdm_upwind` and `sl_linear` converge in
  **one** sweep against five, and a 10x coupling stops `fvm_vs_fdm/agreement` converging at all (40
  sweeps) — while `mass_t0`, `max_rel_drift` and `min_density` do not move at all. A verdict without
  it cannot tell an MFG from two uncoupled PDEs, which is what #1891 is about.
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
