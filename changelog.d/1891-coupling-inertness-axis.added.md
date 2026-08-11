- **`--self-test` gains a second axis: delete `f(m)` from every fixture and see which cells notice**
  (Issue #1891). The existing axis injects 10% mass drift into the recorded density, which proves a
  mass oracle reads the density it reports and says nothing about whether the verdict depends on the
  problem being a coupled MFG at all. Measured, it does not: **all five cells that are PASS today
  stay PASS with the coupling identically zero** — `fdm_upwind`, `sl_linear`, `fvm_muscl`,
  `sl_linear_2d`, and `fvm_vs_fdm/agreement`, whose independence claim rests on a second
  discretisation solving the same coupled problem and which agrees just as well when there is no
  coupling to solve.
  Why, by construction: `mass_t0` is 1 by normalisation, `max_rel_drift` is a property of the FP
  time-stepping which holds on whatever drift field it is handed, and `min_density` is the `t=0`
  value of the initial condition. None of them reads the coupled solution.
- **Recorded as a ratchet, not a pass.** Every currently-PASS cell is in `COUPLING_INERT_BASELINE`,
  so the new axis cannot fail the build on a defect it merely discovered. What it catches is the list
  **growing** — a new or recovered cell arriving with an oracle that cannot tell an MFG from two
  uncoupled PDEs — and it fails the other way too, telling you to remove a cell from the list when it
  starts discriminating. Same structure as `check_fail_fast.py` / `fail_fast_baseline.json`.
  Shrinking the list is the work, and it is #1891.
- **The axis carries a positive control on its own seam**, because without one it reported its own
  no-op as a result. Every cell here is inert, so "deleted the coupling and nothing changed" and
  "never deleted the coupling" print identically — verified: stubbing the seam back to a pass-through
  still produced `SELF-TEST PASSED: ... 5 known-inert ... none new`. The control evaluates a
  fixture's `f` under the flag and aborts if it does not return 0. Mutation-verified in both
  directions: dropping a cell from the baseline reports it `INERT (NEW)` and exits 1; stubbing the
  seam now exits 2 with `the coupling seam did not fire -- f(1,2) returned 2.0, not 0`.
- All four fixtures route their coupling through one `_coupling_pair` helper, so the deletion has a
  single owner rather than four.
