- **`--self-test` gains a second axis: transform `f(m)` in every fixture and see which cells notice**
  (Issue #1891). The existing axis injects 10% mass drift into the recorded density, which proves a
  mass oracle reads the density it reports and says nothing about whether the verdict depends on the
  problem being a coupled MFG at all.
- **A family of mutations, because deletion alone is the weakest member.** Setting `f(m)` to zero
  also makes the problem easier, so surviving it is weaker evidence than it looks. Measured over the
  five cells that are PASS: deletion and a sign flip leave all five PASS; **scaling by 10 turns
  `fvm_vs_fdm/agreement` FAIL**. A cell counts as inert only if it survives every member, so the
  known-inert list is **four**, not five — `fdm_upwind`, `sl_linear`, `fvm_muscl`, `sl_linear_2d`.
  Those four would report PASS on two uncoupled PDEs. `fvm_vs_fdm/agreement` has a coupling it can
  feel; its independence claim is still weaker than it reads, since two discretisations agreeing on
  an uncoupled problem is not the same statement, but it is not blind.
  Why the four are blind, by construction: `mass_t0` is 1 by normalisation, `max_rel_drift` is a
  property of the FP time-stepping which holds on whatever drift field it is handed, and
  `min_density` is the `t=0` value of the initial condition.
- **Recorded as a ratchet.** The four are in `COUPLING_INERT_BASELINE`, so the axis cannot fail the
  build on a defect it merely discovered; it fires when the list grows, and when a cell starts
  discriminating and the list is not updated. Same structure as `check_fail_fast.py`. Shrinking the
  list is #1891 — and it shrank by one here, which is what the family bought over deletion alone.
- **The control reaches the call sites, not just the helper.** A first version checked that
  `_coupling_pair` returns a transformed function, which proves the helper works and nothing about
  whether any fixture asked it: measured, a fixture edited to stop calling the seam left the whole
  axis byte-identical and exit 0 — the same *"changed it and nothing happened"* versus *"never
  changed it"* ambiguity the axis exists to avoid, one level up. Each fixture is now built under the
  flag and its Hamiltonian evaluated, and a call site that bypasses the seam aborts with exit 2
  naming the fixture and the mutation.
- **Two logic defects in the first version, both found by review and both pinned.** `recovered` was
  computed against the whole baseline rather than the cells actually judged, so a baselined cell that
  had merely stopped passing — never mutated at all — was reported as "now discriminates on the
  coupling", announcing a capability regression as an improvement. And its return preempted axis 1's,
  so a cell that does not read the density it reports had its verdict replaced by a coupling message.
  The second is invisible to an exit-code assertion, since both orderings return 1; its test asserts
  which message came out.
  Mutation-verified, each anchor asserted before applying and the file SHA-256-verified restored:
  a fixture bypassing the seam → exit 2; a cell dropped from the baseline → exit 1; the family cut
  back to deletion only → exit 1 (`fvm_vs_fdm/agreement` newly inert); `recovered` over the whole
  baseline → its test red; `recovered` checked before `inert` → its test red.
