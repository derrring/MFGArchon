- **The weekly discrimination sweep can go green again, and closes its own issue when it does**
  (Issue #1817). The sweep has never once succeeded: its first run aborted on a test that was red on
  the runner and green locally, #1828 fixed that the next day, and nothing re-ran it because the
  schedule is weekly and nothing closes the issue. The baseline it compares against was measured at
  `bec28ce5` (`collected: 5683`) against 5872 now, and it refuses a drop *and* a rise, so it could not
  have gone green even with the runner fixed. Re-measured on `db3496f9`; a local run and the CI sweep
  agree on all six counts and on the 212-distinct-tests total. `discrimination.yml` also gains the
  `resolve-on-success` edge that #1806 gave `nightly.yml` — it had only an opening edge, which is why
  #1817 stayed open for four days after its cause was fixed.
- **Three of the six mutations lost killers, and the counts show only one of them** (Issue #1817).
  `diffusion_scalar_2x` 129 → 145 and `optimal_control_sign` 34 → 40 are rises (8 of the first from
  #1837's heat-kernel oracle, 3 of the second from #1844's time-level test), and
  `bc_noflux_reads_as_clamp` 11 → 9 is the only visible fall. Diffing the committed kill matrices
  shows two more: `diffusion_scalar_2x` lost `test_dpp_error_converges_under_refinement` behind its
  +17, and **`drift_coefficient_2x` is 19 → 19 with a 1-for-1 swap underneath** —
  `test_one_newton_step_reduces_the_mfg_residual` stopped noticing and a periodic-capability test
  took its place. Both of those tests still exist, as do the two semi-Lagrangian ones in the next
  bullet; none was deleted, they stopped discriminating. A count-based ratchet
  cannot see an equal-size swap, by construction, so the matrix is the only place that record exists
  — which is why it is committed beside the baseline rather than merely produced.
- **`bc_noflux_reads_as_clamp` 11 → 9 traced, and the conclusion is narrower than the count suggests**
  (Issue #1817, refs #1852). One lost killer is a genuine duplicate: #1827 deleted a test whose
  assertion is the identical construction, asserted against the identical set, as a live test in the
  same file that still kills. The other two were the **only killers that exercised this convention
  through an actual solve** (`solve_hjb_system`); they stopped when #1819 reworked the semi-Lagrangian
  fold. What remains is 8 tests restating the mapping function's own output table plus one whose kill
  is an artefact — so the convention is still watched, but no longer at the level where getting it
  wrong changes an answer. Recorded rather than silently accepted, and filed as #1852.
