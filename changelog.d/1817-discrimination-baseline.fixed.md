- **The weekly discrimination sweep can go green again, and closes its own issue when it does**
  (Issue #1817). The sweep has never once succeeded: its first run aborted on a test that was red on
  the runner and green locally, #1828 fixed that the next day, and nothing re-ran it because the
  schedule is weekly. The baseline it compares against was measured at `bec28ce5` (`collected: 5683`)
  against 5872 now, so it refused in both directions at once. Re-measured on `db3496f9`, and confirmed
  by two independent instruments — a local run and the CI sweep agree on all six counts and on the
  212-distinct-tests total. `discrimination.yml` also gains the `resolve-on-success` edge that #1806
  gave `nightly.yml`: it had only an opening edge, which is why #1817 stayed open for four days after
  its cause was fixed.
- **Two of the six counts rose, and the third fell without losing coverage** (Issue #1817).
  `diffusion_scalar_2x` 129 → 145 and `optimal_control_sign` 34 → 40, attributed: 8 of the first come
  from #1837's heat-kernel oracle, 3 of the second from #1792's time-level test. `bc_noflux_reads_as_clamp`
  11 → 9 reads as a loss and is not one, traced killer by killer: one was a test #1827 deleted whose
  assertion is byte-equivalent to a live one in the same file (same BC construction, same expectation,
  and that live test is itself still a killer), and two were `H=0` semi-Lagrangian tolerance tests that
  noticed the mutation *incidentally* and stopped when #1819 reworked the fold — replaced by a
  byte-identity pin against a reference implementation, which is a deliberate detector of the same
  convention. The ratchet counts killers, so removing a duplicate and trading two weak detectors for
  one strong one both register as regressions; the numbers are recorded with that attribution rather
  than silently accepted.
