- **Two reported statuses now describe the run they name** (Issue #1684, items 3 and 5).

  `DistributionConvergenceMonitor.get_convergence_summary()["converged"]` was
  `len(converged_iterations) > 0` -- "some iterate, ever, met the criteria". A run that met them
  and then diverged reported success. It now reports the last iterate's state.
  `convergence_iteration` still records the first iterate that met the criteria, so "did it ever
  converge" remains answerable and is now a different question from "did it end converged".

  `MultiPopulationIterator` measured convergence from `M` alone -- `U_old` was never captured, so
  the value function could not enter the test at any tolerance. It now captures both and reports
  the larger per-population change. `errors` keeps its shape and type but now means what its
  documented description ("Final per-population errors") already implied; `errors_M` and `errors_U`
  are added so a non-converged run says which field failed.

  The m error is the **map's** residual, `M_map - M_old`, not the damped step `M - M_old`. Since
  `M = (1 - r) * M_old + r * M_map`, the damped step is identically `r * (M_map - M_old)`, so an
  error measured on it scales with the relaxation factor -- #1684 items 6 and 7, "turning damping
  down makes anything converge", already fixed and pinned for the single-population path in
  `nonlinear_solvers.py` and `fixed_point_iterator.py`. Measured on one sweep of the pinning test's own
  fixture, before the correction: 4.095006e-01 / 2.047503e-01 / 4.095006e-02 / 4.095006e-03 at
  relaxation 1.0 / 0.5 / 0.1 / 0.01, exactly 1/r; after, 4.095006e-01 at all four. `u` is never damped here,
  so its half was always the map residual. Pinned by an invariance assertion rather than a value.

  The caveat that remains: these are absolute max-norm changes, so `u` and `m` meet one tolerance
  in their own units, while `FixedPointIterator` tracks `l2distu_rel` / `l2distm_rel`. Aligning the
  two criteria is a single-source question and is deliberately not folded in here.

  Pinned by `tests/unit/test_alg/test_status_reports_the_final_state_1684.py`, of which 4 of 6 fail
  against the pre-fix source and 2 of those fail behaviourally rather than on a missing attribute.
