- **A ratchet on the periodic-BC capability claim** (Issue #1822). `tests/unit/test_alg/test_periodic_capability_invariant_1822.py`
  runs one solve from exactly periodic data through every solver that declares `BCType.PERIODIC`
  and asserts the output is still periodic -- `x_min` and `x_max` are the same physical point on an
  endpoint-inclusive grid, so the seam is zero in exact arithmetic. **Nine of the eleven declaring
  solvers fail it.** The two that appeared to honour it, `FPFVMSolver` (2.2e-16) and `FPGFDMSolver`
  (2.2e-11), were certified by the test data rather than by the code: both numbers were measured on
  a datum symmetric about the midpoint, under which a NO_FLUX solve and a PERIODIC solve agree. On
  the phase-shifted datum this file now uses they score 1.79e-01 and an outright raise. Resolved
  since, across this release: `FPFVMSolver` was repaired (the wrap sat on a torus one cell too
  long), and `FPGFDMSolver` stopped declaring PERIODIC, having never had a periodic code path.
  `HJBGFDMSolver` is a third case: it honours PERIODIC on a cloud with no detected boundary points
  and raises `NotImplementedError` on the default one, which is a reachability defect (#1841), not
  a false declaration.
  Each failure carries `xfail(strict=True)`, so repairing a solver turns its XFAIL into an XPASS and
  fails the suite until the marker is removed in the same change -- the count can only go down.
  Coverage is guarded from an independent source: an AST walk over `mfgarchon/alg/numerical/` for
  declaration sites, rather than the import list the matrix is built from, because comparing the
  matrix against its own source is tautological and was measured passing while the matrix silently
  stopped covering a solver.
