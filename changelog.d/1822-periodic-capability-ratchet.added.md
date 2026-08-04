- **A ratchet on the periodic-BC capability claim** (Issue #1822). `tests/unit/test_alg/test_periodic_capability_invariant_1822.py`
  runs one solve from exactly periodic data through every solver that declares `BCType.PERIODIC`
  and asserts the output is still periodic -- `x_min` and `x_max` are the same physical point on an
  endpoint-inclusive grid, so the seam is zero in exact arithmetic. **Nine of the eleven declaring
  solvers fail it**; `FPFVMSolver` (2.2e-16) and `FPGFDMSolver` (2.2e-11) are the two that honour
  the claim. `HJBGFDMSolver` is a distinct case: it declares PERIODIC and raises
  `NotImplementedError` for it.
  Each failure carries `xfail(strict=True)`, so repairing a solver turns its XFAIL into an XPASS and
  fails the suite until the marker is removed in the same change -- the count can only go down.
  Coverage is guarded from an independent source: an AST walk over `mfgarchon/alg/numerical/` for
  declaration sites, rather than the import list the matrix is built from, because comparing the
  matrix against its own source is tautological and was measured passing while the matrix silently
  stopped covering a solver.
