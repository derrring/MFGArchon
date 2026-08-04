- **133 tests that pin API shape and never touch the numerics** (Issues #1800, #1706). Pre-1.0 the
  API is meant to move, so a test asserting `hasattr`, a constructor signature, or
  `raises(TypeError)` on a call it never makes numerically is a liability: it costs an edit every
  time the surface changes and cannot catch a wrong number. Net **-1635 lines**.
  Selected mechanically -- `hasattr` / `inspect.signature` / `raises(TypeError|AttributeError)`
  present, no numerical call (`solve_*`, `assert_allclose`, `approx`, `norm`, `trapezoid`), and no
  custom assertion helper. Excluded by subject: `test_check_fail_fast`, `test_protocol_compliance`,
  `test_no_hasattr_*`, `test_solver_traits`, the deprecation suites and `test_removed_tier1_*` --
  for those, pinning the shape IS the point.
  `tests/unit/test_alg/test_bug15_sigma_fix.py` is deleted entirely: the selection took all four of
  its tests, which is what #1800 independently reported by hand ("reintroducing Bug #15 leaves all
  four tests green"), and what remained was a dead `if __name__ == "__main__":` block calling them.
