- **Tests needing `joint_socp` now skip without cvxpy instead of failing** (found while verifying
  #2090). `joint_socp.py` raises `ImportError("cvxpy is required for joint SOCP")`, and cvxpy is in
  the `numerical` extra, so `uv run --extra dev pytest tests/unit` reported **17 failures** that
  read like real breakage in `test_howard_refuses_undecomposable_hamiltonian_2011.py` (16) and
  `test_gfdm_mms_source_1991.py` (1). The guard is on the `monotonicity_scheme` rather than at
  module level: `test_newton_still_accepts_the_same_hamiltonian` is the accept control for the
  sixteen refusal tests and needs no SOCP, and the other five tests in the MMS file do not either —
  a module-level `importorskip` would have taken the control down with the tests it controls.
  Measured both ways: with cvxpy blocked, 6 passed / 17 skipped / 0 failed; with it present,
  23 passed / 0 skipped.
