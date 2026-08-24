- **Tests that never needed cvxpy no longer ask for it, and the tiers that install it now prove
  they did.** `test_howard_refuses_undecomposable_hamiltonian_2011.py` (17 tests) and
  `test_gfdm_mms_source_1991.py` asked for `monotonicity_scheme="joint_socp"`, which needs cvxpy
  from the `numerical` extra and raises `ImportError` at solver construction without it — so
  `pip install -e .[dev]` turned them into failures that read like solver breakage. What those
  files pin is Howard's *decomposition gate* and that `source_term` reaches the Howard branch,
  neither of which is about SOCP stencils, so they now use `qp_m_matrix`, which runs on osqp (a
  base dependency). Measured: **23 passed with and without cvxpy, none skipped** — where guarding
  them with `importorskip` would have given 6 passed / 17 skipped. SOCP keeps its own coverage in
  `test_socp_m_matrix_property`, `test_socp_stencil_enlargement` and
  `test_joint_socp_mirror_symmetry`. `tests/integration/test_diffusion_magnitude_gate.py` does
  genuinely need `joint_socp` and now skips cleanly rather than failing where the extra is absent.
  Finally, `nightly.yml` and `discrimination.yml` verify `import cvxpy` immediately after
  installing `.[dev,numerical]`: per #1658 that extra once resolved to nothing for 3.5 months and
  the only observable was a SOCP `ImportError` inside the suite, so removing that observable
  requires the install to prove itself instead.
