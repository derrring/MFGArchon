- **Every FP solver names its initial density `M_initial`, as the abstract base now does (#2377).**
  This finishes the v0.17.0 rename. When #1356 removed the old names at v0.20, its release note
  said the rename covered the `BaseFPSolver` abstract interface, but the base kept
  `m_initial_condition`, and so did `FPGFDMSolver`, deliberately left unchanged then.
  `WeakFormFPSolver` said `m_initial`, and `FPFEMSolver` and `MeshlessGalerkinFPSolver` inherit its
  signature.

  Measured at `31de2179` over the ten concrete library solvers:
  - `solve_fp_system(m_initial_condition=...)` bound only on `FPGFDMSolver` and on `FPNetworkSolver`,
    through its alias, and raised `TypeError` on the other eight.
  - `M_initial=` raised on four: GFDM and the weak-form family.

  Positional calls are unaffected.

  - The weak-form family's `solve_fp_system(m_initial=...)` still works. It emits a
    `DeprecationWarning`, solves the same problem, and is removed at v0.25.0. Passing the old and the
    new name together raises `ValueError`. `FPGFDMSolver`'s `m_initial_condition` went with the
    solver, withdrawn in this same release (#2583).
  - The weak-form family now raises `ValueError("M_initial is required")` when no initial
    density is given. It used to require it positionally. FDM, the SL solver and the network
    solver already raise this. FVM falls back to the problem's initial density, and the particle
    solver accepts `initial_particles` instead.
  - A test pins the name on the nine library solvers that subclass the base, `FPGFDMSolver`'s refusing
    stub among them. `FPSLJacobianSolver`, the tenth when this was written, was retired in #1756. A
    future solver is covered only if its module is imported there.
  - Known gap: the weak-form family's new warning is the inner of two stacked deprecation
    decorators. It is therefore attributed to `deprecation.py`, and Python's default filters hide
    it outside a test run. This is #2417, which also affects five older functions.
