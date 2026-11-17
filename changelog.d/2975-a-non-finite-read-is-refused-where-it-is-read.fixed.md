- **A value a solver reads during the solve that is NaN or infinite is refused where it is read, naming what was read
  and where** (#2975, stage 1; #2976 folded). One function decides it, `refuse_a_non_finite_read`, and it raises one
  error, `NonFiniteReadError`, a `ValueError` (`docs/user/CONVENTIONS.md`, "What the solve refuses as non-finite").
  Before, the same value gave:
  - **HJB-FDM's Newton step** (`FDM_UPWIND`, the default, and FVM's HJB half): "the Newton step could not be computed"
    and a finite wrong U with `converged` False. Now the step names the term of its residual or Jacobian that is
    non-finite, each re-read as the residual assembles it: the source by its part, the wall, a source the Hamiltonian
    holds, the Hamiltonian's value or its `dp`, at (t, x, p, m). The Jacobian's per-point difference no longer fills
    0.0 for a non-finite value. On a grid of two or more dimensions, "Matrix is exactly singular" and a finite wrong U
    become the same refusal;
  - **FP-FVM given a NaN optimal control**: a wrong M with `converged` True and no warning, since `max|a| > 0` read a
    NaN velocity as no advection. The face velocities and the source are refused before the step, and the mask
    reads a velocity as present when any face is nonzero;
  - **FP-FDM**: "Check CFL condition" for a NaN drift or source. The drift is refused before the step and the source
    where the step adds it, at the rows that keep it, so a NaN only at a Dirichlet node, whose row discards it, still
    solves as before. The check after the step stays, and names the CFL condition, for a non-finite M from finite
    inputs;
  - **HJB-SL on a grid of two or more dimensions**: a non-finite value update was replaced by U at the next time, and
    the step raised only past 10% of the points, so a NaN at a few nodes gave a finite wrong U with `converged` True.
    The fill and the threshold are deleted, and the update names the read behind it;
  - **the network FP**: a NaN transition rate failed `alpha > 0` and was dropped, read as no transition, so a NaN value
    field gave a finite M. The rates are refused;
  - **the doors** (#2976): `solve_hjb_system` returned a non-finite U for a NaN `U_terminal` or `M_density`, and
    `solve_fp_system` blamed the CFL condition. Each concrete door now reads the arrays it is handed first, and a test
    reads every such definition from source.

  Not yet (stage 2): FP-SL, HJB-FEM, the weak-form FP, WENO, GFDM, Howard and the meshless solvers beyond their doors,
  and HJB-SL's one-dimensional path and its stochastic, Carlini–Silva and DPP steps.
