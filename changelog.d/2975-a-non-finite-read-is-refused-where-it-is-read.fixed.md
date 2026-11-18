- **A NaN or infinite value a solver reads makes the solve raise at the step whose result it reaches, or at the
  decision it enters, naming the first such value that step read and where** (#2975, stage 1; #2976 folded). A value
  the step reads and discards, such as a Dirichlet row, is not judged. One function decides it,
  `refuse_a_non_finite_read`, and it raises one error, `NonFiniteReadError`, a `ValueError` (`docs/user/CONVENTIONS.md`,
  "What the solve refuses as non-finite"). Before, the same value gave:
  - **HJB-FDM's Newton step** (`FDM_UPWIND`, the default, and FVM's HJB half): "the Newton step could not be computed"
    and a finite wrong U with `converged` False; on two or more dimensions, "Matrix is exactly singular" and a finite
    wrong U. Now a step that fails is assembled again at its start state, and the value it read there is named. A NaN
    the Hamiltonian meets only at Newton's own iterates goes to the inner-failure policy, as before;
  - **FP-FVM given a NaN optimal control**: a wrong M with `converged` True and no warning, since `max|a| > 0` read a
    NaN velocity as no advection. The mask reads a velocity as present when any face is nonzero, and the sub-step
    count names a non-finite face velocity;
  - **FP-FDM**: "Check CFL condition" for a NaN drift, volatility or source. A step whose result is non-finite now
    names the first value it read that entered it, found by running the step again; the check after the step names the
    CFL condition only for a result computed from finite reads. A capped exit reads the source at its capped nodes, to
    decide the cap. A NaN only where the step discards it, such as a Dirichlet node, solves as before;
  - **HJB-SL on a grid of two or more dimensions**: a non-finite value update was replaced by U at the next time, and
    the step raised only past 10% of the points, so a NaN at a few nodes gave a finite wrong U with `converged` True.
    The fill and the threshold are deleted, and the update names the read behind it;
  - **the network FP**: a NaN transition rate failed `alpha > 0` and was dropped, read as no transition, so a NaN value
    field gave a finite M. The rates are refused;
  - **the doors** (#2976): `solve_hjb_system` returned a non-finite U for a NaN `U_terminal`, and `solve_fp_system`
    blamed the CFL condition for a NaN `M_initial`. Each concrete door reads that row first, and a test reads every such
    definition from source.

  Not yet (stage 2): FP-SL, HJB-FEM, the weak-form FP, WENO, GFDM, Howard and the meshless solvers beyond their doors,
  and HJB-SL's one-dimensional path and its stochastic, Carlini–Silva and DPP steps.
