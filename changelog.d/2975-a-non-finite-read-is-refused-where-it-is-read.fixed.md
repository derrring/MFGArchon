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
    count names a non-finite face velocity, beside the `potential_field` nodes and values it was formed from;
  - **FP-FDM**: "Check CFL condition" for a NaN drift, volatility or source. A step whose result is non-finite now
    names the first value it read that entered it, found by running the step again; the check after the step names the
    CFL condition only for a result computed from finite reads. Both FP-FDM steps take a volatility that varies, or
    holds a NaN, to the Ito branch; a NaN chose the row-scaled branch, which dropped it with a finite wrong M. A capped
    exit's switch is part of the step, so a NaN its outflow reads, in the capped node's wall row or the source, is
    named; `> cap` read it as under the cap, with a finite wrong M. A callable volatility's NaN at a Dirichlet node a
    kept row reads reaches the step through D·g and is named, as it was refused before; at a Dirichlet node no kept row
    reads, such as the corner of a 2-D grid whose walls are all Dirichlet, it is discarded and solves, where it was
    refused before. A NaN at a periodic repeated endpoint, and a NaN `M_initial` at the wall of a 1-D BC whose
    segments are all Dirichlet, where level 0 holds g, are read and discarded, and solve as before. A volatility
    tensor's NaN is named as non-finite, not under a PSD message;
  - **HJB-SL on a grid of two or more dimensions**: a non-finite value update was replaced by U at the next time, and
    the step raised only past 10% of the points, so a NaN at a few nodes gave a finite wrong U with `converged` True.
    The fill and the threshold are deleted, and the update names the read behind it;
  - **the network FP**: a NaN transition rate failed `alpha > 0` and was dropped, read as no transition, so a NaN value
    field gave a finite M. The rates are refused, and a NaN value field is named by its node and the rate that reads it;
  - **the doors** (#2976): `solve_hjb_system` returned a non-finite U for a NaN `U_terminal`, and `solve_fp_system`
    blamed the CFL condition for a NaN `M_initial`. Each concrete door reads that row first, and a test reads every such
    definition from source.

  Not yet (stage 2): FP-SL, HJB-FEM, the weak-form FP, WENO, GFDM, Howard and the meshless solvers beyond their doors;
  HJB-SL's source in either dimension, its one-dimensional path and its stochastic, Carlini–Silva and DPP steps; and a
  constraint's obstacle, which HJB-FDM's projection reads after the step.
