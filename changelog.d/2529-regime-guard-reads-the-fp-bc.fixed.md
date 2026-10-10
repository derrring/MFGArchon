- **`RegimeSwitchingIterator` no longer refuses an FEM or meshless FP with a shared exit** (#2529).
  - **The bug.** Its FP-data guard refuses inhomogeneous FP boundary data on a regime with outflow. For a
    solver with no `boundary_conditions` attribute (the weak-form family), it read the geometry's BC raw.
    The FP reads a shared Dirichlet as an absorbing exit with its value dropped (#2512, row B3), so a
    shared exit cost of 0.7 was refused with "carry data that is not verifiably zero: [0.7]", while the
    FEM FP imposed homogeneous data.
  - **The fix.** The guard now asks the solver's `get_boundary_conditions()`, which ends in
    `BaseFPSolver._fp_view_of_shared` and is what the weak-form family imposes.
  - **Unchanged.** An FP solver given its own inhomogeneous data is still refused.
  - **The guard.** Row B3's structural guard now covers the coupling layer. It also reads the
    `getattr(x, "boundary_conditions", …)` form the fallback used, which the attribute-only rule could not
    see.
