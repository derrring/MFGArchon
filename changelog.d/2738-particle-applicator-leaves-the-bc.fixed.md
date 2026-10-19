- **A particle solve no longer writes into the problem's boundary conditions** (#2738).
  - On the per-point path, `ParticleApplicator.apply` set `domain_bounds` on the BC it was handed when it had none;
    `apply_with_flux_limits`, which no solver calls, did the same. That BC is the problem's own on the shared route when
    the solver's view of it translates nothing, and when a caller hands the solver the grid's own BC. On the legacy route
    (`MFGProblem(spatial_bounds=..., components=MFGComponents(boundary_conditions=bc))`), every later
    `problem.get_boundary_conditions()`, and every solve that reads it, then refused the problem as declaring its
    boundary conditions twice. In a coupled loop that is the first BC read after the first particle solve.
  - The applicator now reads a copy that carries the step's box, and the caller's BC is untouched. The particles and
    the density are bit-identical to before.
