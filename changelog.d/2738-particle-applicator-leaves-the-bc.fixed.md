- **A particle solve no longer writes into the problem's boundary conditions** (#2738).
  - On the per-point path, `ParticleApplicator.apply` and `apply_with_flux_limits` set `domain_bounds` on the BC they
    were handed when it had none. On the shared route that BC is the problem's own. On the legacy route
    (`MFGProblem(spatial_bounds=..., components=MFGComponents(boundary_conditions=bc))`), every later
    `problem.get_boundary_conditions()`, and every solve that reads it, then refused the problem as declaring its
    boundary conditions twice. In a coupled loop that is the first BC read after the first particle solve.
  - The applicator now reads a copy that carries the step's box, and the caller's BC is untouched. The particles and
    the density are bit-identical to before.
