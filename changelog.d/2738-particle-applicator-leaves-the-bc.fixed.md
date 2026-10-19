- **A particle solve no longer writes into the problem's boundary conditions** (#2738).
  - On the per-point path, `ParticleApplicator.apply` set `domain_bounds` on the BC it was handed when it had none;
    `apply_with_flux_limits`, which no solver calls, did the same. That BC is the problem's own on the shared route when
    the solver's view of it translates nothing, and when a caller hands the solver the grid's own BC. On the legacy route
    (`MFGProblem(spatial_bounds=..., components=MFGComponents(boundary_conditions=bc))`), every later
    `problem.get_boundary_conditions()`, and every solve that reads it, then refused the problem as declaring its
    boundary conditions twice. In a coupled loop that is the first BC read after the first particle solve.
  - The applicator now writes nothing. The face a particle leaves through comes from the step's box, as before.
    Where a segment covers part of that face, the BC's owner decides it on the BC's own box: its `domain_bounds`, or
    the bounds of the grid it is bound to, which is the box every solve passed. The particles and the density are
    bit-identical to before.
