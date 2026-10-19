- **A particle solve no longer writes into the problem's boundary conditions** (#2738).
  - On the per-point path, `ParticleApplicator.apply` set `domain_bounds` on the BC it was handed when it had none;
    `apply_with_flux_limits`, which no solver calls, did the same. That BC is the problem's own on the shared route when
    the solver's view of it translates nothing, and when a caller hands the solver the grid's own BC. On the legacy route
    (`MFGProblem(spatial_bounds=..., components=MFGComponents(boundary_conditions=bc))`), every later
    `problem.get_boundary_conditions()`, and every solve that reads it, then refused the problem as declaring its
    boundary conditions twice. In a coupled loop that is the first BC read after the first particle solve.
  - The applicator now queries a BC with no `domain_bounds` through a copy that carries the step's box in that slot,
    and never writes the BC it is handed. The face a particle leaves through and the point a door the grid marks is
    read at past the wall come from that one box, as when it was written, and the copy keeps the BC's grid binding.
    The particles and the density are bit-identical to before in every solve measured. The one difference is a BC
    stepped in two boxes in turn, as one BC object shared by two solves on different domains is: the write kept the
    first solve's box for the second, so a door the second domain reaches was read as if it lay outside, and the copy
    reads each step in its own box.
