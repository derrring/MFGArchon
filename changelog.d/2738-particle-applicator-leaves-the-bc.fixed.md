- **A particle solve no longer writes into the problem's boundary conditions** (#2738).
  - On the per-point path, `ParticleApplicator.apply` set `domain_bounds` on the BC it was handed when it had none;
    `apply_with_flux_limits`, which no solver calls, did the same. That BC is the problem's own on the shared route when
    the solver's view of it translates nothing, and when a caller hands the solver the grid's own BC. On the legacy route
    (`MFGProblem(spatial_bounds=..., components=MFGComponents(boundary_conditions=bc))`), every later
    `problem.get_boundary_conditions()`, and every solve that reads it, then refused the problem as declaring its
    boundary conditions twice. In a coupled loop that is the first BC read after the first particle solve.
  - The applicator now writes nothing. The face a particle leaves through comes from the step's box, as before.
    Given the face, the BC's owner reads each segment's placement at the particle. For a `region_name` the grid
    marks, a particle past the wall is read at its nearest point of the BC's box: its `domain_bounds`, else the box
    of the grid that marks the region. That second box is new for a BC whose domain is its `domain_sdf`, which has no
    box of its own: it is the box the deleted write handed such a BC, so a door it places solves as before instead of
    being refused as outside the grid. In every solve measured, the particles and the density are bit-identical to
    before.
