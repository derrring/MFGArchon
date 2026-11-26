- **A non-finite value at the solve's existing checks now raises `NonFiniteReadError`** (#2975), a `ValueError`, with
  the owner's message. These already refused it, under another type or message:
  - a boundary value that is NaN or infinite raised `NotImplementedError`, and now raises `NonFiniteReadError`, through
    `boundary_value`, `segment_value`, `segment_robin_coefficients`, a `BCSegment`, a `NodeBC` or `MFGProblem` at
    construction, `NodeBC.get_value`, `set_dirichlet_nodes`, `set_source_nodes`, `create_graph_applicator` (its
    `boundary_value` and `source_value`), `GraphApplicator.from_network_geometry`, `GraphApplicator.apply` and
    `apply_direct`, `PreallocatedGhostBuffer.update_ghosts`,
    `bc_to_topology_calculator`, `ImplicitApplicator.apply`, `MeshfreeApplicator.apply`, `apply_field_bc` and
    `apply_particles`, the FEM's `assemble_robin_terms`, and, reading a segment's Robin coefficients, HJB-FDM's
    `compute_hjb_residual`, `compute_hjb_jacobian` and `solve_hjb_timestep_newton`, `HJBGFDMSolver.solve_hjb_system`
    and `HJBHowardSolver.solve_hjb_system` (GFDM's Robin rows), and `FPResolver.resolve` and `HJBResolver.resolve` (a
    provider's resolved coefficient; providers were removed later in this release, #2424), and so every solve that reads one, `problem.solve()` and the doors included.
    Code that caught `NotImplementedError` for it no longer does. A value of a type
    the reader cannot read, and an integer no float holds, keep `NotImplementedError`;
  - a `volatility=` override, the iterator's initial mass, the particle FP's `m_initial`, HJB-FDM's non-finite
    `U_terminal` under a constraint (#2036) and `cfl_substeps`' non-finite CFL number raised `ValueError` with their own
    messages.
- **A callable volatility's or drift's output and a volatility tensor are no longer refused where they are read**
  (#2975). The step that reads them judges them where they entered, so an entry the step discards, such as a periodic
  repeated endpoint, no longer raises, and one that enters raises `NonFiniteReadError` naming the step and the node. A
  volatility's NaN at a Dirichlet node a kept row reads enters the step through D·g, so it still raises, now naming the
  step and the node; at a Dirichlet node no kept row reads, such as the corner of a 2-D grid whose walls are all
  Dirichlet, it is discarded, and no longer raises.

  The message gives the count of non-finite entries and the first of them, `nan`, `inf` or `-inf`, as the old messages
  that printed the value did.
