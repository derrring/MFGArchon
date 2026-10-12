- **Six readers' time defaults to `None`, not 0.0, so a value that declares `t` is refused when no time is given,
  not read at t = 0** (#2512, row B2). A constant, or `g(x)`, needs no time and reads as before. The six are:
  - `PreallocatedGhostBuffer.update_ghosts`, which HJB-WENO relied on (#2605);
  - `FDMApplicator.enforce_values`;
  - `MeshfreeApplicator.apply`;
  - `applicator_fdm.dirichlet_wall_rows`;
  - `base_hjb._get_bc_info_1d`;
  - the deprecated `BaseNeuralSolver.get_boundary_target_values`.
- **A reader that takes a time but was given none now says to pass it, by its own parameter name, when it refuses a
  value that declares `t`** (#2512, row B2). For example: "… declares t, but … was given no time. A caller that has one
  passes it as time=; a solver that builds … without one does not yet support a time-dependent value there (#2615)".
  - This covers the six readers above, `BoundaryConditions.get_bc_value_at_boundary` and `BCSegment.get_value`.
  - It also covers FEM's Dirichlet assembly (`fem.bc_adapter.apply_bc_to_fem_system`, `get_dirichlet_dofs_and_values`)
    and meshless Galerkin's Nitsche assembly (`nitsche.assemble_nitsche_terms`), which take `t=`.
  - A reader with no time to take still advises a value that needs none.
  - Each solver passes its level's time to these. The exception is meshless Galerkin's FP: its Nitsche data are 0,
    an exit, and need none.
