- **Six readers' time defaults to `None`, not 0.0, so a value that declares `t` is refused when no time is given,
  not read at t = 0** (#2512, row B2). A constant, or `g(x)`, needs no time and reads as before. The six are:
  - `PreallocatedGhostBuffer.update_ghosts`, which HJB-WENO relied on (#2605);
  - `FDMApplicator.enforce_values`;
  - `MeshfreeApplicator.apply`;
  - `applicator_fdm.dirichlet_wall_rows`;
  - `base_hjb._get_bc_info_1d`;
  - the deprecated `BaseNeuralSolver.get_boundary_target_values`.
- **FEM's Dirichlet assembly and meshless Galerkin's Nitsche assembly, called with no `t=` on a value that declares
  `t`, now say to pass one** (#2512, row B2). They had advised writing the value as g(x), which drops the time a
  caller has. The functions are `fem.bc_adapter.apply_bc_to_fem_system` and `get_dirichlet_dofs_and_values`, and
  `nitsche.assemble_nitsche_terms`. The solvers always pass the time.
