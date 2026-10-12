- **The weak-form FP adjoint step refuses a Dirichlet wall it cannot impose** (#2608). `FPFEMSolver` and
  `MeshlessGalerkinFPSolver` inherit `WeakFormFPSolver.solve_fp_step_adjoint_mode`. That step solves with no
  condensation and no Nitsche term, so it dropped a Dirichlet wall in silence. Measured at 375122e5 on FP-FEM with
  its own `dirichlet_bc(value=0.7)`: one step left the walls at 1, where the forward step gives 0.7. Meshless
  Galerkin's absorbing wall was left at 1, where the forward step gives 0. HJB-FDM with
  that FP-FEM under `BlockIterator(adjoint_mode="jacobian_transpose")` ran with the walls at 1. It now raises
  `NotImplementedError` and points to `adjoint_mode='off'`, as FP-FDM's adjoint step does (#2531). A BC with no
  Dirichlet wall, such as no-flux or Robin, is solved as before.
