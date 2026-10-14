- **A standalone HJB-FEM or meshless-Galerkin solve no longer solves the heat equation in place of the HJB** (#2642).
  `WeakFormHJBSolver.solve_hjb_system` defaulted a missing `U_coupling_prev` to zeros, and its default Picard branch
  evaluates H at that iterate's gradient. So a call without it evaluated H at p = 0, which for a quadratic H is the
  backward heat equation: on #2642's Dirichlet problem the answer stayed 0.31 off the exact solution at every
  resolution, with no warning. HJB-FDM, semi-Lagrangian and WENO already refused the same call.
  - An input the configured path reads is now required, with the FDM message "`<name>` is required":
    - `M_density` and `U_terminal` on `WeakFormHJBSolver` (HJB-FEM and meshless Galerkin), which defaulted to
      `1/N` and zeros;
    - `U_coupling_prev` on the weak-form Picard branch. The Newton inner solver does not read it, and it stays
      optional there;
    - `M_density` on `HJBGFDMSolver`, which defaulted to ones. That is a coupling value, not "no coupling", as its
      comment said.
  - `solve()` on a weak-form solver configured for Picard (`HJBFEMSolver`, and `MeshlessGalerkinHJBSolver` built
    with `use_newton=False`) is refused. `BaseHJBSolver.solve` would pass the terminal condition as the iterate, a
    linearization point nobody chose. On the Newton branch it is unchanged.
  - `HJBGFDMSolver`'s `U_coupling_prev` stays optional: no GFDM path reads it (#1055).
  - Migration: pass the density, the terminal condition and the previous coupling iterate you mean. To reproduce
    the old standalone result, pass `U_coupling_prev=np.zeros((Nt + 1, N))`, with `M_density=np.ones((Nt + 1, N)) / N`
    on a weak-form solver or `np.ones(...)` on GFDM.
