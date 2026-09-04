- **A manufactured `(u*, m*)` can be generated from a geometry and a boundary condition, and
  Dirichlet walls have a measured convergence order for the first time** (Issue #2201). Every
  manufactured pair in the repository was hand-written for one geometry and one wall, so
  "MMS-unreachable" was ambiguous between a real gap in a solver and a fixture that was never
  admissible for it. `manufactured.pair_for(geometry, bc_type)` returns the pair together with the
  boundary conditions it satisfies exactly — separately for `u` and for `m`, because under
  Dirichlet they differ: `u` vanishes on every wall while `m` sits at a positive floor. That floor
  is measured rather than stylistic. A homogeneous-Dirichlet density is zero at the wall, so the
  exact solution touches zero and any negative truncation error there *is* a negative density:
  such a pair makes `FPFDMSolver` raise at timestep 1 of 17 with `density went to -2.928e-02`.
  With the floor, the same study runs and gives interior EOC 1.05 and 0.99 over nx = 21, 41, 81 —
  first order, which is what the default `divergence_upwind` should give, and the first order ever
  measured at a Dirichlet wall in this repository.
- **ROBIN is refused by the generator, and the refusal says where the blocker is** (Issue #2201).
  `FPFDMSolver`, `FPFVMSolver`, `FPGFDMSolver`, `FPSLSolver` and `HJBFDMSolver` all raise at
  *construction* for a ROBIN segment (#1456); only `HJBGFDMSolver` accepts one, and only for the
  adjoint-consistent `Robin(0, 1)` case. A generated Robin pair would have no consumer able to run
  it. The generator side needs one branch when that changes, since a Robin `g` is not a constraint
  the pair must satisfy but a quantity computed from it.
