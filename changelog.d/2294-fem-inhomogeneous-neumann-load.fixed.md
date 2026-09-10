- **Both FEM solvers assemble the inhomogeneous Neumann boundary load** (Issue #2294). `du/dn = g`
  with `g != 0` owes the weak form `D * int_dOmega g phi_i`; nothing assembled it, so `g=0` and
  `g=5` gave bit-identical solutions while `HJBFEMSolver` and `FPFEMSolver` both reported
  `honors_inhomogeneous_neumann = True`. `assemble_robin_terms` now owns the natural-BC load as
  well, since `NEUMANN(g)` is `ROBIN(alpha=0, beta=1, g)`. A homogeneous wall still assembles
  nothing, so every pre-existing natural-BC solve is unchanged.
