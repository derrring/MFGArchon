- **`fp_drift_coefficient` refuses every Hamiltonian that is not a quadratic `SeparableHamiltonian`, and a problem with none** (#2977). It used to return the problem's `coupling_coefficient` for them, a value no Hamiltonian decided (default 0.5). The scalar c in −c·∇U is a quadratic separable cost's alone. So the solvers that form −c·∇U themselves now refuse such a Hamiltonian by name where they did not:
  - FP-FDM's `potential_field` route and `build_advection_operator`;
  - HJB-FDM's `build_advection_matrix`;
  - the semi-Lagrangian FP solver;
  - the particle solver for a Hamiltonian that is not separable;
  - the meshless HJB's streamline diffusion.

  The solvers that read `optimal_control` themselves are unaffected (`docs/user/CONVENTIONS.md`, "What the Fokker–Planck solver receives").
