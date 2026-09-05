- **The SL/CN family's no-flux wall is the mirror stencil** (Issue #2243). Every implicit-diffusion
  call site in the library now takes `treatment="mirror"`, applying to the Semi-Lagrangian /
  Crank-Nicolson family the decision #2145 had already made for `operators.differential.laplacian`:
  on an endpoint-inclusive grid the wall lies *on* the end node, that node owns `h/2`, and the mass
  is the trapezoid integral. Measured through the shipped routines against an exact heat solution,
  the wall goes from EOC 0.73 / 0.87 / 0.94 to **2.00 / 2.00 / 2.00**; `FPSLSolver` on #2237's MMS
  goes from 1.989e-03 to 2.806e-05, landing exactly on `FPSLJacobianSolver`, which had the correct
  wall all along.

  **Every value function and every density produced by this family moves.** A mass check written as
  `sum(m) * dx` will now report drift that is the convention's and not the solver's — use
  `TensorProductGrid.integrate` / `quadrature_weights_1d`, the owner of that quadrature. Measured
  through a real `FPSLSolver` solve: the grid measure is held to 3.0e-14 where it used to drift by
  1.1e-02, and the rectangle rule is now the one that drifts.

  Also fixes a seventh implementation of the same wall that #2237's census could not see —
  `adjoint.operators._build_1d_laplacian`, which holds the wall without holding `alpha` — and a
  wall row in `build_diffusion_matrix_2d` that carried the stencil's diagonal against the
  interior's off-diagonal, invisible while the two were equal.
