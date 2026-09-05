- **The SL/CN family's no-flux wall is the mirror stencil, and the FP transport moves with it**
  (Issue #2243). Every implicit-diffusion call site now takes `treatment="mirror"`, applying to the
  Semi-Lagrangian / Crank-Nicolson family the decision #2145 had already made for
  `operators.differential.laplacian`: on an endpoint-inclusive grid the wall lies *on* the end node,
  that node owns `h/2`, and the mass is the trapezoid integral. Measured through the shipped
  routines against an exact heat solution, the wall goes from EOC 0.73 / 0.87 / 0.94 to
  **2.00 / 2.00 / 2.00**; `FPSLSolver` on #2237's MMS goes from 1.989e-03 to 2.806e-05, landing
  exactly on `FPSLJacobianSolver`, which had the correct wall all along.

  **`FPSLSolver`'s splat moved to the same measure, and had to.** Its two half-steps must agree or
  the composite conserves neither: the splat kernels are exact transposes of interpolation and
  conserve `sum(m)`, which is what #708 paired with the `half_wall` diffusion of the day. With a
  driven transport (`u = -0.5x`, T=1), the relative trapezoid drift over nx = 26 … 401 reads
  8.2e-02 … 1.3e-02 for the old pairing, 6.3e-03 … 7.6e-02 for the crossed one, and
  **2.5e-15 … 2.7e-13** once both halves carry the grid measure. `splat_1d` and `splat_nd` now
  transport `w * m` and divide back.

  **Every value function and every density produced by this family moves.** A mass check written as
  `sum(m) * dx` will now report drift that is the convention's and not the solver's — use
  `TensorProductGrid.integrate` / `quadrature_weights_1d` / `quadrature_weights_nd`.

  Also fixes: a seventh implementation of the same wall that #2237's census could not see
  (`adjoint.operators._build_1d_laplacian`, which holds the wall without holding `alpha`); a wall
  row in `build_diffusion_matrix_2d` that carried the stencil's diagonal against the interior's
  off-diagonal, invisible while the two were equal; and a `build_diffusion_matrix_1d` docstring
  claiming a Dirichlet symmetry the builder has never had.

- **`quadrature_weights_nd`** (Issue #2243). The tensor-product control volumes, added beside
  `quadrature_weights_1d` so the measure has one owner in every dimension. `TensorProductGrid.integrate`
  and the SL splatting dispatchers both route through it; `d = 1` is not a special case.
