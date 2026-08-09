- **`NewtonSolver`'s finite-difference Jacobian no longer straddles a kink** (Issue #1745). An
  upwind HJB residual selects between two one-sided differences, so it is non-differentiable where
  they tie, and symmetric initial data puts nodes exactly there rather than near there: on the 2-D
  smoke problem the grid centre's neighbours agree to 9e-11 and `dF[60]/dU[59]` is `+13.395` from
  one side and `-7.9998` from the other. The forward quotient reported only the branch on its own
  side, and the resulting step was one the line search cut to ~0.03 -- the inner solve creeped to
  its 30-iteration budget at `1.17e-05` and needed 67 iterations to reach a `1e-6` tolerance. The
  symmetric quotient reports the mean of the two branch slopes and reaches `3.14e-09` in 5. On the
  2-D FDM coupled solve, twelve of the thirteen inner solves now converge in 4-7 iterations with
  final residuals from `4.9e-12` to `7.4e-07`, where before the fifth stalled. The cost is `2n`
  residual evaluations per Jacobian instead of `n`. No capability cell changes status: the 2-D cells
  still fail, the thirteenth inner solve being where the coupled iteration has already diverged
  (Issue #1865) -- for `fdm_upwind_2d` and `fvm_muscl_2d` that failure is still reported by the
  inner Newton, while `fdm_centered_2d` now gets far enough to fail in the FP step instead.
