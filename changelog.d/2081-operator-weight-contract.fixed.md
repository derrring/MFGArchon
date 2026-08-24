- **`DifferentialOperator.get_derivative_weights` documented one weight convention and the default
  operator returned the other** (Issue #2081). The abstract contract stated the raw-value form
  `d|_i = sum_j w_j u_j`, under which a row sums to zero. `LocalRBFOperator` satisfies it;
  `TaylorOperator` (`method="direct"`) and `UpwindOperator` return weights multiplying deviations
  `u_j - u_center`, whose rows sum to O(1e+2)-O(1e+3). Nothing broke, because every live consumer
  independently re-derives the row closure in one of three spellings of the same algebra, and that
  closure is mandatory for a deviation operator and idempotent for a sum-rule one -- which is the
  invariant the stack rests on and was written down nowhere. The ABC and both overrides now state
  it, and a new test file pins the operators: exactness in 1D and 2D on an anisotropic quadratic
  and a per-axis linear field, each operator's convention, the closure invariant two-sided, and
  that assembling from the weights reproduces `op.laplacian` on a non-polynomial field. Before
  this, `grep -rl create_operator tests/` returned zero files, so a convention flip passed the
  suite in either direction.
