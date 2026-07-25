- **The coupled 2D and FVM test fixtures state the drift they actually run at** (Issue #1442
  follow-up). Issue #1442 made the FP solvers read the drift coefficient `c = 1/lambda` from the
  Hamiltonian instead of a private default, so fixtures carrying `control_cost=1.0` silently
  changed drift: the 2D coupled fixture went from 0.5 to 1.0 and six tests began failing with a
  NaN value function, and the 1D FVM fixture went from its stated 0.3 to 1.0 and needed 52 Picard
  iterations against a 30-iteration assertion. Both now set `control_cost` explicitly, and the 2D
  tests assert `result.converged` so a fixture whose calibration rots again cannot pass on an
  unconverged solve.
