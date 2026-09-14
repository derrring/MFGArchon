- **`HJBFDMSolver(advection_scheme="gradient_upwind")` refuses a Hamiltonian its momentum rule is not
  Godunov for** (Issue #2311). The rule is exact when H is even in each momentum component and
  nondecreasing in its magnitude. At construction H is probed along each axis, and a violation raises
  `NotImplementedError` naming #2311: e.g. a `DualHamiltonian` whose Lagrangian is minimised away from 0,
  or a `CongestionHamiltonian` with `c(m) <= 0`. Those used to solve to a finite, wrong answer.
  `advection_scheme="gradient_centered"` still runs them.
