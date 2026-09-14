- **`HJBFDMSolver(advection_scheme="gradient_upwind")` refuses a Hamiltonian its momentum rule is not
  Godunov for** (Issue #2311). The rule is exact when H is even in each momentum component and
  nondecreasing in its magnitude. At construction H is probed along each axis at two points, two
  positive densities and four momentum magnitudes. A violation raises `NotImplementedError` naming
  #2311: e.g. a `DualHamiltonian` whose Lagrangian is minimised away from 0, or a
  `CongestionHamiltonian` with `c(m) < 0`. Those used to solve to a finite, wrong answer. The probe is
  not a proof, and an H that violates the condition only away from the probed momenta is not caught.
  A monotone scheme for such H is #2316.
