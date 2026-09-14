- **`gradient_upwind` is the Godunov (Rouy-Tourin) upwind momentum** (Issue #2308). At a discrete
  local minimum (backward difference < 0 < forward difference) it returns 0. It used to select on
  the sign of the central difference, which returned the one-sided difference of smaller magnitude
  there: a non-monotone numerical Hamiltonian that Newton still converges to without a warning. On
  the 1-D `HJBFDMSolver` manufactured solution at Nx=11, sigma=0.2 the error falls from 6.46 to 0.24.
  The rule is Godunov exactly when H is even in each momentum component and nondecreasing in its
  magnitude, which every `SeparableHamiltonian` and `CongestionHamiltonian` control cost satisfies.
  The HJB Jacobian reads a zero row at such a minimum.
