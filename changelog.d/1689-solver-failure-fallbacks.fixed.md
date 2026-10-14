- **A failed inner solve no longer returns a value the solve did not produce** (#1689 member 4).
  - `HJBSemiLagrangianSolver`'s n-D DPP step used U_next(x) wherever L-BFGS-B reported `success=False`. That is the
    cost of standing still without its running cost dt * L(x, 0, m, t). With a non-smooth control cost the minimiser
    starts at the kink and often reports failure, so a constant potential V shifted U by less than V * T: up to 1.004
    against 1.0 at V = 2 on a 2-D 11 x 11 fixture. The step now takes the cost of the control the minimiser
    returns, whatever `success` says.
  - `NetworkHJBSolver`'s ODE path warned when `solve_ivp` stopped short of t = 0, then filled every level it had not
    reached by extrapolating the trajectory. It now raises `NumericalInstabilityError`, naming the last time level
    it computed and the integrator's message.
  - `DualityStatus.DISCRETE_DUAL` and `CONTINUOUS_DUAL` are documented as what they are computed from, the two
    solvers' scheme families. The assembled operators are not compared (#2671).
