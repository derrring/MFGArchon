- **HJB-WENO reads a time-dependent boundary value at each Runge-Kutta stage's own time** (#2605; #2512, row B2).
  - Its ghost fill had no time, so `PreallocatedGhostBuffer.update_ghosts`'s default read a `g(t)` value at t = 0 at
    every stage of every step, silently. WENO refuses a Dirichlet BC, so the case is a time-dependent Neumann value.
  - Each stage now reads it on the clock the MMS source already used: the backward SSP-RK3 stages at t, t - dt and
    t - dt/2, from the later end of each sub-step.
  - The dimensional split reads each sweep over the interval it covers. Strang's half steps cover
    [t, t - dt/2] and [t - dt/2, t - dt], and its full step covers [t, t - dt].
  - Measured with u = a(t) cos(pi x) + b(t) x, a = 1 + t/2, b = 1/2 + t, and the outward derivative -/+ b(t) on
    each wall (T = 0.5, Nt = 10, sigma = 0.4). The max error at t = 0 for n = 11, 21, 41 was:
    - at 52219be2: 2.21e-2, 4.26e-2, 4.63e-2, which does not converge;
    - now: 1.82e-3, 8.05e-4, 2.13e-4.

    At n = 21 to 161 the order is 2.00, the same as with b constant, so stage-time boundary data costs no order
    here.
  - `HJBWENOSolver.solve_hjb_step` takes `time`, the time the step starts from. Without it, a value that declares
    `t` is refused.
