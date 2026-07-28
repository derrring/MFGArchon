Network FP solves no longer impose mass conservation. `FPNetworkSolver(enforce_mass_conservation=)`
is removed and raises on any value (#1683): it normalised the caller's initial condition to
unit mass and divided every step by its total, so a caller passing a density of mass 5.0
silently got the evolution of one of mass 1.0, and the returned density was exactly
conserving whatever the scheme produced. Conservation is a property of the discretisation --
this scheme has it by construction, measured at 0.00-0.01% drift over 20 steps -- so the flag
did nothing on a healthy solve and forced a false answer on a broken one. Drift is now
reported as a `RuntimeWarning` instead. The clip that preceded the division routes through
`clip_nonnegative_or_raise` and stops the solve when it would fabricate mass: measured on a
5x5 grid network, a steep value field clipped 1.66% of the present mass on 19 of 20 steps and
dt past the CFL limit clipped 60.1% on every step.
