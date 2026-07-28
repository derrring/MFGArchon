Network FP solves no longer impose mass conservation. `FPNetworkSolver(enforce_mass_conservation=)`
is removed and raises on any value (#1683): it normalised the caller's initial condition to
unit mass and divided every step by its total, so a caller passing a density of mass 5.0
silently got the evolution of one of mass 1.0, and the returned density was exactly
conserving whatever the scheme produced. Conservation is a property of the discretisation --
this scheme has it by construction, verified across 504 configurations spanning grid, random
and scale-free topologies, both schemes, and four diffusion scales -- so the flag did nothing
on a healthy solve and forced a false answer on a broken one. Drift is now logged at WARNING
instead. The clip that preceded the division routes through `clip_nonnegative_or_raise` and
stops the solve when it would fabricate mass: a timestep past the CFL limit clips 60.1% of the
present mass on every step. This path passes its own threshold rather than the shared default,
measured -- its discretisation noise reaches 6.7e-06 fabricated on solves whose honest answer
is a drift below 1e-3, while genuine failures start at 9.2e-04.
