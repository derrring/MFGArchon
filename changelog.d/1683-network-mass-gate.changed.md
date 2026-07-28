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
present mass on every step. This path passes its own threshold rather than the shared
default: the shared 1e-8 was measured refusing solves whose honest answer is a 5.8e-5 drift.
The replacement was chosen against 3704 configurations spanning grid, random and scale-free
topologies -- no broken solve is admitted, and the solves it still refuses all have a true
drift between 1e-4 and 1e-3.
