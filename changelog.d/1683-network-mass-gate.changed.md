Network FP solves stop when a non-negativity clip would fabricate mass, instead of clipping
and then dividing by the total (#1683). Measured on a 5x5 grid network, a steep value field
clipped 1.66% of the present mass on 19 of 20 steps and dt past the CFL limit clipped 60.1%
on every step -- all returning exactly unit mass. `enforce_mass_conservation` is kept: the
graph inflow-outflow form was measured to conserve to 0.00-0.01% without it, so unlike the
GFDM case (#1752) it is not masking a non-conservative scheme.
