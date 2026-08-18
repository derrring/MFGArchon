`HJBWENOSolver.solve_hjb_system` accepts `source_term`, so a manufactured solution can reach it.
MMS forcing enters through that slot and most HJB solvers do not accept it, which made WENO's order
unmeasurable rather than merely unverified. The forcing enters each SSP-RK3 stage at that stage's
own time (backward stage times t, t-dt, t-dt/2); added once after the step it is explicit Euler for
the source and caps the whole scheme at first order. `x` follows the documented `(N, d)` contract
from the abstract base, so one manufactured source runs against FDM and WENO alike. Multi-D refuses
the argument rather than applying it once per axis sweep. Measured with the 1D reduction of the GFDM
paper's manufactured pair: HJ-WENO5 delivers fifth order (EOC 5.41, 5.04, 5.01 at sigma=0.0005),
while the scheme's overall order is capped at 2 by the central second-difference diffusion (EOC 2.00
at sigma=1). The two error terms have opposite signs and cancel at intermediate sigma, which is why
an intermediate sweep shows an inflated 5.43 followed by 0.86 rather than a clean crossover.
