`HJBWENOSolver.solve_hjb_system` accepts `source_term`, so a manufactured solution can reach it.
MMS forcing enters through that slot and 8 of 11 HJB solvers did not accept it, which made WENO's
order unmeasurable rather than merely unverified. The forcing enters each SSP-RK3 stage at that
stage's own time; added once after the step it is explicit Euler for the source and caps the whole
scheme at first order. Measured with the 1D reduction of the GFDM paper's manufactured pair: EOC
2.00 at sigma=1 (the central second-difference diffusion sets the rate) and 5.70/5.43 at sigma=0.05
(HJ-WENO5 sets it). Multi-D refuses the argument rather than applying it once per axis sweep.
