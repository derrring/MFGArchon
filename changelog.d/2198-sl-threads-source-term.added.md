`HJBSemiLagrangianSolver` now threads `source_term` through its operator-splitting path, making the
Semi-Lagrangian family reachable by the method of manufactured solutions for the first time. The
forcing enters as a rate, multiplied by the sub-step and evaluated at that sub-step's own physical
time with the same sign as `k = -u_t` — the convention `HJBWENOSolver` already documents, so one
manufactured source runs against both time-stepping solvers.

The three variants that replace the splitting path rather than adding to it — `canonical_cs`, the
L-based DPP path, and `stochastic` — name the parameter and raise `NotImplementedError`. Where the
forcing enters an implicit-alpha* fixed point or a sampled characteristic has not been derived or
measured, and a manufactured source in the wrong place verifies a different equation while still
converging.

This closes the disjointness #2198 measured: the only family that discretises anisotropic
cross-derivatives was the only one MMS could not reach. First measured orders, 2-D no-flux,
`u* = a(t)cos(pi x1)cos(pi x2)` (a product, so its cross derivative is non-zero):

- isotropic `sigma = 0.3` — EOC 1.039, 1.017
- off-diagonal `S = [[0.30, 0.12], [0.12, 0.24]]` — EOC 1.040, 1.017

The second is the first anisotropic accuracy measurement in this repository, and it answers the
warning the ADI step emits for that configuration ("full tensor with off-diagonal terms ... may be
inaccurate"): first order is preserved at these resolutions. Discrimination shown by mutation —
dropping the cross-derivative term collapses the tensor EOC to 0.718, 0.385 while leaving the
isotropic leg bit-unchanged, which is the study's built-in control.

The `#1079` warnings in `hjb_fdm` (both sites) and the `HJBGFDMSolver` refusal now name that
alternative with its measured order, instead of offering only "change your model".
