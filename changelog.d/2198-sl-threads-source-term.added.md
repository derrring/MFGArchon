`HJBSemiLagrangianSolver` now threads `source_term` through its operator-splitting path, making the
HJB half of the Semi-Lagrangian family reachable by the method of manufactured solutions for the
first time. The FP half is NOT: `FPSLAdjointSolver` still does not thread a source, so "the SL
family is MMS-reachable" would be false as an unqualified claim. The
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

The second is the first measurement of anisotropic ORDER in this repository -- order, not accuracy:
it says the scheme converges at first order with an off-diagonal Sigma present, not that the
cross-derivative stencil is accurate, and it answers the
warning the ADI step emits for that configuration ("full tensor with off-diagonal terms ... may be
inaccurate"): first order is preserved at these resolutions. Discrimination shown by mutation —
dropping the cross-derivative term collapses the tensor EOC to 0.718, 0.385 while leaving the
isotropic leg bit-unchanged. That unchanged leg establishes only that the mutation touched nothing
outside the cross term (a diagonal Sigma has none to drop); it is not evidence that the tensor leg
is measuring the stencil correctly, and it does not make the pair a control for anything else.

The `#1079` warnings in `hjb_fdm` (both sites) and the `HJBGFDMSolver` refusal now name that
alternative with its measured order, instead of offering only "change your model".
