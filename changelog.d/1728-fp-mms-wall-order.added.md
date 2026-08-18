An MMS order study for the FP-FDM advection schemes at a no-flux wall that carries drift (#1728).

The test cited as covering `divergence_upwind`'s spatial order solves at zero drift, so substituting
any other advection scheme leaves its output bit-identical and the stated order has never been under
test. This file measures it with the advection term active.

The construction is the zero-flux (Gibbs) pair: for any smooth potential `phi`, the drift
`b* = -grad(phi)` and density `m* = Z^-1 exp(-phi/D)` make the flux `J = b* m - D grad(m)` vanish
everywhere, so the source term is identically zero and the exact solution is stationary. No
`source_term` channel is needed, which matters because only two FP solvers accept one.

Two instances, and the pair is what separates the two defects `gradient_upwind` carries. With
`phi = -A.x` the drift is a constant vector, so the divergence and gradient interior forms coincide
and only the wall closure is under test; repointing `gradient_upwind`'s wall at the conservative
routine repairs it there completely (6.69e-1 -> 2.23e-2, EOC 0.937). With the GFDM paper's published
source-free instance `phi = A(cos K x1 + cos K x2)` the drift is not constant, `m div(alpha)` is
non-zero, and the same repointing leaves the scheme non-convergent (5.81e-1 -> 8.02e-1, EOC 0.108).
So the gradient form is not the FP operator with a bad wall; it is a different operator. Both
defects are pinned, each with its own retirement condition, verified to trip on its own fix and not
on the other's.

Measured, relative L-inf at T over Nx = 21/41/81 in d = 1: `divergence_centered` 1.99e-4 / 5.09e-5 /
1.29e-5 (EOC 1.97, 1.98); `divergence_upwind` 1.63e-2 / 8.53e-3 / 4.36e-3 (0.93, 0.97);
`gradient_upwind` flat at 5.10e-1 with mass drift -1.4e-1. d = 2 agrees.

The order tests assert an error LEVEL beside the order, because the order alone is a min over the
wall closure and the interior stencil: a centered interior still reads EOC 0.89/0.94 and a centered
wall 0.89/0.95, so neither is isolated along the axis the ratios are attributed to -- the same
misattribution #1728 was filed about, one layer in. The level separates them (library 1.63e-2 at
Nx = 21, centered-interior mutant 2.98e-2). The drift vector carries a negative component and the
study is parametrized over its sign, because with one sign only one branch of each upwind selection
is taken: a mutant that always upwinds from the left reproduces the library bitwise at A > 0 and
separates at A < 0, while the library's own numbers are unchanged by the flip (the problem is
mirror-symmetric).

Existing coverage this does not duplicate: `test_solver_bc_support_census_1975.py` already pins the
mass-drift half of the `gradient_*` wall on the same drive. New here are the accuracy form -- the
scheme does not converge at all -- the separation of the wall defect from the interior-form one,
and the magnitude: the constructor warning's "leaks O(1e-2)" is the zero-drift figure, an order of
magnitude below the 1.4e-1 measured at wall-normal drift 0.7.
