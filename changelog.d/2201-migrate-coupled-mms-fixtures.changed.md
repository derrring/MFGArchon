The two coupled-MMS integration fixtures (`test_coupled_mms_2d_no_flux.py`,
`test_coupled_mfg_mms.py`) now assemble their source terms through
`mfgarchon.utils.manufactured` instead of each deriving `S_HJB` / `S_FP` by hand. Hand-written
assembly arithmetic is now ZERO in both files; what remains in each is the exact pair and its
analytic derivatives. `test_mms_validation.ManufacturedSolution` is deliberately NOT subsumed — it
is a velocity-driven FP family that takes a velocity with no `u`, which is a different quantity.

Corrects a false claim in `test_coupled_mfg_mms.py`'s header, which described
`coupling_coefficient` as "an INDEPENDENT knob from lambda" that the fixture sets to match. It is
inert on that problem's solver path — the FDM FP/HJB families resolve the drift through
`fp_drift_coefficient`, which returns `1/control_cost.lambda_` for a quadratic-MINIMIZE
`SeparableHamiltonian` and never reaches the `coupling_coefficient` fallback. This is a scoped
claim, not a package-wide one: the velocity-channel FP families (FVM / FEM / meshless-Galerkin FP,
and the network solvers) resolve the drift through `H.optimal_control` and never call that helper.
The FP scope word matters — `meshless_galerkin/hjb_solver.py` does call it.
Measured at the solve and pinned there — a full coupled solve is bit-identical for
`coupling_coefficient` of 1.0 / 7.0 / 0.5 / -3.0, with a sigma control that moves it. The
agreement the header credited to setting the knob was never contingent on it.

Both fixtures gain a `check_pair` test: the pair's analytic derivatives audited against a finite
difference of `u*` and `m*`. This is the only check either fixture has whose oracle is outside both
the scheme and the assembly, and the only one that can see a wrong cross-derivative — under the
isotropic sigma both use, `tr(D . Hess)` multiplies every off-diagonal Hessian entry by exactly zero.
