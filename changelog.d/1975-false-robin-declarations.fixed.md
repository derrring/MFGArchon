Three shipped declarations said the FP-FDM path supports Robin. It does not: the boundary handlers
are not passed `boundary_conditions` at all, so a ROBIN segment assembles byte-identically to
no-flux. Corrected `FPFDMSolver._solve_fp_1d`'s deprecation docstring, the legacy-BC diagnostic
that recommended `robin_bc` on a path that cannot assemble it, and `robin_bc`'s own docstring.

That byte-identity is reachable only **below** the `_validate_bc_support` gate, which refuses every
ROBIN segment at construction -- uniform and mixed alike -- so it is a property of the assembly
rather than something a caller reaches through the API.

`robin_bc` now says what none of the three said and what matters most in practice: **you do not
need a ROBIN segment for a reflecting wall.** The conservative schemes -- `divergence_upwind` (the
default) and `divergence_centered` -- impose `J.n = 0` structurally by zeroing the total face flux,
conserving mass to machine precision at a wall with wall-normal drift. The `gradient_*` family
imposes `d_n m = 0` exactly, at every T and Nx measured, and therefore loses essentially all the
mass: about -78% at T = 0.20, -99% at 0.30, -99.99% by 0.50 (sigma = 0.3, 81 points on [0,1],
dt = 1e-3, Gaussian initial density centred at 0.5, drift 3.2). The percentage is a function of T
and of the initial condition, so it is quoted to the precision the stated configuration supports;
the `d_n m = 0` mechanism behind it is not T-dependent.

Adding a Robin segment on top of the conservative wall **destroys** it rather than restating it:
`A_robin` contributes a residual outflux `J.n = D*(alpha/beta)*m`, so the implied wall is
`D d_n m = (v_n - D*alpha/beta) m`. The reflecting condition's own coefficients are
`(alpha, beta) = (D_pH.n, D)`, and since this library's FP velocity is `v = -D_pH`, that is
`alpha = -v_n` -- verified by residual on the structurally reflecting solution rather than by
substitution. So `D*alpha/beta = -v_n`, the row that DOUBLES: encoding the reflecting condition as
a Robin segment on a wall that already imposes it is unbounded (+1.7e31% measured), not merely
leaky. (#1975)
