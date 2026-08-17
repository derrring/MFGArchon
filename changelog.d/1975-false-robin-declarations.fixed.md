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
mass: -78.05% at T = 0.20, -98.98% at 0.30, -99.996% by 0.50 (sigma = 0.3, Nx = 81, drift 3.2).
The percentage is a function of T and is now quoted with one; the mechanism behind it is not.

Adding a Robin segment on top of the conservative wall **destroys** it rather than restating it:
`A_robin` contributes a residual outflux `J.n = D*(alpha/beta)*m`, so the implied wall is
`D d_n m = (v_n - D*alpha/beta) m`. The reflecting condition's own coefficients
`(alpha, beta) = (v_n, D)` give `D*alpha/beta = v_n` and hence `d_n m = 0` -- the non-conservative
wall -- so encoding the condition as a `robin_bc` lands on exactly what it warns against. (#1975)
