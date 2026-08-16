Three shipped declarations said the FP-FDM path supports Robin. It does not: the boundary handlers
are not passed `boundary_conditions` at all, so a ROBIN segment assembles byte-identically to
no-flux (measured at alpha=3.2 and alpha=999 across 768 configurations). Corrected
`solve_fp_system_1d`'s deprecation docstring, the legacy-BC diagnostic that recommended `robin_bc`
on a path that cannot assemble it, and `robin_bc`'s own docstring.

`robin_bc` now also says what none of the three said and what matters most in practice: **you do
not need a ROBIN segment for a reflecting wall.** The conservative schemes -- `divergence_upwind`
(the default) and `divergence_centered` -- impose `J.n = 0` structurally by zeroing the total face
flux, conserving mass to machine precision at a wall with wall-normal drift; the `gradient_*`
family imposes `d_n m = 0` instead and loses 75-78% (non-conservative by design, #1075). Handing
the FEM path `alpha = v_n` explicitly double-counts the same term. Reach for `robin_bc` only for a
wall that is not the reflecting one. (#1975)
