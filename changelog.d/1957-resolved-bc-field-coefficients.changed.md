**`FPResolver` resolves the impermeable wall instead of renaming it.** `NO_FLUX` and
`REFLECTING` now become `MathBCType.ROBIN` carrying the coefficients of the flux condition:
`J.n = 0` with `J = v*m - D*grad(m)` is `v_n*m - D*d_n m = 0`, i.e. `alpha = v_n`, `beta = -D`,
`g = 0`. Verified against the condition itself, not against the previous code path: over 16
coefficient sets the resulting ghost leaves `|J.n| <= 3.6e-15`.

They previously became `MathBCType.ZERO_FLUX`, whose own enum comment read *"needs
drift+diffusion for calculator"* — the physical intent passed through under a second name, so
Layer 3 had to dispatch on it again. That member is **removed**; the enum's docstring already
said it "contains no physical intent types", and `ZERO_FLUX` was the exception to its own rule.
`ZeroFluxCalculator` itself stays, still reached through
`bc_to_topology_calculator(use_zero_flux=True)`.

**`ResolvedBC` coefficients may now be fields.** `alpha`, `beta` and `value` were declared
`float`, and that is what blocked the resolution above: the Robin coefficient of a reflecting FP
wall is `D_pH(x, grad u) . n`, which varies along the boundary and is recomputed every Picard
iterate. A float cannot hold it. `ghost_cell_robin` consumes field coefficients as well — its
arithmetic was already elementwise and only the singularity guard assumed a scalar; it now
reports how many boundary points are singular.

**No default for a mathematical parameter.** Resolving an impermeable wall without `drift` or
`diffusion` in `solver_state` raises, naming the missing key. A defaulted drift makes the wall
pure Neumann, which conserves mass at a tangential wall and leaks at every other one — a wrong
answer that still converges. An unrecognised `BCType` likewise raises rather than defaulting to
a zero-flux wall, which was an impermeable boundary imposed on a condition nobody wrote.

Three tests in `test_resolution.py` are re-pointed rather than deleted: they asserted the
`ZERO_FLUX` passthrough, which was the behaviour under replacement.
