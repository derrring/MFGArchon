`AdvectionOperator.as_scipy_sparse()` under `scheme="upwind"` returns the operator's matrix (#1981, #2309).

#1981 reported that `form=` is inert under upwind. Measured at `cf6f40d3`, the extracted matrix did
not represent the operator at all: against `__call__` on a smooth field it was off by **27.31 for a
CONSTANT velocity**, where the two forms coincide.

| velocity | scheme | form | max&#124;A @ m − op(m)&#124; at `cf6f40d3` |
|---|---|---|---|
| constant 1 | centered | either | 0.000000 |
| constant 1 | **upwind** | either | **27.313708** |
| linear 1+2x | centered | either | 0.000000 |
| linear 1+2x | **upwind** | divergence | **80.000000** |
| linear 1+2x | **upwind** | gradient | **47.798990** |

Upwinding chose its difference direction from the sign of the field, so the operator was nonlinear,
and probing it with unit vectors linearised it around impulses. #2031 made the method refuse
`scheme="upwind"`. #2309 makes the upwind operator select by the sign of the velocity, so it is
linear in the field and the refusal is gone: every upwind row above is 0.000000, for both forms, with
`bc=None` and with no-flux. The matrix is still not the operator under an inhomogeneous Dirichlet value, for
either scheme, because the operator is then affine.
