`AdvectionOperator.as_scipy_sparse()` refuses `scheme="upwind"` instead of returning a matrix that
is not the operator (#1981).

#1981 reported that `form=` is inert under upwind, so the public `conservative=` flag "silently does
nothing". Measured, that is the symptom rather than the defect — narrower in one direction and much
wider in another:

- **The operator honours `form=`.** `__call__` / `@` separate `divergence` from `gradient` by 48.97
  on a linear velocity, and the public flag reaches it: 9.13 through
  `grid.get_advection_operator(...) @ m`.
- **The matrix does not represent the operator at all**, not merely form-insensitively. Against
  `__call__` on a smooth field it is off by **27.31 for a CONSTANT velocity**, where the two forms
  coincide and there is no form to lose.

| velocity | scheme | form | max&#124;A @ m − op(m)&#124; |
|---|---|---|---|
| constant 1 | centered | either | 0.000000 |
| constant 1 | **upwind** | either | **27.313708** |
| linear 1+2x | centered | either | 0.000000 |
| linear 1+2x | **upwind** | divergence | **80.000000** |
| linear 1+2x | **upwind** | gradient | **47.798990** |

Upwinding chooses its difference direction from the local sign of the field, so the operator is
**nonlinear and has no matrix**; probing it with unit vectors linearises it around impulses.
`centered` is exact, so the extraction machinery is sound — the refusal is scoped to the scheme, not
to the method.

The method's docstring already said "❌ Do NOT use for implicit solver Jacobians". **A recommendation
the code does not enforce** is the shape this repository keeps finding, so it raises now, and the
message says why there is no matrix and what to use instead (`centered`, or a velocity-sign upwind
construction, which is linear).

Safe because nothing takes that path: every `as_scipy_sparse()` call site in the package is on
`LaplacianOperator`. Pinned by a census, so adding an advection one becomes a decision rather than an
accident — with a positive control confirming the census fires when a caller is introduced.
