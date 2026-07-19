**Breaking (pre-1.0, no shim): `EnergyFunctional` contract frozen and `energy()` quadrature repaired** (Issue #1642, capabilities A1/A2/A3).

`energy()` returned `F[m] / cell_volume` instead of `F[m]` — measured as exactly
`1/dx` (31, 63, 127, 255, 511 at N=32..512, dx=1/(N-1)), so the value doubled
with every refinement instead of converging. `QuadraticInteractionEnergy.energy`
and `PotentialEnergy.energy` now apply the outer quadrature weight and return the
physical integral. The live HJB source path was **not** affected: it calls only
the first variation, which was already exact (bit-identical to the physical
`delta F / delta m` on a scattered non-uniform cloud).

Contract changes on `EnergyFunctional`, all pre-1.0 breaks with no compatibility
shim:

- `lions_derivative` -> `flat_derivative`. The method returns the flat
  (linear-functional) derivative `delta F / delta m`, a scalar field of shape
  `(N,)`; the Lions/Wasserstein derivative is `grad_x delta F / delta m`, shape
  `(N, d)`. Return shapes differ, so the old name named the wrong object.
- New required `weights` member: the quadrature weights of the measure
  representation — cell volumes on a grid, particle masses for an empirical
  measure. Documented explicitly; both ontologies take the same code path.
- All methods take a keyword-only `t` (the source pipeline already carries the
  true per-slice `t`; the analytic branch previously discarded it).
- New required `second_variation(m, *, t=0.0) -> NDArray | None` slot, defaulting
  to `None`. `QuadraticInteractionEnergy` returns the kernel matrix `K`;
  `PotentialEnergy` returns zeros.
- Single-population by decision: a stacked `(K*N,)` multi-population density is
  now refused with a diagnostic naming the expected shape, instead of being
  contracted against a broadcastable operand.
- `PotentialEnergy(potential)` -> `PotentialEnergy(potential, weights)`; weights
  are required, since a default would silently reintroduce mesh-dependence.
- `CombinedEnergy` refuses components whose quadrature weights disagree.

New: `flat_derivative_from_energy_gradient(gradient, weights)` is the single
owner of the factor between the two derivative conventions —
`FiniteDifferenceFunctionalDerivative` perturbs an unweighted Dirac
`m_k += epsilon`, so its output is `w_k * (delta F / delta m)_k`.

**Breaking: `create_lions_source`'s finite-difference path now requires
`weights=`.** Both branches of the function now return the same object, the
pointwise `delta F / delta m` of the *physical* `F[m]`. Previously the FD branch
returned the raw entry gradient `w_k * (delta F / delta m)_k` and the analytic
branch the pointwise derivative, so feeding one energy object through the two
documented paths disagreed by exactly the cell volume — an interaction coupling
that weakened under refinement with no error. The FD branch now applies
`flat_derivative_from_energy_gradient`, which requires it to know the quadrature
weights. There is no default: `w = 1` is what silently reintroduces the fork.
`create_lions_source(F, fd)` -> `create_lions_source(F, fd, weights=w)`, with
`w` a scalar cell volume or an `(N,)` array; passing `weights=` alongside an
`EnergyFunctional` is refused (it already carries them).

An object providing *some* but not all `EnergyFunctional` members is now refused
by `create_lions_source` with the missing members named, instead of falling
through to the finite-difference path — where, if it happened to be callable,
its exact `flat_derivative` was never called and nothing said so.

`EnergyFunctional.weights` is now a read-only array. `validate_weights` returns a
frozen copy rather than a view, so the public member can no longer alias operator
internals: `E.weights[0] = 99.0` reached `conv.as_dense()[0, 0]` and defeated the
strict-positivity precondition; it now raises. New introspection helpers
`energy_functional_members()` / `missing_energy_functional_members(obj)` expose
the required member set for diagnostics.

`ConvolutionCouplingOperator` gains `weights` (always an `(N,)` array) and
`kernel_matrix()` (the unweighted `W`); the scalar-or-array `cell_volume`
property is removed in favour of the single spelling. The `cell_volume=`
constructor keyword is unchanged.
