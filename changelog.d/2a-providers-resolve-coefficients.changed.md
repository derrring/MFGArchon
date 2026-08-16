A `BCValueProvider` may now sit on `BCSegment.alpha` or `.beta`, not only on `.value`, and its
result is no longer coerced with `float()`.

The impermeable wall of a Fokker-Planck equation is Robin with `alpha` the outward normal drift,
`D_pH(x, grad u) . n` — knowable only from the current iterate, which is what a provider is for,
and living on `alpha` rather than on `value`. `with_resolved_providers` resolved `value` alone, so
the one coefficient that needs the iterate could not be supplied, and could not have been a field
if it were.

`BCValueProvider.compute` has always declared `-> float | NDArray[np.floating]`. This is the third
layer in a row found narrowing that back to a scalar on one field, after `ResolvedBC` (#1957).

`has_providers()` widens with it. It gates the fast path — `with_resolved_providers` returns
`self` unchanged when it is False — so a gate covering fewer fields than the resolver would hand
back an unresolved provider *object* where a number belongs, surfacing far downstream as a type
error inside a ghost formula.

Nothing that worked before moves: the shipped adjoint-consistent configuration (#574, #625)
resolves to the same values to ten decimals, the no-provider fast path still returns `self`, and
the full suite is unchanged at 6147.
