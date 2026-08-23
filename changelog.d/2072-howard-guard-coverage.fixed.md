Howard's decomposition guard had two coverage holes that made it **accept what it exists to refuse**
(#2072). Both are pre-existing on `main`, independent of #2069.

**The momentum ladder stepped over the operating range.** `_mags` was
`sorted({0.5, 1, 2, 0.5*_gT, _gT, 2*_gT})`, which on this file's own fixture is
`{0.5, 1, 2, 20, 40, 80}` — while the solve's `max|∇u_T|` for `u_T = cos(2πx)` is `2π = 6.2832`,
sitting in the unsampled `(2, 20)` gap. `_gT` is a *spacing* bound, not a gradient, and overestimates
by 6.4× here, which is what pushed the upper rungs past the hole the widening was meant to close.

Measured on `H = |p|²/2 + bump(|p|²)` with the bump supported on `|p| ∈ (3, 10)` and **no** alpha-free
part, so `_af_bad` cannot be what catches it:

| | old ladder | geometric span |
|---|---|---|
| the bumped Hamiltonian | **ACCEPTED** — finite, plausible, wrong | **REFUSED** |
| `H = \|p\|²/2` exactly (control) | ACCEPTED | ACCEPTED |

The ladder is now `np.geomspace(0.5, max(2*_gT, 2), 12)` unioned with `{0.5, 1, 2}`. A span cannot
have that hole. Overestimating `_gT` stays safe for the original reason — a genuine quadratic matches
the kinetic reference at *every* `|p|`, so more sample points can only make a true refusal stricter,
never invent a false one — and the existing suite confirms it: **13 passed before, 16 after**, with no
previously-accepted Hamiltonian newly refused.

**A single NaN defeated the guard entirely, and the comment claimed the opposite.** It said
`np.maximum` was used rather than the builtin because `max(0.0, nan)` returns `0.0` and would zero
the measurement — "a fail-silent inside the fail-loud guard". It does not prevent that:

```
np.maximum(0.0, nan) = nan     nan > tol = False   -> ACCEPT
max(0.0, nan)        = 0.0     0.0 > tol = False   -> ACCEPT
```

Identical. Measured on a quartic returning NaN only at `|p| = 80`, a probe magnitude the solve never
visits: **accepted**, all-finite output, **153% wrong** against Newton. The guard now refuses on a
non-finite `_af`, `_ke` or `_scale` before comparing anything to a tolerance — a probe that could not
be evaluated is not a probe that passed.

**Not changed here**, and #2072 flags it as a separate decision: `hjb_howard.py:506` returns the
previous timestep with a `logger.warning` when the policy iterate is non-finite. That fallback is
what turns a poisoned solve into a plausible finite field rather than a visible NaN, but it has five
test references and changing a fallback to a raise is a behaviour change owing its own review.

**#2069 is parked as a draft on this**; it removes the `_af_bad` refusal, which was incidentally
covering the bumped class for an unrelated reason.
