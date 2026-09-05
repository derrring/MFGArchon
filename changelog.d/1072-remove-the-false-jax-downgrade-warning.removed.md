`warn_if_jax_scheme_downgraded` and the `_effective_scheme` plumbing that fed it (#1072).

The warning told users that selecting the JAX backend silently replaced their requested stencil with
2nd-order central differences. That was never true. It named `hjb_step` / `fpk_step`, deleted from
every backend on 2026-08-17 (`abd5e214`) — and at #1072's own filing commit `b38ee119` the NumPy
backend's `hjb_step` computed central differences too, with `WENO`/`upwind` appearing zero times in
that file, so the asymmetry the issue reports did not exist. Neither stepper was reachable from a
solve: `.hjb_step(` / `.fpk_step(` had no call site under `mfgarchon/alg` or `mfgarchon/core` at that
commit, which `base_backend.py` now records in its own words ("called by no solver in the package").

A backend decides what array type flows through a computation and nothing else, so it cannot change a
discretization. The warning guarded a condition the package has never been able to reach.
