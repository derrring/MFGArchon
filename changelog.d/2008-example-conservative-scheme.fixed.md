`examples/basic/three_mode_api_demo.py` selected the non-conservative `gradient_upwind` advection
scheme in its expert-mode demonstration and lost 98.17% of its probability mass (#2008). It also did
not run to completion, before or after that line: `compare_schemes()` selects
`NumericalScheme.FDM_CENTERED`, whose FP half goes negative at timestep 3 — the mass-fabrication
gate then stops the solve rather than clip it, and the script died there, short of its own summary.
Measured: the script exits **1** on `main` and **0** here.

All four advection schemes on this problem, cap 60, one outcome each — the failures are not the
same failure, which is why the fix is a scheme and not a family:

| scheme | outcome |
|---|---|
| `gradient_centered` | aborts at timestep 5/20, density `-6.388e-06` |
| `gradient_upwind` | converges at 27, mass `1.000000 -> 0.018302` (−98.17%) |
| `divergence_centered` | aborts at timestep 3/20, density `-6.319e-06` |
| `divergence_upwind` | converges at 49, mass conserved to `5.1e-15` |

The gradient family is non-conservative at a no-flux wall (#1075, #2007), but *how* a given member
fails is a property of the configuration — `geometry/boundary/conditions.py` says exactly that above
its own leak table, which is measured at a different resolution.

In `examples/`:

- `advection_scheme` moves to `divergence_upwind`, the only one of the four that solves this problem.
- `compare_schemes()` reports the gate's refusal as an outcome instead of ending the script, and
  narrows on the gate's own wording. The gate raises a bare `ValueError`, which is also what a
  NaN blow-up and an unknown-scheme dispatch raise, so a broad `except` would print a library bug
  under the word "refused" and still exit 0. Verified by injecting
  `ValueError("Unknown advection_scheme: ...")`: it propagates. The match is `"would fabricate"`,
  not `"fabricate"` — `geometry/boundary/conditions.py` raises a message containing the shorter
  string, and four assertions in the default test tier pin the longer one, so a rename breaks CI
  before it could silently re-broaden the clause.
- Every mode prints `result.mass_conservation_error`, through a helper that renders `None` as
  "not measured" — the field is `None` when the geometry has no volume element, or when the coupling
  loop breaks on iteration 1 with a non-finite `U`, and `solve` returns normally in that case, so an
  unguarded format would raise `TypeError` on the very line meant to make failure visible.
- `max_iterations` 20 -> 60 at all four sites, previously one. The loop already computed
  `mass_conservation_error` (`coupling/fixed_point_iterator.py:973`, since #1672) — it read **0.9817**
  on the leaking configuration against `5.1e-15` on the fixed one — but nothing gates on it and
  nothing printed it, so a 98% drain and a converged residual looked identical from the output.
- The `SUMMARY` reports whether the three modes agreed, **checked rather than asserted**. Levelling
  the caps made all four solves identical to every printed digit, but that is a fact about this
  problem: Auto Mode picks its scheme at runtime from the geometry, so an unstructured mesh sends
  it to FEM while Safe Mode stays pinned to `FDM_UPWIND`.
- **The exit status now carries convergence.** `problem.solve` returns normally when the coupling
  loop breaks early on a non-finite `U`, so a run in which nothing was solved previously printed
  "All three modes produced solutions", explained that their agreement "is the point", recommended
  a mode and exited **0** — #2008's own shape one level up. `main()` returns 1 if any mode did not
  converge, and the entry point propagates it. Verified by injecting a NaN into the HJB solve: exit
  **1**, with the mass line reading "not measured" rather than raising.

In `docs/user/three_mode_api_migration_guide.md`, a separate change: both **"After"** blocks — the
code the guide tells users to migrate *to* — selected `gradient_centered`. They move to
`divergence_upwind`, and Option 1 keeps `NumericalScheme.FDM_CENTERED` so its `fp_config` override
is a genuine non-default rather than a restatement of the factory default (verified:
`FDM_CENTERED` alone yields `divergence_centered`, with the override it yields `divergence_upwind`).
The guide is also corrected on what that enum does: **`FDM_CENTERED` selects the FP half only** —
`HJBFDMSolver.advection_scheme` measures `gradient_upwind` under `FDM_UPWIND`, under
`FDM_CENTERED`, and under `FDM_CENTERED` with the override alike (#1866). So the "2nd order" in the
scheme table is the FP discretisation, which is the half that refuses to run, and overriding it
leaves the plain `FDM_UPWIND` pair. The footnote and the FAQ entry now say that instead of
recommending the override as a route to HJB order.
The "Before" block keeps `gradient_centered`, since it depicts legacy code, with an inline marker for
readers who copy from the fence without reading the note. The scheme table and the FAQ entry that
recommend `FDM_CENTERED` now carry the positivity caveat.

Nothing in any gate runs either file: `pytest.ini` is `testpaths = tests`, every workflow scopes
`pytest` to `tests/`, and `scripts/local_ci.sh` scopes `ruff` to `mfgarchon/` and `tests/`.

On the iteration bump, stated precisely because the first reading of it was wrong: the old
configuration converges at **27** when given a cap of 60 — `converged=True`, `err_M = 6.43e-09`, and
98% of the mass gone, simultaneously — so the scheme change accounts for 27 -> 49, not 20 -> 49.
Losing the mass made the problem easier. (At the shipped cap of 20 neither converged.) 49 is a dip
out of a plateau rather than a settled fixed point, and neighbouring grids are not robust — `Nx=41`
does not converge in 200 iterations — so 60 is right for this configuration and is not a general
margin.
