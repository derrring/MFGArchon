- **Every n-D problem started with the wrong amount of mass** — `(n/(n-1))^d` instead of 1.
  `MFGProblem` normalises `m_initial` by dividing by its discrete integral, and in n-D it computed
  that integral's cell volume itself, as `prod(L/n)`. That is the spacing of n *intervals*; a grid of
  n *nodes* spanning `[a, b]` inclusive has spacing `L/(n-1)`, which is what
  `geometry.get_grid_spacing()` returns and what every mass measurement in this library uses. The
  normaliser and the measurement were reading different rulers, each self-consistent, so nothing
  could raise. Measured against the closed form on six configurations before the fix — 2-D `n=11`
  `1.210000`, `n=15` `1.147959`, `n=21` `1.102500`, 3-D `n=9` `1.423828`, 1-D exact at 1 because it
  takes a different branch — and all six are `1.000000` after. The normaliser no longer computes a
  spacing at all; it asks the geometry, and raises if the geometry cannot supply one rather than
  inventing the number that caused this.
  **Consequences while it was live**: an 11-point 2-D problem carried 21% more mass than written and
  a 9-point 3-D one 42% more, so `coupling=lambda m: m` was correspondingly stronger than the source
  says. The error shrinks like `d/n`, so under refinement it looks like a first-order-convergent
  discretisation error rather than a bug. It does **not** explain #1865: the 2-D smoke fixture still
  fails to converge with the mass corrected, at a peak `m(0,.)` of 9.550 rather than 11.555.
- **Why 5943 tests did not notice, and the pin that now does.** Mass conservation is measured as
  *drift from the initial mass*, deliberately and correctly — drift is the physical property, and a
  ratio is invariant to the cell measure. That makes the entire mass-oracle family structurally blind
  to the initial value being wrong, which is why the full gate is green both before and after this
  change. `tests/unit/test_core/test_initial_density_mass_1888.py` supplies the missing half: an
  external oracle on the absolute value, integrated with the geometry's own spacing rather than the
  normaliser's. Mutation-verified — reverting the fix turns 5 of its 7 assertions red, and the 2 that
  survive are the 1-D rows, which take the other branch and are supposed to be unaffected.
- **The defect was already recorded, and that is the more useful finding.** The docstring of
  `test_mass_conservation_error_1672.py::test_the_metric_is_drift_not_deviation_from_one` described
  this exact fork on 2026-07-21, with the `(N/(N-1))**d - 1` closed form and the 21% figure, and set
  it out of scope. No issue was filed. The test was then written to be invariant to it, removing the
  last reading that could have surfaced it, and it cost a full re-investigation twenty days later.
  That docstring is corrected here. **"Out of scope" without a filed issue is a deletion with extra
  steps** — filing is free.
