- **An n-D problem built through `geometry=` started with the wrong amount of mass** —
  `(n/(n-1))^d` instead of 1. `MFGProblem` normalises `m_initial` by dividing by its discrete
  integral, and it computed that integral's cell volume itself, as `prod((b - a) / n)` from
  `self.spatial_discretization`. **That attribute does not mean one thing.** Passed to the
  constructor it is the *interval* count — `mfg_problem.py:834` builds `Nx_points = [n + 1 for n in
  spatial_discretization]` — and `(b - a) / n` is then exactly the spacing. Taken from a geometry,
  `tensor_grid.py:474` puts the *node* count there instead, and the same expression is one interval
  too wide. Measured on the same 11x11 grid with the same `[0.1, 0.1]` spacing both ways:
  `spatial_discretization=[10, 10]` gave mass `1.0`, `Nx_points=[11, 11]` gave `1.21`; in 3-D at
  `n=9`, `1.0` against `1.4238`.
  The normaliser now asks the geometry for the spacing and raises if it cannot supply one, rather
  than inventing the number. After the fix both paths give `1.000000` at 2-D `n=11/15/21` and 3-D
  `n=9`, with 1-D untouched throughout — it takes a different branch and was always right, which is
  what makes it the control.
  **The dual meaning is the real defect and this does not close it.** The fix stops reading the
  attribute, which makes this one site immune; every other reader of `spatial_discretization`,
  including user code, still has to guess which count it holds. Filed as #1889.
  **Consequences while it was live**: on the `geometry=` path an 11-point 2-D problem carried 21%
  more mass than its source says and a 9-point 3-D one 42%, so `coupling=lambda m: m` applied a
  correspondingly stronger interaction. The offset shrinks like `d/n`, so under refinement it reads
  as a first-order-convergent discretisation error rather than as a bug. It does **not** explain
  #1865: the 2-D smoke fixture still fails to converge with the mass corrected, at a peak `m(0,.)`
  of 9.550 rather than 11.555.
  *An earlier version of this entry said "every n-D problem". That was false — the
  `spatial_bounds=` path was already exact, and 29 multi-dimensional sites across `tests/`,
  `examples/`, `benchmarks/` and `scripts/` use it.*
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
