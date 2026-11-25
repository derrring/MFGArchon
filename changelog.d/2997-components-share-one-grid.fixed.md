- **`GraphMFGSolver`, `RegimeSwitchingIterator` and `MultiPopulationIterator` refuse, by name and before the first
  sweep, components that do not share one time grid (T, Nt) and one iterate shape** (#2997 stage 1c, the user's rulings
  of 2026-11-25). Each route reads every other component's level at the step it is solving, by index.
  - On `main`, `RegimeSwitchingIterator` solved regimes of different Nt, or of different T at one Nt, with no error:
    regime k read regime j's level at its own step, `round(t / dt_k)`, so the regimes were coupled at different times.
  - On `main`, `GraphMFGSolver` solved nodes of different grid sizes, its default couplings truncating to the shorter
    level (#2765).
  - On `main`, `GraphMFGSolver` and `RegimeSwitchingIterator` solved components of one dt and different (T, Nt), as
    (0.3, 5) beside (0.6, 10).
  - The comparison is exact, as `MultiPopulationIterator`'s was (#2335), so a caller passes one T to every component.
    `GraphMFGSolver` compared dt with `np.isclose` on `main`, so a horizon equal within rounding, 0.1 + 0.2 beside 0.3,
    solved there, correctly, and is refused now: the refusal names both values.
  - `GraphMFGSolver` refused a different dt at construction, and `MultiPopulationIterator` a different (T, Nt) in its own
    check; both are now the one refusal, at the solve, whose message names the components and both grids.
