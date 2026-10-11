- **`BoundaryConditions.get_bc_value_at_boundary` and `BCSegment.get_value`, called with no time, refuse a value
  that declares `t`** (#2512, row B2). Their time defaulted to 0.0. So a caller that passed none read `g(t)` or
  `g(t, x)` at t = 0, silently. FP-FDM did this at every level (#2535).
  - Both default to `None` now.
  - With no time, a value that declares `t` raises `NotImplementedError`. The message says to pass `time=` (or `t=`
    for `get_value`).
  - A constant, or a `g(x)` value, needs no time and reads as before.
