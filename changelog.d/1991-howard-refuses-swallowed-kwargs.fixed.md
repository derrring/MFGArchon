- **`HJBHowardSolver.solve_hjb_system` refuses `source_term` and `volatility_field` instead of
  discarding them** (Issue #1991). Its signature ended in `**_unused`, so both arguments, and any
  misspelt keyword, left the value function bitwise unchanged with no diagnostic. It now names both
  and raises `NotImplementedError` pointing at their owners: the constructor's `volatility_field`,
  and `HJBGFDMSolver(..., inner_solver="howard").solve_hjb_system(..., source_term=...)` for a
  source. Unknown keywords raise
  `TypeError`. The `source_term` channel census now selects solvers by having a `solve_*_system`
  method rather than by subclassing a base, which is what hid this class from it.
