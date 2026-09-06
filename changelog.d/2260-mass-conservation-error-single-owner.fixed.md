- **`compute_mass_conservation_error` measures the grid measure, and there is one owner of it**
  (Issue #2260). The exported function computed `sum(M, axis=spatial) * prod(spacing)` — the
  rectangle rule — and published a boolean verdict, `is_conservative`, from it. #2145 established
  that the rectangle rule is not the mass on this library's endpoint-inclusive grid: the two
  boundary nodes each own half a cell, not a full one. Before #2258 the SL/CN family conserved
  `sum(m)` by construction, so the bug agreed with the family it measured by accident; #2258 moved
  that family onto the grid measure, and from that merge the disagreement became **every SL solve**.
  Measured directly: `FPSLSolver`, driven transport (the fixture in
  `tests/unit/test_alg/test_sl_cn_wall_2243.py`, potential `u = -0.5x`, nx=41, nt=200), grid-measure
  drift 8.549e-15 — reported by the pre-fix formula on the identical output as 9.2685% drift and
  `is_conservative=False`.

  The function now builds its weights the same way `FluxDiagnostics._axis_weights` already did —
  via `quadrature_weights_nd`, not a second derivation of the trapezoid formula. The returned dict
  gains a `"measure"` key naming what the verdict is about, per the issue's own closing question:
  *"`is_conservative` should say which measure it is a verdict about, or stop being a verdict."*

  **One concept, three implementations, found while fixing this.** `base_variational.py` had a
  second `compute_mass_conservation_error` (correct quadrature, `trapezoid`; zero callers anywhere
  in the package — deleted) and `sinkhorn_solver.py` had a third, private one (correct quadrature,
  but a different statistic — `std` across the trajectory rather than max deviation from the
  initial slice; kept, since a Sinkhorn density need not start at its own converged mass the way a
  time-marched solve does, but now delegates its quadrature to the same owner instead of
  re-deriving `np.trapezoid` inline). Verified numerically equivalent to the pre-refactor output to
  8e-17 before landing, since this half of the change is a pure refactor and must not move the
  number Sinkhorn reports.

  `implicit_diffusion.py`'s module docstring carried a caution about this exact mismatch since
  #2237, describing it as still live; marked `[FIXED 2026-09-06, #2260]` rather than left to read as
  current.
