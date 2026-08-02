- **`HJBSemiLagrangianSolver` refuses an `interpolation_method` it will not honour** (Issues #1809,
  #1664). The argument was stored unvalidated and both interpolators default to linear for anything
  they do not recognise, so the declared method and the honoured one could differ silently in two
  ways. The 1D path is `if method == "cubic": PCHIP else: linear` — it recognises two methods where
  the nD path recognises five — so `nearest` selected genuinely different interpolants by dimension:
  measured on `U = [0, 10, 0, 10, 0]` at `x = 0.30`, **8.0 in 1D against 10.0 in nD**, and a typo
  such as `"cubis"` was accepted at construction and collapsed to linear at both. Separately,
  `RegularGridInterpolator` needs 4 points per axis for `cubic` and 6 for `quintic` (measured, not
  read from the docs) and raises below that — the only condition that reaches the RBF fallback,
  whose chain then substituted `RBF -> nearest neighbour` behind a `logger.debug` for the method the
  caller asked for. Construction now refuses both, naming the honoured set at that dimension or the
  axis and the minimum. The honoured sets and per-axis minima are single-sourced in
  `hjb_sl_interpolation.py`, since the gap between accepted and honoured is what both issues are
  about. No configuration in the tree is affected: 275 existing semi-Lagrangian tests pass unchanged.
