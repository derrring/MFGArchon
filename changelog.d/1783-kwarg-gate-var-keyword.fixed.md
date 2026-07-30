- **The coupling kwarg gate no longer drops `volatility_field` when a solver declares `**kwargs`**
  (#1783). `_build_hjb_kwargs` / `_build_fp_kwargs` asked `inspect.signature` whether the solver
  named the parameter — which answers "does this callable name it", not "can this solver consume
  it". `MeshlessGalerkinHJBSolver` and `HJBHowardSolver` both declare `(self, *args, **kwargs)`, so
  the field was dropped on the HJB side while the paired FP solver consumed it: measured with
  `problem.sigma = 0.3` and a field of mean 0.7, HJB ran at `D = 0.045` against FP's `D = 0.245`, a
  5.4x mismatch with no warning and a converged density for a problem nobody posed. The gate now
  refuses, matching the `source_term` branch three lines below it (#1424). Both sides route through
  one `_require_kwarg`, so a fix to one is no longer a fix to one. A scalar equal to `problem.sigma`
  is exempt — `MFGProblem.volatility_field` defaults to it, so every ordinary solve carries a
  non-None value that cannot cause a mismatch.
