- **`ConvergenceInfo.plot_convergence()`'s `except ImportError` covers the import statement only**
  (Issue #2090, items 2 and 3; item 1 shipped in #2091, item 4 belongs to #2089). It wrapped the
  whole method body, so an `ImportError` raised from *inside* matplotlib — `plt.figure()` does that
  on a backend whose own dependency is absent — was reported as "Matplotlib not available for
  plotting", which names the one package that is installed.

  The test for that fallback had never entered it. matplotlib is a hard runtime dependency, so the
  `try` always succeeded, and the assertion was `isinstance(plot_succeeded, bool)` over a variable
  assigned only literal `True` and `False` — it could not fail in any state of the code. Three tests
  replace it, each verified against a mutation that kills it and nothing else.

  The fixture changed with them, which closes the issue's first defect: `[1e-1, 1e-3, 1e-5]` is
  three exact powers of ten, so `semilogy` could render nothing but a straight line — the
  "convergence history" that made no sense — and a fixture of round constants also cannot separate
  `self.residual_history` from the same numbers written inline.
