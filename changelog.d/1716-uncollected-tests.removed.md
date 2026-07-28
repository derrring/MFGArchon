- **Files that look like tests and run in no tier** (Issue #1716) — `benchmarks/validation/`
  (6 `test_*.py` plus diagnostics, excluded by `norecursedirs`),
  `benchmarks/test_solver_performance.py` (imports a module deleted before its own last
  edit), `tests/verify_environment.py` (reported a false ❌ for a legitimately removed
  module) and a demo file for the closed #598. The unresolved 2-D FDM mass finding buried
  in that directory is preserved as #1745. Repository drops 4,305 lines.
