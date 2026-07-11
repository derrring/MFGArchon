- **Corrected three single-source pin docstrings that overclaimed their discrimination** (Issue
  #1569). No regression was unguarded (a companion test discriminates in every case), but a lying
  pin docstring erodes the single-source guard layer. Fixed: (1) `test_backend_literal_equals_single_source`
  is an IEEE identity of `0.5*sigma**2`, not a backend-kernel guard — no backend is imported; the
  numpy D-application is pinned behaviorally by the magnitude gate. (2) `test_compute_normal_from_bounds_default_tol_is_onwall_tol`
  pins the default's VALUE only — a signature default is the evaluated value, so a bare-literal revert
  passes it; the import-vs-literal re-fork is caught by the companion source-grep test. (3) The
  magnitude-gate header no longer lists `#1169` (anisotropic off-diagonal) / `#1183` (varying-sigma
  mean-collapse) as caught here — a constant-isotropic eigenmode cannot see them; named the tests that
  actually guard them.
