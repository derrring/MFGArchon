- **Removed `except Exception: pytest.skip` masks around solver calls; added a fast-tier coupled
  mass-conservation gate** (Issue #1567). Ten sites wrapped `solver.solve()` / `solve_hjb_system()`
  in `try/except -> pytest.skip`, converting a solver raise into a green skip (the testing analogue
  of a fail-silent fallback). Un-masked all ten (`test_mass_conservation_1d.py` x6,
  `test_hjb_gfdm_solver.py` x2, `test_collocation_gfdm_hjb.py` x1) so a solver raise now fails the
  test. The one live-skipping site drove the GFDM solve through an incomplete hand-rolled mock
  (missing `T` / `dimension`); rebuilt it on a real `MFGProblem`. Added
  `test_coupled_fdm_mass_conservation_fast_tier`: a small deterministic HJB-FDM + FP-FDM coupled
  solve (~1s, non-`@slow`) that pins total-mass conservation on every PR — previously every coupled
  mass test was `@slow`, so the paper-critical invariant was checked only on the nightly tier.
