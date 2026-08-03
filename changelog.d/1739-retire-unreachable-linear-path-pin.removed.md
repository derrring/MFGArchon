- **`TestTheLinearPathIsUNCHANGEDByTheConsolidation` (2 tests), whose subject #1739 made
  unreachable**. The class pinned the batch linear path's out-of-bounds policy (extrapolate, not
  clamp) against values captured before the interpolation-backend consolidation. It reached that
  policy through a periodic BC, and only because the periodic fold was dead: feet left the domain
  at every step, so the choice of interpolant decided the answer. With the fold repaired, the
  solver's supported set `{NO_FLUX, NEUMANN, PERIODIC}` maps to `{reflect, periodic}` and nothing
  reaches that interpolant out of bounds. Measured: swapping the extrapolating `interp1d` for a
  clamping `np.interp` -- the mutation the class exists to catch -- moves the fixture sum by
  `2e-16` (1 ULP) where it moved it by `1.9e-3` before. Re-capturing the constants would have kept
  a test that pins numbers and can no longer fail for its stated reason. Its own positive control,
  `test_the_fixture_actually_sends_feet_out_of_the_domain`, is what reported this: it counts feet
  rather than asserting on a dispatch string, and its docstring had already constructed this fix
  and measured the count going 8/328 to 0/328. The cubic half of the consolidation invariant is
  unaffected and stays pinned by `test_a_solve_survives_a_non_representable_domain[cubic-*]`.
