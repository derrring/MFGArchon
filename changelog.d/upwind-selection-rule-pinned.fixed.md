- **`gradient_upwind`'s selection rule is now pinned by an identity rather than a tolerance.** The
  only test naming the rule — `test_monotone_increasing`, whose docstring says "upwind should select
  backward difference" — asserted `max|du - 2x| < 0.1`. Forward and backward differences are both
  $O(h)$ on smooth data, and on $u = x^2$ their errors are *equal in magnitude* (0.0100 each at
  $n=100$, symmetric about the exact value), so no tolerance separates them at any $n$. Inverting
  the selection left it green.

  What separates them is the value: at $x = 0.5$, backward $= 0.990000$ and forward $= 1.010000$,
  differing by exactly $h\,u'' = 0.02$. The new test compares against the one-sided stencil
  functions directly, on the interior (these use `np.roll`, so node 0's backward difference wraps
  and reads a spurious $-98.01$ on this fixture).

  Both branches are covered now. Only the increasing one had a test, so the forward branch — taken
  whenever information travels left, which is half of every advection problem — was exercised by
  nothing in `tests/unit/test_operators`. Found by a mutation sweep: inverting the rule left all 348
  tests in that directory green, and was caught only downstream, by
  `tests/integration/test_hjb_fdm_2d_validation.py::test_2d_solve_fixed_point`.
