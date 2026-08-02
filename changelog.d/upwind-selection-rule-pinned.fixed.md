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

  Review found the first version's vacuity guard vacuous in the same way it was written to prevent:
  it compared the FULL arrays while the identities it certifies are asserted on `[1:-1]`, so node
  0's wraparound alone satisfied it. On a linear fixture both identities pass vacuously (interior
  forward == backward) and the guard still reported "they differ". It now uses the same slice, on
  both fixtures.

  Review also found three non-equivalent mutations surviving, all in the *predicate* rather than
  the branch bodies: `>=` weakened to `>`, and the predicate reading `grad_forward` or
  `grad_backward` instead of `grad_central`. Strictly monotone fixtures never reach
  `grad_central == 0`, so they pin which stencil each branch returns and not the rule choosing
  between them — and that rule is what makes the scheme Godunov. A quadratic with an interior
  extremum lands a node exactly on the tie. **Both** signs are needed: at a maximum `grad_backward`
  is positive and a `grad_backward` predicate agrees by coincidence; at a minimum it is negative
  and they part. All five mutations now redden.
