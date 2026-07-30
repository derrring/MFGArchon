- **Two fail-loud guards now have a test that measures the wrong physics, not just the refusal**
  (#1714). Measured across the campaign's 26 first-parent commits and 86 added test functions:
  only **one** commit contains any numeric assertion, and 20 of the 26 test files still contain
  none — reverse-applying a guard hunk yields `DID NOT RAISE` every time, never a wrong value.
  `test_sl_maximize_fail_loud_1547.py` now asserts the identity the guard's own message cites,
  that MINIMIZE and MAXIMIZE give `alpha* = -grad(u)/lambda` and `+grad(u)/lambda` and are exact
  opposites; `test_weak_form_hjb_p2_gradient_1252.py` now asserts that the P2 vertex shape
  function integrates to ~0 over a triangle while the midside one integrates to 1/6, which is why
  row-sum lumping collapses at vertices and why clamping to 1e-15 produced 1e15-scaled gradients.
  Both hold with or without the guard, so they pin the physics rather than the guard's existence.
