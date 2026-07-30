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
  Second batch: `test_fem_degenerate_assembly_f7.py` now measures that a collinear element leaves
  the STIFFNESS matrix non-finite while the mass matrix stays clean, and that `skfem.asm` emits no
  warning at all — the RuntimeWarnings fire earlier, when the basis builds its inverse affine map,
  so a caller handed a basis sees nothing. `test_s4_ic_dof_mismatch.py` now measures that
  zero-padding a P2 initial density does not merely mis-place it: the integral goes from 0.104696
  to exactly 0.000000, because a P2 field carries all its mass in the edge DOFs that padding
  zeroes. That is the same shape-function identity behind the P2 lumping guard, so two independent
  guards rest on one fact.
