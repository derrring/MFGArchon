`EXTRAPOLATION_LINEAR` and `EXTRAPOLATION_QUADRATIC` now reach their ghost formulas through
`PreallocatedGhostBuffer`. Neither of its two dispatch chains had a branch for either member,
and they failed differently: the mixed chain fell to `else: # Fallback for unknown BC types:
use reflection`, and the uniform chain had no terminal `else` at all, so the ghost cells kept
the buffer's zero-initialised contents.

`fp_semi_lagrangian` builds an `EXTRAPOLATION_QUADRATIC` BC every timestep and reached the
first. Measured on `U = 0.5x^2`, where the true Laplacian is 1 everywhere, the wall row came
back **−19** — a 2000% error, now 6.2e−15.

Only the high wall showed it: the parabola is symmetric about `x = 0`, so the reflection ghost
and the quadratic ghost coincide at the low wall. A symmetric fixture cannot see this, which is
why it survived.

The formulas were already written and directly tested; nothing reached them from here. Twelve of
sixteen probed cells are byte-identical — every other `BCType` on both chains — and are pinned as
unchanged. A grid too small for the one-sided stencil now raises rather than silently dropping to
a lower order.
