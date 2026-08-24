**BREAKING for `ghost_cell_robin(..., beta=0, grid_type=VERTEX_CENTERED)`.** The vertex-centred Robin
ghost is now one formula for every `alpha`, and `beta = 0` raises instead of falling into a branch
that had no formula for it (#2064).

The wall **is** the interior node on a vertex layout, so `u_b = u_interior` and
`du/dn = (u_ghost − u_interior)/dx`. Substituting into `alpha*u_b + beta*du/dn = g`:

```
u_ghost = u_interior + dx*(g − alpha*u_interior)/beta
```

Verified exact on 16 combinations — 2 walls × 2 (slope, offset) × `{(0,1), (2,1), (2,0.5), (−1.5,2)}` —
with zero failures.

**The two-branch structure was the defect, not just its arithmetic.** It split on
`abs(alpha) > 1e-12` purely to avoid dividing by `beta`, and the `alpha != 0` arm solved for a
quantity multiplied by `alpha` rather than for the ghost — returning **−10.5 where 3.3 is exact**, at
**both** walls, independently of any sign. Measured with an offset, where a wrong formula and a
mishandled sign separate: **−21.5 where 4.7 is exact**.

The split also hid a fact #2063 had to establish separately: **no sign term belongs here at all**.
`du/dn` already carries the wall's direction, which is why `outward_normal_sign` could be removed —
the unified formula simply has nowhere to put one.

**`beta = 0` IS computable, and an earlier revision of this change refused it.** `alpha*u + 0*du/dn
= g` is the Dirichlet condition `u = g/alpha` — determined, not degenerate. The two-arm code this
replaces already returned exactly that (measured: `alpha=2, beta=0, g=6` → `3.0`), and
`enforcement.py`'s `enforce_robin_value_nd` computes the same `rhs/alpha` rather than refusing.
Raising would have converted a correct answer into an error and put this owner at odds with that one.

The threshold was dimensionally wrong as well. `|beta| < 1e-12` is not scale-free: the same physical
condition scaled by `1e-13` was refused while its exact answer was computable, and `alpha=1e6,
beta=1e-11` passed and returned a result **182% wrong**. Concretely, an FP wall sets `beta = -D`, so
`sigma = 1e-6` → `D = 5e-13` would have been refused and told to become a Dirichlet wall at
`g/alpha = 0` — an **absorbing** wall, the opposite condition to the mass-conserving one requested.

Only `alpha = beta = 0` constrains nothing, and that is what now raises — with a message that says
so, rather than sending the reader to compute `0/0`.

**Corrected in the same review:** the count in an earlier draft said "16 combinations"; `pytest
--collect-only` gives **8** — `(slope, offset)` is fixed, so that axis does not exist. And "the split
existed purely to avoid dividing by `beta`" is inverted: the guard tested **alpha** and protected the
division by **alpha**; dividing by `beta` is what the unified form introduced.

The new tests use `u = a*x + b` with the rhs constructed **from** the field — an external oracle, not
either formula restated — and include `alpha = 0` as the continuity check that the unified form
degenerates to the Neumann ghost, which is what the old structure special-cased.
