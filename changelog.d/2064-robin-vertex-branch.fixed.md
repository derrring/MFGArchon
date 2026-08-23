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

**`beta = 0` is not a Robin condition.** `alpha*u + 0*du/dn = g` is the Dirichlet condition
`u = g/alpha`, and the ghost that imposes it is
`ghost_cell_dirichlet(interior_value, g/alpha, VERTEX_CENTERED)` — **`g/alpha`, not `g`**. The
refusal carries that division, because a reader who follows it to `ghost_cell_dirichlet(g)` imposes a
different boundary value. Only one literal `beta=0` call exists in the tree, in a test; production
usage is zero.

The new tests use `u = a*x + b` with the rhs constructed **from** the field — an external oracle, not
either formula restated — and include `alpha = 0` as the continuity check that the unified form
degenerates to the Neumann ghost, which is what the old structure special-cased.
