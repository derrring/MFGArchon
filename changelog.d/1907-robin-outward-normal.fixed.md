**Robin's `du/dn` is the outward normal derivative at both walls.** The low wall applied the
outward sign a second time, imposing `alpha*u + beta*du/dx = g` — the axis condition, a
physically different wall. Measured on `main`, `alpha=1, beta=0.3, g=0.7`: the residual of the
condition `BCSegment.beta` declares was **6.17 at the min wall** and 0 at the max.

On a cell-centred grid the ghost lies outside at *both* walls, so the quotient toward it,
`(u_g - u_i)/dx`, already is the outward normal derivative; no further factor is due.
`ghost_cell_robin` owns that derivation and the three sites that restated it now call it:
`PreallocatedGhostBuffer._apply_linear_reflection`, `._apply_ghost_for_face`, and
`BaseStructuredApplicator._compute_ghost_robin`.

**User-visible consequence.** `uniform_bc(ROBIN, alpha, beta, value)` reads as "the same
condition on every wall" and was not: the two walls carried different physics whenever
`beta != 0`. `robin_bc(g, alpha=0, beta=1)` and `neumann_bc(g)` are the same condition and
disagreed at the low wall by exactly `2*dx*g`.

**What changed and what did not.** Of thirteen probed cells, **nine are byte-identical** — every
`BCType` other than `ROBIN`, plus `ROBIN` with `beta = 0`, where the derivative term vanishes and
the two conventions coincide. The four that moved are all `ROBIN` with `beta != 0` at the low
wall, and each now satisfies the declared condition to `3e-14` or better across six coefficient
sets including negative `beta` and negative `g`.

**Second behaviour change, from the deletion.** Two of the removed implementations silently
mirrored the interior cell when `alpha/2 + beta/dx == 0`. A singular coefficient means the
condition does not determine the ghost, so the surviving owner raises instead of inventing an
answer.

The pin is the **condition itself**, not a comparison between paths: once all sites route through
one owner, path-A-vs-path-B is tautological and would pass over a broken owner. Two mutations
were measured to redden it — restoring the axis convention in the owner (15 of 20 failures) and
restoring the silent mirror (1).

Two existing tests are re-pointed rather than deleted. `test_ghost_spacing_1904.py` had said so
itself: *"the sign convention this asserts is the applicator's own … which #1907 reports is one
factor too many there … it will need re-pointing when #1907 lands."* And
`test_fp_particle_gradient_bc_1255.py` pinned **both** conventions, one per wall, because it
recorded what the code did rather than what the condition says.
