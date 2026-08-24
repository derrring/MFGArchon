**BREAKING for anyone importing it from `mfgarchon.geometry.boundary.calculators`.**
`calculator_to_constraint` is deleted, with the two tests that were its only callers (#2078, #2070).

It promised *"mathematical equivalence as required by GKS stability"* with the ghost-cell path and
delivered it for **one of three** BC tiers. Measured on `u = 3x + 0.4`, `dx = 0.1`:

| tier | ghost path | this function |
|---|---|---|
| Dirichlet | +1.500000 | **+1.000000** — returns `g`, the *vertex-centred* form, unconditionally |
| Neumann | +0.800000 | +0.800000 ✓ |
| Robin, min | +0.250000 | **+3.750000** |
| Robin, max | +3.550000 | **+2.370000** |

The Robin tier was a three-way disagreement — **code ≠ comment ≠ live path**, none of them exact.
Its comment carried `beta/(2*dx)`, the factor #1350 removed from the live path; its own algebra was a
mis-solve of that stale relation; and the code then applied a different wrong bias. It also
multiplied `beta` by an `outward_sign`, which #2063 deleted from `ghost_cell_robin` after measuring
that passing the physically correct sign is what breaks it.

**Deleted rather than fixed or retired**, on the same criteria that governed `_compute_ghost_*` in
#2057: zero production callers (control — `ghost_cell_neumann` returns 1 under the same query), not
surfaced in the package namespace (`hasattr(mfgarchon.geometry.boundary, ...)` is False), and wrong.
Fixing it would maintain correctness for a consumer that does not exist; delegating it to the three
owners would leave a function nobody calls.

Its only tests were **characterization tests**: their comments restate the body's arithmetic line by
line (`denom = -1 + 0.2 = -0.8`, `weight = ... = 1.5`) and assert those numbers, so they could not
fail however wrong the physics was. That is the same pattern #2067 found on the `_compat` Neumann
side and #2064 found in the vertex Robin arm.

**Before deleting, its prose was checked for claims about other code** — the failure #2057 committed,
where removing an orphaned method also removed the repository's only record that `beta/(2*dx)` was
stale. Here the stale factor is already carried by #2078 with measurements, so nothing was the sole
carrier.

Net: **−155 lines.**
