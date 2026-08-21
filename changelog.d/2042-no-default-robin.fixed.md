**There is no default Robin** (#2042). `_update_ghosts_mixed` synthesized a `__default__` segment
from `bc_type` and `default_value` only, so `alpha`, `beta` and callable values were dropped and
`BCSegment`'s own `alpha=1.0, beta=0.0` took their place — and **`beta = 0` is Dirichlet**. A
defaulted Robin therefore became a *different* boundary condition rather than an approximate one,
bit-for-bit identical to `dirichlet_bc(value=g)` and independent of the α and β actually passed. It
now raises, naming what is missing and what the silent default would have produced;
`BoundaryConditions` carries `default_bc` and `default_value` and no `default_alpha`/`default_beta`,
so there is nothing to infer from. This continues #1100's ruling, where `_resolve_default_bc`
already refuses to guess an unset `default_bc`.

A **uniform** BC reaching that path now forwards its single segment instead of rebuilding a partial
copy — the information is present, and forwarding makes the two ghost paths agree exactly where they
previously diverged (Robin, and a callable Neumann value). This is not the `bc.segments[0]` fallback
the surrounding comment rejects: that one fired on *mixed* BCs, where `[0]` means highest priority.

The discrimination baseline is regenerated in the same change. `bc_uniform_dispatch_reads_as_mixed`
falls **43 → 12**, and the fall is the fix rather than a regression: the 31 tests that stopped
noticing were detecting the divergence between the two ghost paths, which is exactly what the
forward removes. Attributed by A/B on the same mutation and anchor — 43 on `main`, 12 with this
change, product code the only difference — and recorded as a note in the baseline, because the
ratchet reads any fall as a regression and cannot see that distinction. Four other rows rose and no
row went `INEFFECTIVE`.
