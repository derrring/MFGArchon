`get_ghost_values_nd` produces a Robin wall when asked for one. Its two ghost helpers took
`(bc_type, bc_value)` and nothing else, so a Robin condition arrived with its coefficients
already discarded and both branches returned the adjacent interior cell — the impermeable-wall
mirror. Measured on three coefficient sets the ghost was `3.10000` every time, and the residual
of the condition the caller wrote ranged over **1.7 to 5.2**; it is now within `7.1e-15`.

The function is deprecated and kept until v0.25.0 (#1955). It is not in the package `__all__`
but is reachable as an attribute of `mfgarchon.geometry.boundary`, which is the surface that
notice applies to — so a user still on the deprecation path was getting a wall that was not the
one they asked for.

Both branches now call `ghost_cell_robin`, which makes #1957's "one owner" literally true. A
Robin condition whose segment carries no coefficients raises instead of taking `BCSegment`'s
`alpha=1.0, beta=0.0` defaults, which are a Dirichlet wall — the #1558 failure mode, and the
same hole #1956 closed on the particle side.

Dirichlet, Neumann and periodic are byte-identical, pinned as such: #1955 pinned this function
across the module's deletion and this change must not disturb that.
