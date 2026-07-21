Refuse a mixed per-axis boundary condition in the nD semi-Lagrangian fold instead of silently
applying one axis's operation to all of them.

`get_bc_type_string` returns the first segment's type by contract. `_trace_characteristic_backward`
passed that single string to `apply_boundary_conditions_nd`, which loops over every dimension and
applies the same geometric operation to each -- so a no-flux wall on one axis beside a periodic
axis reflected both, and reordering the segments changed the physics with no diagnostic.

A construction-time guard already existed but was bypassable: the solver re-reads
`get_boundary_conditions()` on every solve, so a BC set or replaced after construction reached the
fold unchecked. The check now sits at all seven sites that read the BC type, and the per-node
`except Exception -> RuntimeError` handler in `_advect_pointwise` lets the refusal through with its
declared type instead of retyping it.

The refusal is pinned behaviourally across the 45-cell dispatch matrix (dimension x characteristic solver x diffusion method), with a source-level invariant test as a backstop.

Per-axis handling is the actual fix and remains open on #1560; until then the library refuses the
configuration rather than solving a different one.
