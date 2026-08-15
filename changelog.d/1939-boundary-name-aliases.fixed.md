Boundary-name aliases resolved on the FDM ghost path and not on the query path, so one
`BoundaryConditions` object answered differently depending on which route asked. A Dirichlet exit
declared as `"left"` — the alias `BCSegment`'s own docstring lists first — was honoured by
`pad_array_with_ghosts` and ignored by `get_bc_type_at_boundary`, `get_bc_value_at_boundary` and
`BCSegment.matches_point`, which compared the raw strings with `==`. All three now route through
`parse_boundary_face`, the owner, and compare faces.

Two defects in that owner are fixed with them. It now parses `dim{N}_{side}`, which
`applicator_particle._get_boundary_id` emits for `d >= 3` and which previously resolved to **axis 0** —
so a condition declared on the fourth axis was applied to the first while its own axis took the
default. And an unrecognised prefix such as `"inlet_min"` now returns `None` instead of
`BoundaryFace(0, side)`: a naming mismatch falls through to the caller's declared default rather than
silently governing a boundary nobody named.
