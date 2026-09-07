One owner for the #1560/#1697 per-axis collapse guard. `HJBSemiLagrangianSolver.__init__` carried an
inline copy of the predicate and the two had already diverged: the copy read
`getattr(bc, "default_bc", None)`, so on the #1691 rename signature -- `segments` present,
`default_bc` renamed away -- it treated the absence as "no default" and constructed, while
`geometric_operations` refuses to guess and raises. A per-axis disagreement carried by the renamed
field was invisible to construction and surfaced only at solve time, from another module.

The guard is now `bc_utils.refuse_mixed_per_axis`, split out of `checked_bc_type_string`, which is
that guard plus `get_bc_type_string`. The two are separate responsibilities: a constructor wants the
refusal and has no use for the collapsed value, and `get_bc_type_string` raises `ValueError` on a
segment-free BC that the guard itself accepts.
