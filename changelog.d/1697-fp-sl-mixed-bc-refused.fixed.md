Refuse a mixed per-axis boundary condition in the FP semi-Lagrangian adjoint fold, and give the
collapse a single owner shared with the HJB side.

`BoundaryConditions.type` deliberately raises `ValueError` for a mixed BC. `bc_utils` swallowed
that raise and fell through to `segments[0].bc_type`, so the one signal that a BC is mixed was
discarded and the fold applied that single operation to every axis. Reordering the segment list
changed the density by 145% in relative L1 while both runs conserved mass to 4e-15.

Segment order is not the only lever: `get_bc_type_string` never reads `default_bc`, so a
partially-covering segment list plus a differing default collapses identically **with no
permutation available**. A guard unioning only over `segments` lets that form through. The new
`bc_utils.geometric_operations` unions both, and `bc_utils.checked_bc_type_string` is now the one
owner of the refusal for HJB-SL (#1560) and FP-SL (#1697) alike -- the private helper added to
`hjb_semi_lagrangian.py` in #1696 is now a thin wrapper that only binds the consumer name.

Per-axis handling is the actual fix and remains open. It is deliberately not attempted here:
`splat_linear_nd` is periodic-blind (it clamps corner indices), so routing a periodic axis through
it would move traffic onto a known-wrong path that the current uniform configuration avoids.
