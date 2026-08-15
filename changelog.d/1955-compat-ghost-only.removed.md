`geometry.boundary._compat` keeps only its ghost-value entry point. The array-padding
functions `apply_boundary_conditions_1d/2d/3d/nd` and `create_boundary_mask_2d`, with their
twelve private helpers, are deleted -- 1453 of the module's 1856 lines. Their public export
had already been withdrawn (#577 Phase 3); no library module imported them, and the only
remaining importers were two test files reaching into the private module by name.

`get_ghost_values_nd` is unchanged and stays until its declared v0.25.0 removal. It is the
one name still reachable as an attribute of `mfgarchon.geometry.boundary`, so it keeps the
deprecation contract the module published. Its output is pinned byte-identical across this
change over five cases (uniform Dirichlet / Neumann / periodic, a mixed BC, and 1D).

This removes **six of the package's nineteen `BCType` dispatch chains** (19 -> 13). Fourteen of
the nineteen end in a silent `else` or no `else` at all, so an unhandled BC type produces a
wrong answer rather than an error; five do raise, three of them in this module. The six deleted
here are the largest block that could go without a design decision, not a block of uniformly
silent ones.

**Coverage this removes and does not replace:** no test now asserts that
`BCType.EXTRAPOLATION_LINEAR` or `EXTRAPOLATION_QUADRATIC` routes to its ghost formula. The
formulas themselves stay pinned (`ghost_cell_linear_extrapolation`,
`ghost_cell_quadratic_extrapolation` are tested directly), but the only dispatch that reaches
them is now `bc_to_topology_calculator`, which has no test at all.

The user guide's ghost-cell examples move to `pad_array_with_ghosts`. Two of its "Common
Issues" entries are deleted rather than translated: both documented errors of the removed
`apply_boundary_conditions_2d`, and neither reproduces on the canonical path -- measured, an
unbound BC and a mixed BC without `domain_bounds` both pad without raising, element-wise
identically to the bound and bounded forms.
