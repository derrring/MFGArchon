The semi-Lagrangian pair now honours a different boundary condition on each axis -- no-flux walls in
x beside a periodic y, say -- instead of refusing the BC. `HJBSemiLagrangianSolver` and `FPSLSolver`
read each axis's operation face by face through the new `bc_utils.per_axis_operations`, fold each
foot coordinate by its own axis's operation, hand the ADI sweep one boundary per axis, and identify the
coincident end nodes on periodic axes only.

Still refused, at construction and at every point of use: an axis whose two faces ask for different
operations, such as periodic on one face only; and, in a mix of operations, a segment the face reader
cannot place -- one with no `boundary` (#2467), which every `sdf_region`, `normal_direction` and
`region_name` segment is; one with `boundary="all"`, which the face reader applies to no face (#1953);
one whose `boundary` names no face of the domain (a misspelt name, a Gmsh tag, an axis past the
dimension), which the face reader drops and the default replaces; and one whose `region` restricts it
to part of a face, which the face reader stretches over the whole face (#2490). The `"all"`, `region`
and no-face refusals are wider than they need to be -- they also reject a segment whose operation
matches the faces it lands on, or a misspelt one the default would have matched -- and the first two
wait on #1953 and #2490; `main` refused every mix.

`TensorProductGrid` now binds its periodic convention to a BC whose periodicity comes from
`default_bc`, not only to one with a periodic segment (#1822), and so does the solver-level binder for a
BC handed to a solver; both read one predicate, `bc_utils.declares_periodic`. The grid stores the
completed BC for every solver, so this corrects more than the semi-Lagrangian pair: on a
`segments=[]`, `default_bc=PERIODIC` BC the grid's Laplacian was 9.3e-01 (relative) and an HJB-FDM solve
1.9e-01 away from `periodic_bc` on `main`, and are identical to it now. Left unbound, the SL pair solved
such a BC up to 11% off the same BC with its periodic faces named.

Pinned by separability on a channel with no-flux walls over [0, 1] and a periodic axis over [0, 2]:
a separable problem must equal the sum (HJB) or product (FP) of its one-axis solutions -- from the 1-D
solver on each axis's own interval for ADI and FP-SL, with the periodic axis in y and in x; from the
same solver's uniform 2-D solves for the stochastic step -- which the scheme reproduces to rounding,
2.6e-15 or below on the shipped code. Giving every axis one operation misses by 0.25 (HJB, ADI) and
0.17 (FP). The law pins per-axis handling, not each operation: it compares the scheme with itself on
one axis at a time.

`bc_utils.refuse_mixed_per_axis` and `bc_utils.checked_bc_type_string` are removed: nothing calls
them once the pair reads per axis.

The n-D DPP step (`L1ControlCost`) used to clip its foot on every axis; it now folds it by each
axis's operation, as the 1-D DPP step already did. On a periodic axis that is a correction: the
exact DPP minimum is the wrapped one, and on one step of a periodic fixture the exact minima under
the two folds differ by 0.25. On a no-flux axis, for a running cost that does not decrease as any
|alpha_d| grows -- L1 is one -- the exact minimum does not depend on the fold, so the change there is
the optimiser's. That step misses the exact minimum under either fold, on main as well (#2501).

`diffusion_method="explicit"` still reads no BC on any axis, uniform or not (#2504).
