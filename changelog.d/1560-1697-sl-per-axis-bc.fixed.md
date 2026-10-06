The semi-Lagrangian pair now honours a different boundary condition on each axis -- no-flux walls in
x beside a periodic y, say -- instead of refusing the BC. `HJBSemiLagrangianSolver` and `FPSLSolver`
read each axis's operation face by face through the new `bc_utils.per_axis_operations`, fold each
foot coordinate by its own axis's operation, hand the ADI sweep one boundary per axis, and identify the
coincident end nodes on periodic axes only.

Still refused, at construction and at every point of use: an axis whose two faces ask for different
operations, such as periodic on one face only; and, in a mix of operations, a segment with no
`boundary` (#2467) or one restricted to part of a face by `region`, `sdf_region`, `normal_direction`
or `region_name`, which the face reader would stretch over the whole face (#2490).

Pinned by separability on a channel over [0, 1] x [0, 2]: a separable problem must equal the sum (HJB)
or product (FP) of the 1-D solver's solutions on each axis's own interval, which the scheme
reproduces to rounding -- 2.6e-15 or below on the shipped code. Giving every axis one operation
misses by 0.25 (HJB) and 0.17 (FP). The law pins per-axis handling, not each operation: it compares
the scheme with itself on one axis at a time.

`bc_utils.refuse_mixed_per_axis` and `bc_utils.checked_bc_type_string` are removed: nothing calls
them once the pair reads per axis.

The n-D DPP step (`L1ControlCost`) used to clip its foot on every axis; it now folds it by each
axis's operation, as the 1-D DPP step already did. On a periodic axis that is a correction: the
exact DPP minimum is the wrapped one, and on one step of a periodic fixture the exact minima under
the two folds differ by 0.25. On a no-flux axis, for a running cost that does not decrease as any
|alpha_d| grows -- L1 is one -- the exact minimum does not depend on the fold, so the change there is
the optimiser's. That step misses the exact minimum under either fold, on main as well (#2501).

`diffusion_method="explicit"` still reads no BC on any axis, uniform or not (#2504).
