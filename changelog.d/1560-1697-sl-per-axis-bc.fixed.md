The semi-Lagrangian pair now honours a different boundary condition on each axis -- no-flux walls in
x beside a periodic y, say -- instead of refusing the BC. `HJBSemiLagrangianSolver` and `FPSLSolver`
read each axis's operation face by face through the new `bc_utils.per_axis_operations`, fold each
foot coordinate by its own axis's operation, hand the ADI sweep one boundary per axis, and identify the
coincident end nodes on periodic axes only. An axis whose two faces ask for different operations --
periodic on one face only -- is still refused, at construction and at every point of use.

Pinned by a separability law: on a channel, a separable problem must equal the sum (HJB) or product
(FP) of the same solver's one-axis solves on uniform BCs, which the scheme reproduces to rounding.
Measured at 1e-15 or below on the shipped code; giving every axis one operation misses by 0.33 (HJB)
and 0.22 (FP).

`bc_utils.refuse_mixed_per_axis` and `bc_utils.checked_bc_type_string` are removed: nothing calls
them once the pair reads per axis.

The n-D DPP step (`L1ControlCost`) used to clip its foot on every axis; it now folds it by each
axis's operation, as the 1-D DPP step already did. On a periodic axis that is a correction: the
exact DPP minimum is the wrapped one, and on one step of a periodic fixture the exact minima under
the two folds differ by 0.25. On a no-flux axis the exact minimum does not depend on the fold, so the
change there is the optimiser's. That step misses the exact minimum under either fold, on main as
well (#2501).
