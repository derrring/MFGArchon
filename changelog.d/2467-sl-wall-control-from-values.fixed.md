The semi-Lagrangian pair now takes U's gradient at a wall node from U itself rather than from the finite-difference ghost (#2467). This changes `HJBSemiLagrangianSolver`'s control and `FPSLSolver`'s drift. Every axis that is not periodic takes the second-order one-sided difference of U at its walls; on a 2-point axis this is the single difference (u_1 - u_0)/h. Periodic axes and interior nodes are unchanged.

Periodicity is now decided per axis, through the boundary-face resolver. Before, a mixed geometry read through `FPSLSolver`'s own `boundary_conditions` took the first segment's type for the whole boundary.

The cell-centred no-flux ghost had given the wall half the one-sided slope. On a sigma = 0 problem with a Hopf-Lax closed form, the wall error was 1.6x the interior error at 101 points, and is now 1.01x. In 2-D the corner went from 1.71x to 1.06x. Wall mass translated from the wall had lagged by 3.7e-03 at 51 points, and now lags by 5.4e-04.

**Changed behaviour with `enable_adaptive_substepping=False`.** A solve whose characteristics leave through a wall at foot CFL of 1 or more now runs away and raises (#2438's mechanism, #2465). Before, the half slope let the wall foot decay, and the solve returned a finite value that does not converge under refinement. The default, with sub-stepping, is unaffected.

This also removes the semi-Lagrangian regression that blocked the node-centred ghost #1935 needs. That ghost gives the boundary datum at the wall whatever U is, which froze the wall characteristic.
