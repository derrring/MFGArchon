The semi-Lagrangian pair now takes U's gradient at a wall node from U itself rather than from the finite-difference ghost (#2467). This changes `HJBSemiLagrangianSolver`'s control and `FPSLSolver`'s drift. On a non-periodic domain every wall node takes the second-order one-sided difference of U; periodic domains and interior nodes are unchanged.

The cell-centred no-flux ghost had given the wall half the one-sided slope. On a sigma = 0 problem with a Hopf-Lax closed form, the wall error was 1.6x the interior error at 101 points, and is now 1.01x. In 2-D the corner went from 1.71x to 1.06x. Wall mass translated from the wall had lagged by 3.7e-03 at 51 points, and now lags by 5.3e-04.

This also removes the semi-Lagrangian regression that blocked the node-centred ghost #1935 needs. That ghost gives the boundary datum at the wall whatever U is, which froze the wall characteristic.
