The semi-Lagrangian pair now takes U's gradient at a wall node from U itself rather than from the finite-difference ghost (#2467). This changes `HJBSemiLagrangianSolver`'s control and `FPSLSolver`'s drift.

- Every axis that is not periodic takes the second-order one-sided difference of U at its walls.
- A 2-point axis takes its single difference, (u_1 - u_0)/h.
- Periodic axes and interior nodes are unchanged.

The cell-centred no-flux ghost had given the wall half the one-sided slope. On a sigma = 0 problem with a Hopf-Lax closed form, the wall error was 1.6x the interior error at 101 points, and is now 1.01x. In 2-D the corner went from 1.71x to 1.06x. Wall mass translated from the wall had lagged by 3.7e-03 at 51 points, and now lags by 5.4e-04.

**Changed behaviour.**

- **Two BC shapes now refused.** Periodicity is read per face, and two shapes cannot be read that way:
  - a geometry that mixes periodic and non-periodic faces with a segment that has no `boundary`, because the two face resolvers can disagree about which faces such a segment covers;
  - an axis periodic on one face only.

  A segment named by its face and restricted to part of it is accepted. So is a uniform BC, whatever its unused `default_bc`. The solvers' own refusal of a mixed BC already covered both shapes, except where `FPSLSolver` is given its own `boundary_conditions`, the path these refusals now close.
- **With `enable_adaptive_substepping=False`, some solves now run away and raise** (#2438's mechanism, #2465). Their characteristics leave through a wall at a foot of one cell or more per step; not every such solve runs away. It was measured at sigma = 0, and on the stochastic path at sigma = 0.1 and 0.3. Before, the half slope let the wall foot decay and the solve returned a finite value. The default, with sub-stepping, is unaffected.

This also removes the semi-Lagrangian regression that blocked the node-centred ghost #1935 needs. That ghost gives the boundary datum at the wall whatever U is, which froze the wall characteristic.
