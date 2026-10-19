- **A coupling cost is evaluated at an HJB solver's own points** (#1642, stage 1b-1). HJB-GFDM on a grid problem
  evaluates its source at its collocation points. The coupling cost's part of that source was refused there, so
  `Model(coupling_cost=...)` could not be solved with HJB-GFDM unless the collocation points were the grid's nodes in
  their order.
  - Off the grid's nodes, the part is the bound energy's `flat_derivative_at(m, x, *, t)`. That method is the
    protocol `PointwiseFlatDerivative`, in `mfgarchon.operators.interaction.energy_functionals`.
  - `PotentialCost(V)` evaluates V(x). `InteractionCost(K)` evaluates the grid's quadrature at x,
    Σ_j K(|x − y_j|) w_j m_j, densely. A sum evaluates each of its parts.
  - `PotentialEnergy(..., V=)` and `QuadraticInteractionEnergy(..., nodes=)` take what that evaluation needs, and
    `ConvolutionCouplingOperator.kernel` exposes the kernel.
  - **Refused by name**, at points other than the nodes: a user's energy without `flat_derivative_at`; a sum with
    such a part; and an energy built without V or without its nodes. The message names the solvers that evaluate at
    the grid's nodes.
  - **A weak-form HJB solver refuses a problem that declares `coupling_cost` or `terminal_coupling`**, by name, at
    construction. Such a solver has no binding of its own yet (stage 1b-2). On a grid problem, meshless Galerkin on its
    own cloud read the grid-bound energy at its dofs by index, a coefficient taken as a point value (#2672). It ran
    silently with either energy on a cloud of the grid's nodes, and with a terminal coupling on any cloud of the
    grid's size.
