- **HJB-FEM P1 binds `Model(coupling_cost=...)` on its own discretization** (#1642, stage 1b-2). A problem on a mesh
  can now carry a coupling cost. Before, the problem refused it when it was built.
  - The measure is `LumpedMassPointMeasure`, in `mfgarchon.alg.numerical.fem.discretization`. Each dof, which for P1
    is a mesh vertex, has its consistent-mass row sum, the integral of its basis function, as its weight. The domain's
    measure is the sum of the element volumes. `HJBFEMSolver(order=1)` binds the cost on this measure through
    `bind_on_measure` when it is constructed.
  - `BaseHJBSolver.coupling_binding()` returns that binding. For any other solver it is the problem's own binding on
    its grid. The couplers built on `BaseCouplingIterator`, and `MultiPopulationIterator`, hand it to
    `compose_hjb_source(..., binding=)`. On a mesh, a coupling cost solves through `problem.solve()`,
    `FixedPointIterator`, the `BlockIterator` family, and `MultiPopulationIterator` with one population.
  - With `PotentialCost(V)` the solve gives the same U as V in the Hamiltonian, to 2.2e-16. `InteractionCost`
    converges at second order to the exact integral on a polygon.
- **Behaviour change: a coupling cost on a mesh or an implicit domain now constructs.** This covers a
  `Hyperrectangle` and a `PointCloudGeometry`.
  - There `problem.coupling_energy` is None, and the declared cost is `problem.declared_coupling_cost`.
  - At solve time, a reader that cannot bind the cost refuses it by name:
    - HJB-GFDM through a coupler. Its binding at its collocation points is #2761.
    - `NewtonMFGSolver`'s `MFGResidual`, `RegimeSwitchingIterator` and `GraphMFGSolver`.
    - A solver's standalone `solve()`.
    - `compose_hjb_source` called without a binding.
  - The meshless Galerkin solver refuses the cost at construction.
  - On a network the problem still refuses a cost when it is built.
- **Refused by name:**
  - `HJBFEMSolver(order=2)` with a coupling cost. In 2-D its vertex row sums are 0, and in 3-D they are negative.
  - A user's cost without `bind_on`.
  - `InteractionCost(use_fft=True)`.
- HJB-FEM already refused a PERIODIC condition at construction (#1456), so this measure never integrates over a
  periodic axis. `PointMeasure` now states that a measure on a periodic domain is its implementer's responsibility.
- The advice in the standalone `solve()` refusal now passes `binding=solver.coupling_binding()`.
