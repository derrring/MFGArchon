- **HJB-FEM P1 binds `Model(coupling_cost=...)` on its own discretization** (#1642, stage 1b-2). A problem on a
  triangle or tetrahedron mesh, or on a generated `Mesh1D`, can now carry a coupling cost. Before, the problem refused
  it when it was built.
  - The measure is `LumpedMassPointMeasure`, in `mfgarchon.alg.numerical.fem.discretization`. Each dof, which for P1
    is a mesh vertex, has its consistent-mass row sum, the integral of its basis function, as its weight. The domain's
    measure is the sum of the element volumes. `HJBFEMSolver(order=1)` binds the cost on this measure through
    `bind_on_measure` when it is constructed.
  - `BaseHJBSolver.coupling_binding()` returns that binding, and None for a solver that binds none of its own. The
    couplers built on `BaseCouplingIterator`, and `MultiPopulationIterator`, hand it to
    `compose_hjb_source(..., binding=)`. On a grid the solved problem's own binding is read whatever the solver; off a
    grid the solver's is read only when it was bound from the solved problem's own cost object, and a solver bound
    from another cost, an equal one built again or a copy included, is refused by name. On a mesh, a coupling cost solves through `problem.solve()`,
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
  - `HJBFEMSolver(order=2)` with a coupling cost. On triangles its vertex row sums are 0, and on tetrahedra they are
    negative.
  - HJB-FEM with a coupling cost on a mesh whose `MeshData` has no element volumes to read the domain's measure
    from: quad and hexahedron meshes, unless their `element_volumes` are set.
  - A user's cost without `bind_on`.
  - `InteractionCost(use_fft=True)`.
- HJB-FEM already refused a PERIODIC condition at construction (#1456), so this measure never integrates over a
  periodic axis. `PointMeasure` now states that a measure on a periodic domain is its implementer's responsibility.
- The advice in the standalone `solve()` refusal now passes `binding=solver.coupling_binding()`.
