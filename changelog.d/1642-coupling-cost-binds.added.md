- **`Model(coupling_cost=...)` is bound and solved** (#1642, stage 1). It was accepted and read by nothing.
  - The value is a specification, from `mfgarchon.operators.interaction`: `InteractionCost(kernel)` for
    F[m] = (1/2) ∫∫ K(x − y) m(x) m(y), `PotentialCost(V)` for F[m] = ∫ V m, or their sum. A part may be an
    instance of any class whose `bind(problem)` returns an `EnergyFunctional`, checked at construction. `a + b` takes
    a user's part on either side; `sum([...])` takes one where its first element is the library's;
    `CombinedCost([...])` takes the parts in any order.
    Each binds to the `EnergyFunctional` it names: `QuadraticInteractionEnergy`, `PotentialEnergy`, `CombinedEnergy`.
  - `MFGProblem` binds it at construction on its grid and measure, as `problem.coupling_energy`, and binds it again
    on a rebuild. The HJB carries its flat derivative δF/δm at each density slice as a cost-signed source, summed with
    the Hamiltonian's own coupling f and with `source_term_hjb`. Every grid coupler, `GraphMFGSolver` and Newton
    compose it.
  - In a multi-population problem, each population's `Model` carries its own F_k, which reads that population's
    density alone. Coupling across populations belongs in H_k.
  - **Refused by name:**
    - a plain `F[m] -> float`, which carries its own quadrature, and an `EnergyFunctional` already bound to a grid.
      The message names `create_lions_source` and `source_term_hjb`;
    - `PotentialCost(V)` with a `V(t, x)`: F is autonomous, and binding would read V at t = 0 only. The message
      names the Hamiltonian's `potential(t, x)`;
    - a coupling cost on a network, where no solve takes a `Model` (#2045). The message names
      `NetworkMFGComponents(node_potential_func=...)`;
    - FEM and meshless geometries, until stage 1b.
  - Which fields declare a problem's HJB source now has one owner, `hjb_source_fields`. The grid couplers,
    `GraphMFGSolver`, `BaseHJBSolver.solve`'s refusal and multi-D WENO's refusal read it, so each refusal names
    `coupling_cost` too.
  - `create_lions_source` and `create_nonlocal_source` read an n-D grid's density slice, with weights per node or
    one scalar weight. They refused every 2-D array as a time trajectory, so the route the plain-`F` refusal names
    failed on 2-D grids, and the message sent the user looking for a trajectory they had not passed.
