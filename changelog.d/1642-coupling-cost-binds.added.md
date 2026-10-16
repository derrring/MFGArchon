- **`Model(coupling_cost=...)` is bound and solved** (#1642, stage 1). It was accepted and read by nothing.
  - The value is a specification, from `mfgarchon.operators.interaction`: `InteractionCost(kernel)` for
    F[m] = (1/2) ∫∫ K(x − y) m(x) m(y), `PotentialCost(V)` for F[m] = ∫ V m, or their sum (`a + b`, `CombinedCost`).
    Each binds to the `EnergyFunctional` it names: `QuadraticInteractionEnergy`, `PotentialEnergy`, `CombinedEnergy`.
  - `MFGProblem` binds it at construction on its grid and measure, as `problem.coupling_energy`, and binds it again
    on a rebuild. The HJB carries its flat derivative δF/δm at each density slice as a cost-signed source, summed with
    the Hamiltonian's own coupling f and with `source_term_hjb`. Every grid coupler, `GraphMFGSolver` and Newton
    compose it.
  - In a multi-population problem, each population's `Model` carries its own F_k, which reads that population's
    density alone. Coupling across populations belongs in H_k.
  - **Refused by name:**
    - a plain `F[m] -> float`, which carries its own quadrature. The message names `create_lions_source` and
      `source_term_hjb`;
    - `Model(terminal_coupling=...)`, which was read by nothing, until stage 2 binds it;
    - a coupling cost on a network, where no solve takes a `Model` (#2045). The message names
      `NetworkMFGComponents(node_potential_func=...)`;
    - FEM and meshless geometries, until stage 1b.
  - Which fields declare a problem's HJB source now has one owner, `hjb_source_fields`. The grid couplers,
    `GraphMFGSolver`, `BaseHJBSolver.solve`'s refusal and multi-D WENO's refusal read it, so each refusal names
    `coupling_cost` too.
