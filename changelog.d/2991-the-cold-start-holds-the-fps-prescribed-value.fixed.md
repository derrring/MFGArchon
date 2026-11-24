- **A coupled route's cold start holds the FP's own value at a node the FP prescribes** (#2991). FP-FDM's implicit
  Dirichlet rows and FP-FEM's condensed dofs hold a value p at every later level whatever they are handed: 0 on the
  problem's Dirichlet face, which the FP reads as an exit, and g(t_n) on the FP's own DIRICHLET(g). The cold starts put
  `m_initial` there, and each sweep blends the FP's row into the iterate with a weight theta < 1, so the returned
  density kept prod(1 - theta_j)(m_initial - p) there after any number of sweeps, and the first sweep handed the HJB
  `m_initial` there. The FP solver now says what its step prescribes, `BaseFPSolver.prescribed_density` (none by
  default): FP-FDM through `prescribed_nodes_and_values`, the function its step's right-hand side reads, and FP-FEM
  through `_dirichlet_dofs_and_values`, the read its step condenses with. The cold start asks it: `initialize_cold_start`
  (FixedPoint, FictitiousPlay, the block iterators, `problem.solve`), `MultiPopulationIterator`'s initial iterate, and
  the first-sweep iterate of `GraphMFGSolver` and `RegimeSwitchingIterator` (`cold_start_iterate`). A warm start is the
  caller's array and is left as handed, so it keeps its remnant.
  - Measured with the walls planted at 7.0 (1-D, 11 nodes, Nt = 5, relaxation 0.5): one sweep returned 3.5 at the
    walls on the problem's exit, and returns 0. RegimeSwitching's first sweep already took the FP's row whole; what
    changes there is its first HJB's input, its first FP source's density, and the result it returns when its first HJB
    diverges.
  - **What may change.** Any output computed from the iterate's later levels at those nodes may differ: the returned
    density at a finite iteration count, convergence and the iteration count, U under a coupling that reads the density
    there or beyond its own node, and the terminal value the first sweep reads at T. Measured instances, not a complete
    list, are in #2991.
  - A non-finite or raising g(t_n) on the FP's own BC fails in the cold start, by the name the FP's step fails by.
    Where the HJB reads that g too, the solve now fails there, before the HJB, by the FP's name.
  - An FP solver whose class does not define `prescribed_density` (one that does not subclass `BaseFPSolver`) is
    refused by name in the cold start, a TypeError: define it, or subclass `BaseFPSolver`, whose default is None. Main
    ran it with `m_initial` at every later level. Every FP solver the library ships subclasses `BaseFPSolver`.
