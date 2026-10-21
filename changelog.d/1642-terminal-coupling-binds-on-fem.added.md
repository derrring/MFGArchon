- **HJB-FEM P1 binds `Model(terminal_coupling=...)` on its own discretization** (#1642, stage 1b's G line). On a mesh
  the problem declares G and binds none: `problem.declared_terminal_coupling` holds it, and `terminal_energy` is None
  there. `HJBFEMSolver(order=1)` binds it at construction on the point measure it binds F on, its dofs with the
  consistent mass matrix's row sums, and returns it from `terminal_binding()`.
  - `MFGProblem.terminal_value(m_T, binding=, reader=)` reads that binding off a grid. It refuses by name, naming the
    reader:
    - no binding;
    - a binding on other points;
    - a binding of another problem's G;
    - a binding beside the problem's own on a grid, one G with two bindings.
  - `NetworkMFGProblem.terminal_value` takes the same arguments and refuses a binding.
  - The couplers built on `BaseCouplingIterator` hand the solver's binding to `terminal_value`, as they hand F's to
    the HJB source. On a mesh G solves through `problem.solve()`, `FixedPointIterator` and the `BlockIterator` family.
    `GraphMFGSolver`, `RegimeSwitchingIterator` and `MFGResidual` pass none, and refuse G off a grid by name, as they
    refuse F. `MultiPopulationIterator` refuses a declared G on any geometry (the user's ruling 5).
  - With `PotentialCost(V)` as G the solve gives bitwise the same U as V added to `u_terminal`.
  - HJB-FEM P2 and the meshless solver refuse a declared G at construction, as they refuse F.
- **Behaviour change: a terminal coupling on a mesh or an implicit domain now constructs,** where the problem refused
  it.
