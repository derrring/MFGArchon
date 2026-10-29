- **`kde_lattice_aliasing`, `kde_resolves_on_grid`, `KDE_LATTICE_ALIASING_TOL` and `deposit_on_nodes`**
  (`mfgarchon.utils.numerical.particle.interpolation`, #2556): whether a kernel estimate can be read on a lattice, and
  the deposit a grid reader uses where it cannot. `kde_lattice_aliasing` decides B, a bound on how far the estimate's
  sum over the lattice's nodes misses its mass, against a tolerance. It returns a value above the tolerance exactly
  when B is, to within the rounding of its sums, and summed only until that is decided; except that a kernel too
  narrow for its lattice points to be enumerated gets infinity, read as unresolved whatever its B (a backstop that errs
  toward the deposit). It returns None for a sample with no kernel, and raises by name where its K overflows
  float64: where 2πσ/h passes 1.3408e154 on some axis, σ the kernel's width and h the spacing.
  `kde_resolves_on_grid` decides it against `KDE_LATTICE_ALIASING_TOL` (0.1). `deposit_on_nodes` shares each
  particle's mass among the nodes of its cell by their tent functions and divides by the grid's quadrature weights. A
  position past a wall by at most the spacing of its bound in ``dtype`` (the positions' own) is on the wall; one past
  it by more is refused by name, or left out with ``outside="drop"``. It refuses by name an empty sample and weights
  that are not finite and positive at every node.
