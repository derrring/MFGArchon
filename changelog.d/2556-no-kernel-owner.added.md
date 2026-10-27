- **`kde_lattice_aliasing`, `kde_resolves_on_grid`, `KDE_LATTICE_ALIASING_TOL` and `deposit_on_nodes`**
  (`mfgarchon.utils.numerical.particle.interpolation`, #2556): whether a kernel estimate can be read on a lattice, and
  the deposit a grid reader uses where it cannot. `kde_lattice_aliasing` decides B, a bound on how far the estimate's
  sum over the lattice's nodes misses its mass, against a tolerance. It returns a value above the tolerance exactly
  when B is, summed only until that is decided, or None for a sample with no kernel. `kde_resolves_on_grid` decides it
  against `KDE_LATTICE_ALIASING_TOL` (0.1). `deposit_on_nodes` shares each particle's mass among the nodes of its cell
  by their tent functions and divides by the grid's quadrature weights. It refuses by name a position outside the box,
  an empty sample, and weights that are not finite and positive at every node.
