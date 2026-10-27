- **`kde_lattice_aliasing`, `kde_resolves_on_grid`, `KDE_LATTICE_ALIASING_TOL` and `deposit_on_nodes`**
  (`mfgarchon.utils.numerical.particle.interpolation`, #2556): whether a kernel estimate can be read on a lattice, and
  the deposit a grid reader uses where it cannot. `kde_lattice_aliasing` gives the bound B on how far the estimate's sum
  over the lattice's nodes misses its mass. It is decided against a tolerance and summed only until the decision is
  made. It is None for a sample with no kernel. `kde_resolves_on_grid` decides it against `KDE_LATTICE_ALIASING_TOL`
  (0.1). `deposit_on_nodes` shares each particle's mass among the nodes of its cell by their tent functions and divides
  by the grid's quadrature weights. It refuses a position outside the box by name.
