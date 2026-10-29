- **A grid density reader deposits a sample whose kernel the grid cannot read** (#2556, C3-2, #2820). Before it builds a
  kernel, each of these readers asks whether the grid's lattice can read it: `FPParticleSolver`'s 1-D, n-D and GPU
  density readers, the particle-to-grid projector on a `TensorProductGrid`, and `interpolate_particles_to_grid(method="kde")`
  on a lattice, the one it lays out itself or a caller's `grid_axes` (a callable bandwidth is read through scipy's own
  estimate). A grid's own spacing is read whatever its nodes' rounding: the one the solver's grid declares, or the one
  made up from a geometry's bounds where it has no grid (#2757), and the one a `TensorProductGrid` declared uniform
  gives the projector. A declared spacing that is not positive and finite, as integer bounds whose span wraps in their
  dtype give (#2836), is replaced by the nodes' own lattice in both. Axes no grid declares a spacing for (the utility's own and a
  caller's, and a `TensorProductGrid` declared custom) are a lattice where their nodes lie on one to within 1e-3 of its
  spacing, so an axis rounded to float32 or far from 0 is one while its rounding stays under that. The sample must
  have a kernel
  (`kde_sample_has_spread`). The bound B = Σ_{m≠0} |ĝ_H(k_m)| |φ̂_N(k_m)|, k_m = 2πm/h, on the kernel estimate's
  aliasing on the lattice must also be at or under 10% (`kde_lattice_aliasing`, `kde_resolves_on_grid`,
  `KDE_LATTICE_ALIASING_TOL`). By Poisson summation, B bounds how far the estimate's sum over the nodes misses its mass
  on the infinite lattice of the grid's nodes, or on a periodic box's whole cells. On a box with walls, the mass the
  kernel puts past a wall is a separate matter, the boundary estimator's.
  A sample that fails is deposited on the grid's nodes by their tents (`deposit_on_nodes`), which holds its mass and its
  mean on the grid's measure, walls included. The solver records the levels it deposited (`deposited_levels`, also on
  `FPParticleResult`) and logs them once per solve. The projector logs each deposit and records what it leaves out.
  - The projector and `interpolate_particles_to_grid` deposit the sample's restriction to the grid's box, as the
    kernel estimate is: a particle outside it is left out, each particle kept carries 1/N of the whole sample, and the
    projector records the count left out and their share of the mass (`particles_outside_last_deposit`). The solver's
    readers refuse a particle outside the box by name.
  - A position past a wall by at most the spacing of that wall's bound, in the dtype the positions were computed in, is
    on the wall: a float32 device holds 0.3 as 0.30000001, and the GPU path deposits a particle there on the wall.
  - Under the default `kde_normalization`, a deposited level carries the caller's mass exactly; the carried factor,
    which corrects a kernel reconstruction's drift, is calibrated on the first kernel level of the solve.
  - What such a sample got before (at f40ddecd, ahead of #2556's C3; a 2-D 11 × 11 grid, 300 particles of sd 0.005):
    - mass 3.6e-50 between nodes under `reflection` and `standard`;
    - mass 1 on a node under `reflection`, and 55 under `standard`;
    - with `M_initial` given, a carried factor that reached mass 5.2e27 four steps later.

    Two survivors of an absorbing exit in 2-D raised a `RuntimeError` wrapping scipy's `LinAlgError`. A 1-D sample with
    no spread went to its nearest node at 1/Dx, and an n-D point to its nearest node. Each of these is now the deposit,
    mass 1 on the grid's measure.
  - On the CPU readers, `kde_method="cic"` is the same tent deposit. It was a cell-centred deposit read at the nodes, whose
    mass on the grid's measure was n/(n-1) per axis away from the walls (1.21 on an 11 × 11 grid) and less near them
    (0.3025 for a sample at a corner). The GPU path does not read `kde_method`.
  - A sample the lattice can read still gets the kernel estimate.
  - The check needs a lattice, so these keep the kernel estimate (#2827): a projector target with an axis of one node,
    one declared custom whose axes are not a lattice, or one that is not a `TensorProductGrid`; and axes of
    `interpolate_particles_to_grid`, its own or a caller's, that are not a lattice (a non-uniform axis, an axis of one
    node, bounds with no span, or nodes rounded past 1e-3 of the spacing).
