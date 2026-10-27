- **A grid density reader deposits a sample whose kernel the grid cannot read** (#2556, C3-2, #2820). Before it builds a
  kernel, each of these readers asks whether the grid's lattice can read it: `FPParticleSolver`'s 1-D, n-D and GPU
  density readers, and the particle-to-grid projector on a uniform grid. The sample must have a kernel
  (`kde_sample_has_spread`). The bound B = Σ_{m≠0} |ĝ_H(k_m)| |φ̂_N(k_m)|, k_m = 2πm/h, on the kernel estimate's
  aliasing on the lattice must also be at or under 10% (`kde_lattice_aliasing`, `kde_resolves_on_grid`,
  `KDE_LATTICE_ALIASING_TOL`). By Poisson summation, B bounds how far the estimate's sum over the nodes misses its mass.
  A sample that fails is deposited on the grid's nodes by their tents (`deposit_on_nodes`), which holds its mass and its
  mean on the grid's measure, walls included. The solver records the levels it deposited (`deposited_levels`, also on
  `FPParticleResult`) and logs them once per solve. The projector logs each deposit.
  - What such a sample got before (at f40ddecd, ahead of #2556's C3; a 2-D 11 × 11 grid, 300 particles of sd 0.005):
    - mass 3.6e-50 between nodes under `reflection` and `standard`;
    - mass 1 on a node under `reflection`, and 55 under `standard`;
    - with `M_initial` given, a carried factor that reached mass 5.2e27 four steps later.

    Two survivors of an absorbing exit in 2-D raised a `RuntimeError` wrapping scipy's `LinAlgError`. A 1-D sample with
    no spread went to its nearest node at 1/Dx, and an n-D point to its nearest node. Each of these is now the deposit,
    mass 1 on the grid's measure.
  - `kde_method="cic"` is the same tent deposit. It was a cell-centred deposit read at the nodes, whose mass on the grid's
    measure was n/(n-1) per axis: 1.21 on an 11 × 11 grid.
  - A sample the lattice can read still gets the kernel estimate.
  - The projector's check needs a lattice, so a grid with a non-uniform axis keeps the kernel estimate (#2827).
