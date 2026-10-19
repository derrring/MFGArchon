- **`FPParticleSolver(preserve_indices=True)` works on every CPU particle path under a segment-aware BC** (#2490,
  after #1119). The 1-D path and the grid-drift n-D path raised `NotImplementedError` whenever the BC was
  segment-aware -- an absorbing segment, or segments placed on the faces even when all are no-flux; only the
  callable-drift path kept each particle's row. All three now keep it through one owner of the bookkeeping,
  `_IndexPreservingTrajectory`: particle j is row j of every slice of `M_particles_trajectory`, NaN once absorbed, and
  the density is bit-identical to the same solve without the flag. The GPU path refuses an absorbing BC (#1910) and
  is unchanged.
