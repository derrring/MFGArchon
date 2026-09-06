- **`SolverResult.mass_conservation_error` reports particle absorption, instead of being blind to**
  **it** (Issue #2188). `FixedPointIterator` measured mass conservation by integrating the grid
  density `M` over the domain — correct for a grid-based FP solver, wrong for `FPParticleSolver`.
  `FPParticleSolver`'s `M` is a kernel-density reconstruction that integrates to ~1 *whatever
  particle count it is built from* — 2000 particles and 9 particles both reconstruct to a
  normalised density — so absorption was structurally invisible to the grid measurement. Measured
  in the issue: an absorbing boundary that killed 99.6% of the particles reported **less** error
  (4.7079e-02) than a no-flux solve that lost none (4.9722e-02); the reported number was
  anti-correlated with the defect the field exists to detect, and what it actually tracked was
  `sigma`, through the KDE bandwidth.

  New seam: `BaseFPSolver.mass_conservation_error_override()`, default `None` (meaning "nothing
  solver-specific, use the generic grid measurement" — every non-particle solver is unaffected).
  `FPParticleSolver` overrides it to report the fraction of particles absorbed, computed from
  `M_particles_trajectory` — the state the grid projection discards. Same functional form as the
  measurement it supplements: `max_t |count_t / count_0 - 1|` in place of
  `max_t |mass_t / mass_0 - 1|`.

  **Three representations of "how many survived", not two.** A non-absorbing BC stores a
  fixed-shape trajectory array. An absorbing BC with the default `preserve_indices=False` stores a
  list of variable-length arrays, whose own length is the surviving count. `preserve_indices=True`
  *also* stores a list, but every entry has the **same** length — absorbed particles are marked
  `NaN` in place rather than dropped, so counting by `len()` alone silently reproduces #2188 one
  representation down. Found only by running that mode and checking, not by reading the storage
  method's docstring, which does not mention it. All three are handled; the two absorbing
  representations were verified to agree exactly on the same random seed and physics.

  `SolverResult.mass_conservation_error`'s docstring now says the functional is
  solver-family-dependent, per the issue's own closing request.
