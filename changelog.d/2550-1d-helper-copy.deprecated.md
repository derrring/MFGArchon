- **`utils.numerical.particle.apply_boundary_conditions_gpu` / `_numpy` are deprecated** (#2550). They were a second
  copy of `mfgarchon.utils.particle_utils`' pair, the one the particle solver's 1-D GPU path calls, and now call it:
  the same values, with "dirichlet" refused by name.
