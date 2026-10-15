- `fp_particle_density.sample_particles_from_density` is removed (#2688).
  - Its `implicit_domain=` branch imported `sample_from_density_with_domain`, which does not exist, so it raised
    `ImportError`.
  - Nothing called it.
  - Use `mfgarchon.utils.numerical.particle.sample_from_density(..., navigable=domain.contains)`.
