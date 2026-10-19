- **`preserve_indices=True` with a BC that absorbs nothing is accepted on every particle path** (#2490). The
  callable-drift path refused it, although with nothing absorbed the trajectory already keeps every particle's row; the
  grid paths accepted it. The trajectory is then the one the solve keeps without the flag.
- The flag does not turn on the per-step particle history on the grid paths: a history makes `solve_fp_system`
  return an `FPParticleResult`, and the coupling iterators read their return as the density array. The preserved
  trajectory is `M_particles_trajectory`. The callable-drift path keeps the history on, as before.
