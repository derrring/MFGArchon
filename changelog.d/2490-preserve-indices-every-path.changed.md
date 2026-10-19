- **`preserve_indices=True` with a BC that absorbs nothing is accepted on every CPU particle path** (#2490). Which
  paths accepted it depended on the BC's form. Under a BC that is not segment-aware -- uniform no-flux, say -- the grid
  paths accepted it and the callable-drift path refused it. Under two or more no-flux segments on the faces the
  callable-drift path accepted it and the grid paths refused it. With nothing absorbed the trajectory keeps every
  particle's row, and is the one the solve keeps without the flag.
- On the callable-drift path the flag turns on the per-step particle history, as before, so under a BC that is not
  segment-aware `solve_fp_system` now returns an `FPParticleResult` where it raised.
- The flag does not turn on the per-step particle history on the grid paths: a history makes `solve_fp_system`
  return an `FPParticleResult`, and the coupling iterators read their return as the density array. The preserved
  trajectory is `M_particles_trajectory`.
