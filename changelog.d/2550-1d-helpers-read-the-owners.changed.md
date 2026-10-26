- **The 1-D particle helpers refuse "dirichlet" and a non-finite position by name** (#2550, #2485).
  `utils.particle_utils.apply_boundary_conditions_numpy` / `_gpu` clamped a particle to the wall for "dirichlet",
  which keeps the mass an absorbing wall should take; a function that returns positions cannot remove a particle, so
  both now raise `ValueError`, and so does the deprecated pair in `utils/numerical/particle/boundary.py`, which calls
  them. A non-finite position, which both kept or turned into NaN, is refused as the absorbing wall's reach refuses it
  (`corner.walls_reached`). The particle solver's 1-D GPU path passes only "periodic" or "no_flux", and now raises on
  a non-finite particle position, as its CPU path does, where it returned a density whose later rows were not finite.
- **`apply_boundary_conditions_numpy` folds and wraps through the owners** (`corner.reflect_positions`,
  `periodic.wrap_positions`). On an axis whose span is at most 1e-14 it now leaves the particles as they are, as the
  owners do, where its wrap returned NaN at a span of 0 and moved them onto the lower bound above it; the GPU wrap does
  the same. A non-float input is read as float64, and a float32 one is computed against float64 bounds.
