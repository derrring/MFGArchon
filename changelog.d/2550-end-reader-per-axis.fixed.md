- **`ParticleApplicator.apply` acts per axis at a corner one of whose faces is periodic** (#2550; #2803's shape, for a
  reader that has only where a step ended). It gave such a particle one action for every axis: the corner's segment
  wrapped every axis or folded every axis, so an end past a channel's corner wrapped across its wall or folded across
  its seam. Now the corner's segment decides only whether the particle is absorbed, and otherwise each face the end is
  at or past acts on its own axis by its own segment: a periodic face wraps it, any other folds it (CONVENTIONS.md
  § *Particles at a wall*). `apply_with_flux_limits` reads the same rule, and a particle a full exit turns back is
  returned by the faces it reached, each on its own axis.
- **`MeshfreeApplicator`'s absorbing mode reads the absorbing wall's owner** (`corner.absorb_at_walls`). On a geometry
  that is not a box, a particle exactly on its bounding box's face and off the geometry now leaves; it was kept. On a
  box nothing changes.
- Removed with no caller: the private `FPParticleSolver._apply_boundary_conditions_with_flux_limits`.
