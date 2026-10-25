- **The particle FP reads a step under a periodic boundary condition by its path** (#2704, #2803).
  `FPParticleSolver` now ends every CPU step at its first contacts with the particles' boundary
  (`reflect_at_first_contacts`, #2733), a periodic outer boundary's included: at a periodic face the rest of the step
  goes on from its image on the opposite face. Before, a BC that declared a periodic face took
  `ParticleApplicator.apply`, which reads where a step ended:
  - at a corner it gave one action for every axis, so a step past a channel's corner wrapped through the wall, or
    reflected across the seam (#2803);
  - an exit the path crossed was read by the corner order at the end: on an exit strip beside periodic faces it
    absorbed 188 particles where 196 crossed (#2704 comment 6335147485).

  At a corner one of whose faces is periodic, the owner's corner order decides only whether the particle is absorbed;
  otherwise each face acts on its own axis (CONVENTIONS.md § *Particles at a wall*). The loop searches once per
  period a step crosses, so a step that wraps more than `MAX_WRAPS` (1000) times is refused by name: one of about 1e17
  periods would never end, since its shift by one period rounds away.
- **Obstacles with a periodic outer boundary are placed in 1-D and 2-D**, refused since #2688. An obstacle touching a
  face of a periodic axis is refused by name, since a wrapped step would go on from an image it could hold. In 3-D and
  above a particle domain under any periodic face stays refused (#2807).
- Removed with their last callers: the private `FPParticleSolver._apply_boundary_conditions_nd` and
  `_apply_boundary_conditions_segment_aware`, and the unexported `fp_particle_bc.apply_boundary_conditions`.
