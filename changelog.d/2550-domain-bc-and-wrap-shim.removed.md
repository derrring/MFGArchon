- **`Hyperrectangle.apply_boundary_conditions` and `ImplicitDomain.apply_boundary_conditions` are removed**, deprecated
  since v0.12.0 (#2550). They had no library caller, and each carried its own copy of the particle wall rules. The
  replacement is `MeshfreeApplicator(domain).apply_particle_bc(particles, bc_type)`, which differs from them:
  - `Hyperrectangle`'s "periodic" shifted a particle by one span, so one more than a span past a face stayed outside;
    the applicator wraps any number of spans. Its "absorbing" kept a particle exactly on a face; the applicator removes
    it.
  - `ImplicitDomain`'s "reflecting" projected to the interior and its "absorbing" kept what `contains` holds, and its
    "periodic" raised. On a geometry that is not a box the applicator folds and absorbs against the bounding box
    (#2693), and it wraps there.
- **`geometry.boundary.corner.wrap_positions` is removed**, deprecated since v0.17.0 (#2550). Use
  `geometry.boundary.periodic.wrap_positions`, which it called. The export `geometry.boundary.wrap_positions`, which
  resolved to it, now resolves to the periodic owner: the same values on its signature, without the warning.
