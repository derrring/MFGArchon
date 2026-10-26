- **`geometry.boundary.corner.absorb_at_walls`, the absorbing wall's owner for a reader that has only where a step
  ended** (#2550). A particle at or past a face of the box, within the reader's band, leaves: it is removed and never
  moved, and its exit position is its end. It returns which particles leave and their exit positions, and writes
  nothing. `walls_reached`, beside it, names the face each particle is at or past per axis. `ParticleApplicator` and
  `MeshfreeApplicator`'s absorbing mode read them.
