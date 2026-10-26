- **`ParticleApplicator.apply`, `apply_with_flux_limits` and `MeshfreeApplicator`'s absorbing mode refuse a non-finite
  particle position by name** (#2550). No wall rule can place one, since its fold, wrap and exit are undefined.
  `ParticleApplicator` kept it -- a NaN unmoved, an infinite coordinate turned into NaN by its fold or wrap -- or,
  past an absorbing face, counted it as an exit; the absorbing mode dropped it as though absorbed. Now the absorbing
  wall's reach (`corner.walls_reached`) raises `ValueError`, naming the first such row. The particle solver's own path
  already refused one (`ImplicitDomain.first_contact`). The absorbing mode on a geometry with no bounds keeps every
  particle, as before, and reads no wall.
- **At a corner, `ParticleApplicator` reads every face's segment**, so a face whose type has no particle interpretation
  (`EXTRAPOLATION_LINEAR` / `_QUADRATIC`) is refused there even where another face wins the corner, unless the winner
  absorbs the particle; it used to be refused only where it won.
