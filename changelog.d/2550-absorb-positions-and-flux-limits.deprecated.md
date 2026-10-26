- **`geometry.boundary.corner.absorb_positions` and `ParticleApplicator.apply_with_flux_limits` are deprecated**
  (#2550).
  - `absorb_positions` clamps a particle to the wall and keeps it, which is not an absorbing wall: there the mass that
    reaches the wall leaves through it. It still clamps while it warns. For an absorbing wall use `absorb_at_walls`; for
    a clamp, `np.clip`.
  - `apply_with_flux_limits` has no library caller: no FP solver honours a capacity-limited exit, and each refuses a
    segment carrying `flux_capacity` (#2575). With no limit it is `apply`.
