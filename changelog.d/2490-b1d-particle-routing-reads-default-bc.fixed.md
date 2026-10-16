- **`FPParticleSolver` chooses its reflecting fast path by the condition every wall carries, `default_bc`
  included** (#2490 stage B1d). The routing read the segments alone: a reflecting segment beside
  `default_bc=DIRICHLET` took the reflecting fast path, so the walls the default makes absorbing reflected every
  particle. It now reads `BoundaryConditions.uniform_segment`, which counts `default_bc` wherever no global segment
  covers the boundary. The fast path serves only where one reflecting or periodic condition covers every wall:
  four NO_FLUX faces beside a NO_FLUX default now take it, and a REFLECTING segment beside a NO_FLUX default, two
  conditions, takes the per-point path. The GPU path's refusal names those reasons; it said every BC it refuses
  needs particles removed at a DIRICHLET segment.
