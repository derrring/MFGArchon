- **The boundary-conditions guide's region sections, and two refusals' advice, match what the library does** (#2490).
  - The guide's § *Region Marking Methods* marked one name twice, and a second `mark_region` with a name raises. It now
    builds a region over several faces with `mark_region(name, mask=grid.union_regions(...))`.
  - § *Priority Resolution* overlapped two sub-face strips that every face-level path refuses (#2472), and did not import
    `BoundaryConditions`. It now shows an overlap of whole faces, which the face-level and per-point readers both
    resolve. It also says which solvers read sub-face strips per point.
  - Two refusals of a sub-face cover told the user to "use a solver that resolves boundary conditions per point".
    - The one for a `region=` or `sdf_region` placement now names `HJBGFDMSolver` and `FPParticleSolver`, which read it
      alone or coupled.
    - The one for a marked region now says that only `FPParticleSolver` with a callable drift reads it per point, and
      that a `region=` or `sdf_region` placement is the route (#2795).
