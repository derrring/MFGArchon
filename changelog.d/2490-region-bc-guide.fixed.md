- **The boundary-conditions guide's region sections, and two refusals' advice, match what the library does** (#2490).
  - The guide's § *Region Marking Methods* marked one name twice, and a second `mark_region` with a name raises. It now
    builds a region over several faces with `mark_region(name, mask=grid.union_regions(...))`.
  - § *Priority Resolution* overlapped two sub-face strips that every face-level path refuses (#2472), and did not import
    `BoundaryConditions`. It now shows an overlap of whole faces, which the face-level and per-point readers both
    resolve. It also says which solvers read sub-face strips per point.
  - Two refusals of a sub-face cover told the user to "use a solver that resolves boundary conditions per point".
    - Both now name `HJBGFDMSolver` and `FPParticleSolver` and point to the guide's § *Priority Resolution*, which
      lists, as measured, for which conditions of the BC each reads a region on part of a face.
  - `mixed_bc_from_regions`' docstring said a per-point reader matches a kept region point by point. It now says how
    the point reader reads it, and points to the guide's list.
