- **`examples/advanced/region_based_bc_corridor.py` is removed** (#2490). It failed at construction, and past that its
  sub-face inlet and outlet were refused by the FDM ghost path (#2472). The guide's § *Region-Based Boundary
  Conditions* shows its whole-face regions and priority in a form the library accepts, and says which solvers read a
  region on part of a face. No example now runs a solve with a sub-face placement.
