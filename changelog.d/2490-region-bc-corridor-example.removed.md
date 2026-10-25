- **`examples/advanced/region_based_bc_corridor.py` is removed** (#2490). It failed at construction, and past that its
  sub-face inlet and outlet were refused by the FDM ghost path (#2472). The guide's § *Region-Based Boundary
  Conditions* shows what it showed, in the form the library accepts.
