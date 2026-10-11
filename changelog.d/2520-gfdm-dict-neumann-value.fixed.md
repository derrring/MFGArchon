- **HJB-GFDM solved a dict BC's Neumann value as 0** (#2520). Its Neumann row read a dict's `"values"` only as an
  index dict, so `{"type": "neumann", "values": g}` with a number or a callable `g` solved the homogeneous condition.
  - Measured on #2513's 1-D oracle (g = 0.7, nx 21): the max error was 5.0e-1 (Newton) and 5.5e-1 (Howard),
    bit-for-bit `neumann_bc(value=0)`'s.
  - The deprecated dict form is now translated to `neumann_bc(value=g)` (#2597) and solves bit-for-bit as it does, at
    2.4e-2.
