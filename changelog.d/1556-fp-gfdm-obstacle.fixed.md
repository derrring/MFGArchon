- **FPGFDMSolver threads obstacle_sdf into its TaylorOperator** (Issue #1556, G-004 FP residue). The FP
  GFDM operator was built with no obstacle SDF, so its density-derivative stencils (D_lap / D_grad)
  coupled straight through obstacle walls while the HJB-GFDM side was visibility-filtered (#1124) —
  asymmetric physics on the same cloud in a coupled obstacle solve. `FPGFDMSolver` now accepts
  `obstacle_sdf` / `visibility_samples` / `visibility_margin` and passes them to `TaylorOperator`,
  matching the HJB side. Default `None` is inert (no change without an obstacle).
