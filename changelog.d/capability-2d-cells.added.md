- **Four 2-D capability cells** (`scripts/capability_matrix.py`, Refs #1745) — every cell
  was 1-D, so a scheme could conserve mass to 2.2e-16 in one dimension and not run at all
  in two with the matrix silent. `SL_LINEAR` passes in 2-D; `FDM_UPWIND`, `FDM_CENTERED`
  and `FVM_MUSCL` all raise the same `ConvergenceError` from the `HJBFDMSolver` they share.
  UNSUPPORTED goes 3 → 6, and a test pins that any scheme with a 1-D mass cell has a 2-D
  sibling.
