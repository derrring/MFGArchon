- **`HJBGFDMSolver` now preserves a spatial `volatility_field` through every HJB path** (Issue
  #1725). The solver normalizes a scalar, native-shaped or flattened grid field, meshfree
  collocation field, or space-only callable once per solve into one collocation-space coefficient.
  Batch and per-point residual/Jacobian assembly, LLF augmentation, the DMP diagnostic, and Howard
  policy iteration all consume that same value. Nonconstant fields are no longer replaced by
  their mean, structured-grid fields are mapped to collocation points, implicit-domain fields
  keep their node ordering, callables are evaluated once per node, and invalid representations
  fail before sparse assembly. Time-/density-dependent callables fail instead of being frozen as
  static fields; a problem-owned `(d,d)` array that is ambiguous between a grid field and tensor
  volatility must be supplied as an explicit solve-level field. Scalar solves remain
  byte-identical.
