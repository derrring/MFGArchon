- **`HJBGFDMSolver` now consumes a spatial `volatility_field` instead of its mean** (Issue #1725).
  The batch residual and Jacobian each resolved sigma through a byte-identical copy of one
  expression ending in `_get_sigma_value(None)`, whose documented contract collapses an array to
  its mean and evaluates a callable once at the domain centre — so `volatility_field=linspace(0.1,
  0.5, N)` produced a solve byte-identical to the scalar `0.3`. `assemble_hjb_residual` has
  accepted a per-node field since Issue #1071, so the coefficient was being discarded one layer
  above an assembly that could consume it. Both copies now route through one `_batch_sigma()`
  owner. Scalar solves are bit-identical; LLF augmentation is unchanged.
