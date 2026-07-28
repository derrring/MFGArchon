- **The recommended semi-Lagrangian FP path stops instead of repairing** (Issue #1683) —
  `FPSLSolver._clip_nonneg` warned once per solve and returned the clipped density, so a
  run whose mass the clip had moved came back indistinguishable from a healthy one. It now
  routes through the shared mass-fabrication gate. **Breaking**: configurations whose clip
  creates more than 1e-8 of the present mass now raise. Measured across five
  configurations, four clip nothing and the fifth clips 11.9% while drifting 10.2%.
