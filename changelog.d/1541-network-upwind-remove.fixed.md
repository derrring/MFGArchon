- **FPNetworkSolver removes the identity-map "upwind" scheme and fails loud** (Issue #1541, RFC #1574).
  `scheme="upwind"` was a silent identity map — `_compute_edge_flow` returned the same positive flow
  for both edge orientations, so the paired flows cancelled to exactly zero net drift and the step had
  no diffusion term, leaving the density unevolved — and `"flow"` was never implemented. A correctly-
  implemented conservative upwind would be byte-identical to `"explicit"` (which already sources rates
  via `H.optimal_control` in inflow-outflow form + diffusion), a single-source duplicate. `scheme` is
  now validated to `{explicit, implicit}` at construction (`NotImplementedError` otherwise) and the
  dead `_upwind_step` / `_compute_edge_flow` methods are removed.
