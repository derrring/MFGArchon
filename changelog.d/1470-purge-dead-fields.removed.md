- **Removed dead `NetworkMFGComponents.diffusion_coefficient` / `drift_coefficient`** (Issue #1470 Strand A).
  Neither field was ever read by any solver — the network FP diffusion is `FPNetworkSolver`'s own knob
  (#1532) and the FP drift is `fp_drift_coefficient` — so setting them on the components was silently
  ignored. Removed (flagged DEAD in #1535). Minor breaking change: passing them now raises `TypeError`,
  but the kwargs never had any effect.
