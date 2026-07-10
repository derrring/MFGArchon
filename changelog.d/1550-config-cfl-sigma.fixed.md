- **Config CFL number uses the PDE coefficient D = sigma^2/2** (Issue #1550). `MFGGridConfig.cfl_number`
  and `validate_cfl_stability` computed the diffusive CFL with bare `sigma^2` (= 2D), so the
  stability warning fired at twice the correct threshold and `cfl_number` disagreed with the
  #1426/S0-14-corrected solver diagnostics (hjb_fdm/fp_fdm). Both now use `0.5*sigma^2`. The `sigma`
  field description is corrected to "SDE volatility" (it was mislabelled "Diffusion coefficient").
  Diagnostic/validation-only; no solve numerics change.
