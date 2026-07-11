- **Diffusion-magnitude gate now pins the weak-form (FEM) FP solver** (Issue #1566). The standing
  `test_diffusion_magnitude_gate.py` docstring named `#1152` (weak-form used `volatility_field`
  directly as `D`, skipping the `/2`) among the bugs it catches, but no test instantiated
  `WeakFormFPSolver` — the paper solver's diffusion magnitude was named-but-unpinned. Added
  `test_weak_form_fem_fp_diffusion_magnitude`: a cosine eigenmode under pure diffusion on a 1D
  `FPFEMSolver` must decay at `exp(-D k^2 T)` with `D = sigma^2/2`; mutation-verified to fail on the
  `#1152` factor error. Coverage note updated to list the weak-form path.
