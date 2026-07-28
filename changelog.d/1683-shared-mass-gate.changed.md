- **One owner for the clip-would-fabricate-mass invariant**
  (`mfgarchon.utils.numerical.mass_fabrication_gate`, Issue #1683) — nine FP paths clip a
  negative density and several then renormalise, which makes a diverging solve
  indistinguishable from a healthy one: finite, non-negative and exactly mass-conserving.
  The first two sites are migrated. **Breaking**: the strict-adjoint FP-FDM step
  (`solve_fp_step_adjoint_mode`) now raises instead of clipping-and-renormalising when the
  clip would create more than 1e-8 of the present mass; it previously returned a repaired
  density that reported exact conservation over an 8.39% clip.
