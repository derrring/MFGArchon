- **The legacy-BC guard now has a test that measures what coercion costs** (#1714, #1559). The
  guard refuses to silently assemble a legacy `periodic` as no-flux, and its comment recorded the
  reason as "differs from canonical periodic_bc by O(1) once mass reaches the wall (verified with
  an off-center bump)" — a measurement that was made and never committed. Reproduced and pinned:
  an off-centre bump at x=0.12 leaves `3.24e-04` of mass in the last five nodes under no-flux and
  `1.10e-01` under periodic, a factor of 340, with a 55% relative L1 difference over the field.
  Coercion does not perturb the answer, it replaces it. Mutation-checked against the mechanism
  rather than the guard: routing periodic to the no-flux boundary handler reddens it.
