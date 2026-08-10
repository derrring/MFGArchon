- **The 2-D capability cells now run at a coupling strength comparable to their 1-D sibling's**
  (Issue #1865). `coupling=lambda m: m` reads the density directly, and a probability density on a
  2-D grid is intrinsically peakier than the 1-D one carrying the same mass — 1.818 against 9.549 on
  these fixtures — so the 2-D cells were applying **5.3x** the interaction and the matrix was
  reporting that as an effect of dimension. The scale is derived, `peak_1D / peak_2D`, computed from
  the two fixtures rather than written down: a literal has to be maintained against both, and the
  first version of it (`1.8180 / 9.5495`) was already wrong in the seventh digit.
- **Three cells go UNSUPPORTED -> PASS, and not because a solver improved.** `fdm_upwind_2d`,
  `fdm_centered_2d` and `fvm_muscl_2d` failed at the unscaled coupling and solve at the comparable
  one. That is the direction `--check-baseline` exists to guard, so it is stated here, in the
  baseline's own `_comment`, and in the fixture docstring: **the configuration those cells used to
  fail at is no longer covered by the matrix.** It is the aggregating regime — `f` rewards density,
  Lasry–Lions monotonicity fails, uniqueness is not guaranteed and the Picard map is not a
  contraction — so "can this configuration solve at all" had no answer to certify and the cells were
  red for a reason no code change could address. It stays on #1865, and may earn a cell later if the
  library gains an outer solver that makes the regime answerable.
- **The old docstring claimed four things that were false.** It said the fixture was the 1-D one
  "unchanged in everything but dimension. Deliberately the same Gaussian". Measured: `exp(-30 r^2)`
  against `exp(-10 r^2)`, `T` 0.2 against 1.0, `Nt` 6 against 10, and `sigma` **0.4 against 0.0**.
  All four stay, and the new docstring tabulates them with reasons rather than repeating the claim.
  The `sigma` row is the load-bearing one: the 1-D fixture runs at zero diffusion, which is the
  degenerate regime where the Godunov residual is non-smooth and Newton can reach a non-viscosity
  branch (#1878), so lifting it would make the 2-D cell a second instance of that problem rather
  than a test of dimension. Measured for the same reason: a faithful lift runs 3.2 s against this
  fixture's 3.9 s and still fails, so matching them buys honesty and no answer.
- **Pinned by what is not tautological.** The scale being derived in code means "the constant equals
  its own derivation" proves nothing, so `tests/unit/test_capability_2d_coupling_scale.py` asserts
  instead that the two crowds see the same `f(m)` — which fails the moment the scale stops being
  applied. Mutation-verified: replacing the derived scale with `1.0` turns it red. Two earlier
  versions of its helper reached for `hamiltonian.coupling` and `hamiltonian.evaluate_hamiltonian`,
  neither of which exists on `SeparableHamiltonian`; both assertions were red with `AttributeError`
  for a reason unrelated to what they test, and the mutation produced the identical red. **Only
  running the mutation showed it** — a test that fails for the wrong reason discriminates nothing.
