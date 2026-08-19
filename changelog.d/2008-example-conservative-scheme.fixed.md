`examples/basic/three_mode_api_demo.py` selected the non-conservative `gradient_upwind` advection
scheme in its expert-mode demonstration and lost 98.17% of its probability mass (#2008). It also
did not run to completion, before or after that line: `compare_schemes()` selects
`NumericalScheme.FDM_CENTERED`, which routes to `divergence_centered`, whose density goes negative
at timestep 3 — the mass-fabrication gate then refuses to clip rather than report a conserved
density it did not compute, and the script died there, short of its own summary. Measured: the
script exits **1** on `main` and **0** here.

Four changes, all in `examples/` and `docs/`:

- `advection_scheme` moves to `divergence_upwind`. Measured on the example's own problem:
  `gradient_upwind` runs `1.000000 -> 0.018302`, `divergence_upwind` holds `1.000000` to `-0.00%`.
  Of the four schemes `FPFDMSolver` accepts, it is the only one that solves this problem — the two
  `gradient_*` forms leak and `divergence_centered` aborts.
- `compare_schemes()` reports that abort as an outcome instead of ending the script. Refusing to
  fabricate mass is the behaviour worth demonstrating; crashing on it is not.
- Every mode now prints `result.mass_conservation_error`. The loop already computed it
  (`coupling/fixed_point_iterator.py:973`, since #1672) and it read **0.9817** on the leaking
  configuration against `5.1e-15` on the fixed one — but nothing gates on it and nothing printed
  it, so a 98% drain and a converged residual looked identical from the output. `err_M` measures an
  increment, and it read `9.27e-07`: **0.93x** the `1e-6` tolerance, not orders under it.
- `max_iterations` 20 -> 60 in all three modes, previously only one. All three converge at 49.

`docs/user/three_mode_api_migration_guide.md` selected `gradient_centered` in **both of its "After"
blocks** — the code the guide tells users to migrate *to*. Those move to `divergence_upwind`, with
a note that the scheme change is not part of the API migration. The "Before" block keeps
`gradient_centered`, since it depicts legacy code rather than recommending it.

Nothing in any gate runs either file: `pytest.ini` is `testpaths = tests`, every workflow scopes
`pytest` to `tests/`, and `scripts/local_ci.sh` scopes `ruff` to `mfgarchon/` and `tests/`.

On the iteration bump, stated precisely because the first reading of it was wrong: the old
configuration converges at **27** when given a cap of 60, so the scheme change accounts for 27 -> 49,
not 20 -> 49. Losing the mass made the problem easier. (At the shipped cap of 20 neither converged.)
Note also that 49 is a dip out of a plateau rather than a settled fixed point, and the neighbouring
grids are not robust — `Nx=41` does not converge in 200 iterations — so 60 is right for this
configuration and is not a general margin.
