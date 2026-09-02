Three places now say that **scheme choice depends on the diffusion regime, not only on dimension** —
and that the library reports the pairing rather than enforcing it.

**Diagnosed first.** #1878 recorded that the 1-D `HJBFDMSolver`'s inner Newton fails on most time
steps of the smoke fixture and left the cause open, suggesting "the Jacobian or the assembly rather
than a hard problem". Measured, both halves are refuted:

- `compute_hjb_jacobian` agrees with a finite-difference Jacobian of `compute_hjb_residual` to
  **2.4e-08 … 2.0e-06** relative, against an FD-vs-FD control of 3.7e-06 — at the noise floor.
- At a real failing state the residual is **smooth**: one-sided directional derivatives agree at
  **261.543** from both sides.
- `scipy.least_squares` from zero reaches `|R|_inf = 2.2e-04` in **5099** function evaluations,
  where the same call at the healthy neighbouring time index reaches 3.0e-14 in **11**.

It is a hard problem, and the reason is the regime: `_smoke_problem` passes no `sigma`, so it takes
`MFGProblem`'s documented deterministic default `sigma = 0`, the HJB is **first order**, and
characteristics cross. Measured on six 1-D fixtures — varying terminal condition, initial density,
coupling strength and resolution — counting the library's own non-convergence warnings over 5 Picard
sweeps:

| | `sigma = 0` | `sigma = 0.4` |
|---|---|---|
| FDM_UPWIND warnings | 9 … 28 | **0 on all six** |
| FDM_UPWIND outer error | 17 … 3000 | 0.002 … 0.085 |
| SL_LINEAR outer error | 0.099 … 0.332 | — |

**Refining the grid makes it worse** — 12 warnings at `Nx = 21` against 28 at `Nx = 41` — so it is
not accuracy you can spend resolution on. `SL_LINEAR` uses **no Newton at all** on this class,
because a semi-Lagrangian scheme follows characteristics instead of solving a nonlinear system
across them.

**What changed:**

- The inner-Newton warning now names the **regime** when `sigma == 0`, not only the symptom, and
  points at `SL_LINEAR` or at stating the model's diffusion. A caller reading "residual stopped
  decreasing" goes looking for a bug in the Jacobian; there isn't one.
- `NumericalScheme`'s docstring gains a *Regime, not only dimension* section. Its per-scheme "Use
  case" lines are about dimension, scaling and smoothness; `SL_LINEAR` was sold as "better scaling
  than FDM" and never as the right pairing for a degenerate problem.
- Both capability fixtures state their `sigma`. `_smoke_problem` takes the `sigma = 0` default while
  `_smoke_problem_2d` states `sigma=0.4`, so **the 1-D and 2-D cells are not the same experiment** —
  comparing their statuses varies scheme, dimension and regime at once, and regime is the axis the
  matrix does not have.

**Nothing is enforced and no default changed.** `sigma = 0` is documented as "the legitimate
deterministic sentinel" and is reconstruction-safe; the fixture is not buggy. Per the library's
division, whether a configuration *has* a property is measured and attributed, not refused — only
"I cannot answer this" is refused.

Recorded because it was the reason for a narrower change: putting `sigma` in each cell's baselined
artifact would drop the `intended` note of all nine non-PASS cells, since carry-forward compares the
whole artifact. It went in the fixture docstrings instead. Appending to the warning turned out to be
artifact-neutral — measured, `library_said` folds and truncates at 126 characters, so the new clause
never reaches it and the baseline is byte-unchanged. (#1878)
