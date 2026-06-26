# Changelog

All notable changes to MFGArchon will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **BC-capability gate: solvers fail loud on an unsupported boundary-condition type** (Issue #1456,
  first increment). The `BoundaryCapable` protocol (`supported_bc_types` + `_validate_bc_support`)
  existed but was un-adopted (1/12 solvers declared it, 0 enforced), which is why a BC type a solver
  cannot honor was silently collapsed to its default (usually Neumann) and solved the wrong problem.
  Added the missing `BaseMFGSolver._validate_bc_support` (raises `NotImplementedError` on an
  unsupported `BCType` at construction) and wired it into three template solvers with honest,
  audit-derived `_SUPPORTED_BC_TYPES`: `HJBGFDMSolver` (declared but never enforced → now enforced;
  added the adjoint-consistent `ROBIN`), `FPParticleSolver` (codifies its existing fail-fast; supports
  reflect/wrap **and** Dirichlet=absorbing), and `FPSLSolver` (previously collapsed Dirichlet/Robin to
  its zero-flux Neumann stencil → now raises). Byte-identical for every supported BC; only a
  genuinely-unsupported BC (which was silently mis-solved) now raises. Remaining solvers are migrated
  in follow-ups per #1456.

- **`HJBWENOSolver` boundary-condition resolution single-sourced** (Issue #1429, S0-21). Replaced a
  private 4-accessor copy of the BC resolution chain (which also diverged at the terminal — private
  `neumann_bc` vs the inherited `None` vs the ConditionsMixin `periodic_bc`) with a delegation to the
  inherited single source `BaseMFGSolver.get_boundary_conditions()` (the Issue #634 pattern already
  applied to the SL solver), keeping WENO's no-flux fallback for its concrete-BC ghost-buffer
  requirement. Byte-identical for real problems (`self._boundary_conditions` is never set, and the
  geometry/problem accessors resolve to the same stored BC); a convention-agreement pin was added.

- **`CoefficientField` diffusion default-value precedence fixed: `override → volatility_field → sigma`**
  (Issue #1412). The FDM/HJB solver sites (`hjb_fdm`, `base_hjb`, `fp_fdm_time_stepping`) hand-built
  `CoefficientField(override, problem.sigma)`, so a `None` per-solve `volatility_field` fell back to the
  **derived scalar** `problem.sigma` (`mean(array)`, or `1.0` for a callable) — silently dropping a
  spatially-varying `volatility_field`. They now route through the `MFGProblem.get_diffusion_coefficient_field(override=…)`
  factory (extended with `override` / `field_name` / `dimension`), whose precedence prefers the full
  `volatility_field` over the scalar placeholder. Byte-identical for all scalar-σ problems (every current
  golden) and for the #1248-forwarded `solve()` path (override non-None); corrects only the direct-call
  `None`-override edge on array/callable-σ problems. New precedence pins in `test_mfg_problem`.
- **σ-value lookup single-sourced via `pde_coefficients.resolve_diffusion_source`** (Issue #1412,
  generalizing #1071 to the diffusion coefficient). The per-solve `volatility_field` override
  resolution (scalar / per-point array / callable → scalar; batch path = array mean, callable at the
  domain center) is now one shared function; `HJBGFDMSolver._resolve_diffusion_source` delegates to it
  (byte-identical, pinned by `TestResolveDiffusionSource`'s convention-agreement test). A 6-solver FP
  audit confirmed the cross-path HJB/FP diffusion divergence the issue targeted is **already closed**
  (HJB-GFDM honors the override via #1316; SL/WENO and the scalar-only FP solvers fail loud; FP-FDM
  honors it via #1248) — this is a consolidation, not a behavior change. The σ-structure axis
  (constant `D` vs spatially-varying `σ(x)`) is orthogonal to spatial dimension: scalar-only solvers
  are constant-coefficient in both 1D and nD, and **fail loud** on a spatial override rather than
  silently mean-collapsing it (which would re-open the divergence). New `test_issue_1412_fp_volatility_fail_loud`
  pins those previously-unpinned GFDM / SL / SL-adjoint guards so a future "route through the shared
  resolver" refactor cannot silently drop them.
- **`FPParticleSolver` per-solve volatility no longer mutates the shared `problem.sigma`** (Issue
  #1412). The grid-drift paths previously applied a custom `volatility_field` by monkeypatching
  `self.problem.sigma = effective_sigma` before dispatch and restoring it in a `finally` (#1248) —
  a shared-state / re-entrancy hazard. It is now a solver-local `_effective_sigma_override` attribute
  (the #1316 pattern), resolved in `_get_grid_params` through `resolve_diffusion_source`. Byte-identical
  (a seeded `volatility_field=s` solve equals a `problem.sigma=s` solve to 1e-10); `problem.sigma` is
  never written. New `test_issue_1412_fp_particle_sigma_override`.

### Fixed

- **LLF effective volatility now tracks the per-solve `volatility_field` override** (Issue #1429,
  S0-13). `HJBGFDMSolver._llf_sigma_eff` was frozen at `__init__` from `problem.sigma`, so an
  LLF-augmented solve (#1059) with a #1316 per-solve volatility override stabilized off the base σ,
  not the override. `solve_hjb_system` now recomputes `_llf_sigma_eff` from the installed override
  (unconditionally, so a later `volatility_field=None` solve resets it). Byte-identical when LLF is
  off (the default) or no override is passed; idempotent across Picard iterations (does not reopen
  the #1059 frozen-ν stability note).
- **`FPNetworkSolver` treats a scalar `volatility_field` as SDE volatility σ (D = σ²/2), not as D**
  (Issue #1429, S0-15). The float branch consumed `volatility_field` directly as the diffusion `D`,
  diverging from the `base_fp` contract used by FDM/FVM/GFDM (where `volatility_field = σ`). It now
  routes through the single source `diffusion_from_volatility` (`D = σ²/2`). Affects only an explicit
  scalar `volatility_field` passed to the network FP solve; the `None` path (D-valued
  `diffusion_coefficient` knob) is unchanged.
- **Dead solver knobs now fail loud instead of silently doing nothing** (Issue #1426, all of
  S0-23–S0-27). Knobs accepted, documented, and stored on `self` but **never read** — a non-default
  value was a silent no-op (worse than an error) — now raise `NotImplementedError` on a non-default
  value, with defaults unchanged so existing usage is unaffected:
  `FPGFDMSolver(boundary_indices=…/domain_bounds=…)` (S0-26), `FPSLJacobianSolver(characteristic_solver=…)`
  (S0-27), `FPNetworkSolver(max_iterations=…/tolerance=…)` (S0-25 — the implicit step is a direct
  `spsolve`, not iterative), and `NetworkPolicyIterationHJBSolver(policy_tolerance=…)` (S0-25 — policy
  iteration converges on policy stability, not a value tolerance). Joins the already-shipped S0-23/24
  (`congestion_mode` / `weno_m_parameter`). `NetworkHJBSolver.tolerance` / `max_policy_iterations` are
  live and unchanged.

- **Semi-Lagrangian HJB scheme corrected — was ~24% wrong even at λ=1** (Issue #1413, PR #1417).
  The H-based SL update `u^{n+1}(x − dt·∇u) − dt·H` combined the characteristic foot with a
  pointwise `−dt·H`, double-counting the kinetic term (~3×), and used a λ=1-only foot (`∇u`
  instead of `∂H/∂p`). The default SL solver was therefore ~24% off the analytic Hopf-Lax
  solution **even at λ=1** and non-convergent under refinement (FDM matches the analytic to
  0.6%). Issue #575 had corrected only the state-term `(V+f)` sign on a coupling-dominated case,
  which masked the kinetic error. Replaced with the consistent Lax-Oleinik form
  `u_at_foot + dt·(H(p) − 2·H(0))` (foot `x − dt·∂H/∂p`) in the shared `_sl_value_update` helper,
  applied to every H-based SL path (default 1D batch/per-point/nD, `_with_dt`, stochastic CS).
  Validated <1.2% vs the analytic Hopf-Lax and <0.5% vs FDM across {kinetic, +V(x), +f(m)} ×
  λ∈{0.5, 1, 2}; new `TestSLHJBConsistency` regression gate. No paper experiments use SL.

- **FP/HJB drift single-sourced from `control_cost`, not `coupling_coefficient`** (Issue #1420,
  gotcha G-017; PR #1441 + follow-up). The MFG optimal feedback `α* = -∇U/control_cost` was computed
  in the FP-FDM coupled path (and the strict-adjoint `build_advection_matrix` /
  `solve_fp_step_adjoint_mode` pair) from the independent `MFGProblem.coupling_coefficient` field
  (default 0.5) instead of the Hamiltonian's `control_cost`. With `coupling_coefficient ≠
  1/control_cost` the HJB and FP used inconsistent control costs and converged to the wrong fixed
  point (exp16 Tier-2 had the Towel equilibrium ~4-5× too wide). The drift coefficient is now
  single-sourced via `pde_coefficients.fp_drift_coefficient` (`1/control_cost` for a
  quadratic-MINIMIZE `SeparableHamiltonian`; falls back to `coupling_coefficient` for
  `QuadraticMFGHamiltonian` / non-Hamiltonian solves). Byte-identical when `coupling_coefficient ==
  1/control_cost`; the LQ-FDM coupled golden was regenerated for the corrected physics (validated
  end-to-end against exp16, `KL_tavg = 8.0e-3`).

- **Semi-Lagrangian FP drift now includes the `1/control_cost` factor** (Issue #1420, audit finding
  S0-03). `FPSLJacobianSolver` and `FPSLSolver`/`FPSLAdjointSolver` computed the drift as `α = -∇U`,
  dropping the `1/λ` factor entirely, so for `control_cost ≠ 1` the transported drift had the wrong
  magnitude (the HJB used `λ` while the FP advected `c_eff = 1`). The drift is now
  `α* = -∇U/control_cost` (and the Jacobian-SL divergence shortcut `div(α) = -c·ΔU` carries the same
  `c`), single-sourced via `pde_coefficients.fp_drift_coefficient`. Byte-identical when
  `control_cost == 1`; corrected otherwise. New `test_fp_sl_drift_control_cost_s003` regression gate.

- **Coupling layer fails loud instead of advecting the value function as a velocity** (Issue #1420
  V2). For a smooth-separable Hamiltonian, `resolve_fp_drift_kwargs` routed `U` to a velocity-only FP
  solver (one exposing `drift_field`=α* but no `potential_field`, e.g. `FPGFDMSolver`) as
  `drift_field=U` — silently advecting the value function as a velocity. Since such solvers are
  meshfree (the coupling layer cannot derive α* at their collocation points), this now raises a
  `ValueError` directing the caller to pass an explicit `drift_field=α*` or use a `potential_field`-
  capable solver. `FPGFDMSolver` declares `_drift_convention = VELOCITY` explicitly. No previously
  passing path is affected (the fp_gfdm tests use the explicit precomputed-drift path).

- **`fp_drift_coefficient` fails loud when it cannot source the drift coefficient** (Issue #1420 V1).
  For a problem with neither a quadratic-MINIMIZE `SeparableHamiltonian` (to source `1/control_cost`)
  nor a `coupling_coefficient` attribute — a duck-typed / malformed problem — the helper previously
  returned `1.0` silently (a wrong-temperature drift with no error). It now raises `ValueError`
  (CLAUDE.md "NO silent fallbacks"). Standard `MFGProblem`s (always carry `coupling_coefficient`) and
  quadratic-Hamiltonian solves are unaffected.

- **Canonical Carlini-Silva SL (`diffusion_method="canonical_cs"`) honors `control_cost`** (Issue
  #1420). The per-node DPP running cost was hardcoded `(1/2)|α|²` (λ=1), so the implicit-α* minimizer
  gave `α* = -∇u` instead of `-∇u/control_cost` — undercutting the solver's λ≠1 support. It is now
  `(λ/2)|α|²` (and the α search bound scales by `1/λ`), so the minimizer yields `α* = -∇u/λ` and the
  LQ value matches the analytic Riccati ratio `u(0)/u(T) = λ/(λ+1)`. Byte-identical at `control_cost
  == 1`. New `TestCanonicalCSControlCostLambda` covers λ∈{0.5,1,2} (sensitivity + Riccati ratio).

### Changed

- **Hamiltonian as single source of truth — solver-level physics re-derivation retired** (Issue
  #1071). The GFDM legacy-LQ vectorized residual was retired (#1407) and its Jacobian
  single-sourced through `assemble_hjb_jacobian_diag` → `H_class.evaluate_dp` (#1408,
  byte-identical; the test-only entry point renamed to the public `assemble_hjb_iteration_matrix`,
  #1414/#1418); the dead `_compute_hjb_jacobian`/`_analytic` pair removed (#1409); the
  semi-Lagrangian H value routed through the `eval_H_batch` shim and the dead
  `_find_optimal_control` dropped (#1410); and **Howard's policy-evaluation Lagrangian
  single-sourced via `control_cost.lagrangian()`, lifting the λ≠1 restriction** (#1416,
  byte-identical at λ=1; Howard-vs-Newton agreement validated for λ∈{0.5, 1, 2}). Each step is
  byte-identical-pinned. Remaining #1071 work is tracked in #1411 (paper-baseline migrations) and
  #1412 (σ-source unification).

- **Ruff updated v0.14.3 → v0.15.17** (pre-commit pin; the prior auto-update branches were stale).
  Resolved the lint surfaced by the version jump (no runtime behavior change): import-sorting (I001),
  comprehension/pytest/typing idioms (C4xx/PT/RUF/TC), and 2 typing-only-import moves into
  `TYPE_CHECKING` in source (`fp_particle`, `jax_backend` — behavior-preserving under
  `from __future__ import annotations`). **`UP042` (str-Enum → `StrEnum`) is now ignored**: it
  changes `str()`/format output (the value vs `"Class.MEMBER"`), which the codebase relies on as a
  tested contract (`test_*_string_representation`), so it is a behavior change, not a safe
  modernization. Test-only naming exceptions (`N801` descriptive test class, `N812` script brevity
  alias) carry targeted `noqa`. Pre-existing `tests/validation/test_duality_convergence.py::
  TestConvergenceRate::test_upwind_first_order_convergence` fails on main too (unrelated to this bump).

- **Backend `hjb_step`/`fpk_step` documented as LQ-only toy steppers** (Issue #1072, docs-only).
  `BaseBackend.hjb_step`/`fpk_step` and all four backend impls (numpy/torch/numba/jax) now state
  in their docstrings that these are experimental LQ-only steppers — each hardcodes a *different*
  default Hamiltonian and **none honors `problem.hamiltonian_class`** — with **no caller in the
  HJB/FP solver fleet** (the production solvers single-source the Hamiltonian via `base_hjb.evaluate_H`
  ← `problem.hamiltonian_class`, #1071). This removes the trap of mistaking #1072 ("Functional
  Operator Lowering", the deferred post-v1.0 RFC to XLA-lower the operator tree + Hamiltonian) for a
  quick patch on `jax_backend.py`. No runtime change. The World-A-vs-World-B design fork and the
  un-defer trigger are recorded on #1072.

- **GFDM Local-Lax-Friedrichs assembly single-sourced through the Layer-B helpers** (Issue #1071
  phase 7). The two batch-Hamiltonian GFDM paths each re-implemented the entire residual /
  Jacobian assembly inline solely to swap the scalar diffusion `σ` for the per-node LLF field
  `σ_eff` (Issue #1059). `assemble_hjb_residual` / `assemble_hjb_jacobian_diag` (`h_eval.py`) now
  accept a per-node `σ` field — residual: `D = σ²/2` elementwise; Jacobian: **row-scales** the
  Laplacian via `diags(D) @ D_lap` (not `D * D_lap`, which for an array does not row-scale) — so
  the GFDM LLF residual/Jacobian branches collapse into the single shared assembly path (scalar `σ`
  for a plain solve, `σ_eff` field when LLF is active). Scalar callers are **bit-identical** (the
  scalar code path is untouched); the field path is byte-identical to the inline LLF expressions it
  replaces, pinned by `test_assemble_hjb_residual_array_sigma_matches_inline_llf_1071` and
  `test_assemble_hjb_jacobian_diag_array_sigma_matches_inline_llf_1071`, with the GFDM LLF
  augmentation suite as the end-to-end check. (LLF stays solver-level — it modifies the diffusion
  coefficient, not `H`/`∂H/∂p` — so the `Regularizer`/`with_regularizer` scaffold is correctly left
  for the physics-only regularizers it was scoped for. The legacy-LQ-vectorized path inlines its own
  Hamiltonian and is out of scope.)

### Removed

- **`BoundHamiltonian` + `bind_cross_density` retired** (Issue #1071, increment 3 — completes the
  multi-population cross-coupling migration). With both the HJB (#1397) and FP (#1398) paths moved
  to the lock-faithful `cross_density` trajectory channel, the `BoundHamiltonian` wrapper and
  `HamiltonianBase.bind_cross_density` had no remaining callers and are deleted, along with the now
  dead bound-H plumbing: the `active_hamiltonian` parameter throughout `base_hjb` (residual,
  Jacobian, Newton step, timestep, backward sweep) and the `hamiltonian_override` keyword on
  `HJBFDMSolver.solve_hjb_system`. The fail-loud guards on `HJBHowardSolver`/`WeakFormHJBSolver`
  and the `MultiPopulationIterator` gate now key off the live channel: `hamiltonian_override` →
  `cross_density`, and `_honors_multipop_hamiltonian_override` → `_honors_multipop_cross_density`
  (this also closes the latent `**kwargs` silent-swallow on `WeakFormHJBSolver`). The
  `resolve_fp_drift_kwargs` `_inner`-unwrap is simplified away (no wrapper left to unwrap). All
  internal/[PROVISIONAL] APIs — no public surface affected. Behavior-preserving: the `cross_density`
  channel was proven byte-identical to the wrapper in #1397/#1398; 911 HJB/coupling + multipop
  tests pass. The wrapper-specific and byte-identity pinning tests are retired (purpose served); the
  FP velocity test is kept as a cross-density *flow* test.

### Changed

- **Multi-population FP drift: lock-faithful `cross_density` channel** (Issue #1071, increment 2
  of the `BoundHamiltonian` retirement). `compute_fp_velocity_field` and `resolve_fp_drift_kwargs`
  gain a `cross_density=` parameter (the stacked `(Nt+1, K*Nx)` trajectory); when given, the
  non-smooth FP velocity path feeds `optimal_control` the stacked density at each integer timestep
  `n` (`cross_density[n]`, sliced per-population via `population_index`) instead of the wrapper's
  `m_all[round(t/dt)]`. `MultiPopulationIterator`'s FP step now passes the population's own
  (unbound) Hamiltonian + `cross_density=m_all` (unconditionally, matching the old unconditional
  `bind_cross_density`) — so smooth-separable H keeps the `potential_field=U` dispatch and the
  whole path is **byte-identical** to the bound-H path. This removes the **last**
  `bind_cross_density` call site; `BoundHamiltonian` is now unused (deleted in increment 3).
  Byte-identity to the bound-H path was verified during migration; once the bound path was deleted
  (increment 3) the wrapper-comparison pin was retired, and the FP cross-density consumption is now
  pinned by `test_fp_velocity_consumes_cross_density_1071` (flow assertion with an m-dependent H).

- **Multi-population HJB cross-coupling: lock-faithful `cross_density` channel** (Issue #1071,
  increment 1 of the `BoundHamiltonian` retirement). `HJBFDMSolver.solve_hjb_system` gains a
  `cross_density=` parameter taking the stacked `(Nt+1, K*Nx)` cross-density trajectory directly;
  the backward loop indexes it at each integer timestep `n_idx_hjb` (where
  `current_time = n_idx_hjb · dt`) and feeds the population's *own* Hamiltonian, which slices the
  other populations via `population_index`. This replaces the `BoundHamiltonian` wrapper's two
  smells — the dead `m` argument and the reverse-engineered `round(t/dt)` row-pick — while keeping
  `HEvalState` physics-only (no `dt` on the state). `MultiPopulationIterator`'s HJB step now uses
  this channel; the FP drift path still uses `bind_cross_density` (migrated in the next increment).
  The bound-H `hamiltonian_override` channel is retained until the wrapper is deleted, and the two
  are mutually exclusive (fail loud). **Byte-identical** to the bound-H path (verified during
  migration; the wrapper-comparison pin was retired in increment 3 when the bound path was deleted).
  Cross-coupling is pinned by `test_hjb_sees_cross_density_bug_1157`, and the granular H-evaluation
  byte-identity by `tests/unit/test_alg/test_hamiltonian_single_source_1071.py`.

### Added

- **Fail-fast ratchet in CI** (Issue #1071). `scripts/check_fail_fast.py` gains
  `--write-baseline` / `--check-baseline` (ratchet) modes, and Quick Validation now runs
  `--check-baseline scripts/fail_fast_baseline.json` against `mfgarchon/`. It fails the
  build only when a category (broad/bare `except`, silent `pass`-in-except, `hasattr`)
  **increases** vs the committed baseline — so the silent-fallback cleanup can't regress,
  while the count is free to ratchet down (regenerate the baseline with `--write-baseline`
  after fixing violations). Baseline at introduction: `hasattr=174, silent_pass=70,
  bare_except=0, broad_except=11`. Physics-style fallbacks (getattr-default, hardcoded
  defaults) are intentionally NOT regex-gated (too many legitimate defaults — 160 seen in
  the scan); those stay covered by periodic judgment-based scans + per-site fixes.

### Changed

- **`QPSolver` warns on non-convergence instead of silently returning the unconstrained
  solution** (Issue #1071). All three backends (OSQP / scipy SLSQP / L-BFGS-B) returned the
  unconstrained least-squares `x0` (incrementing a `failures` stat) when the constrained solve
  did not converge — with no error or warning, so a constraint-violating result was used as if
  it had succeeded. (Distinct from the *exception* path, which `monotonicity_enforcer` already
  warns + falls back on.) For the GFDM monotone-stencil use, that silently produces a
  non-monotone stencil. The shared `_unconstrained_fallback` helper now emits a `logger.warning`
  (constraints not enforced); the `x0` fallback is kept (robustness). Byte-identical for
  converged solves. Regression: `tests/unit/test_utils/test_qp_utils.py::TestQPNonConvergenceWarns1071`.

- **Degraded fallbacks now warn instead of staying silent** (Issue #1071). Two legitimate
  fallbacks that previously masked a real degeneracy are kept but surfaced: `HJBGFDMSolver`'s
  per-point FD Jacobian uses an identity row when a stencil is degenerate (singular Taylor
  matrix) — now emits one aggregated warning naming the affected collocation points (Newton
  convergence is degraded there); and `MeasureField`'s 1D/nD KDE falls back to a fixed
  bandwidth `0.1` on a zero-spread (degenerate) particle cloud — now warns that the density
  estimate is unreliable. Behavior is unchanged (byte-identical for non-degenerate inputs).
  Regression: `tests/unit/test_core/test_measure.py::TestDegenerateCloudWarns1071`.

- **`GeneralMFGFactory._load_function` fails loud on a provided-but-unloadable spec**
  (Issue #1071, fail-fast). A function spec that failed to load — a `lambda` that won't
  evaluate, an unimportable `module.func` path, or an unresolvable name — previously
  logged and returned `None`, silently dropping the user's function (defaulted for
  optional components like `m_initial`/`potential`, or surfaced downstream as a misleading
  "u_terminal required"). All three now raise a clear `ValueError` naming the spec and the
  underlying error. `func_spec=None` (not provided) still returns `None` — the legitimate
  case. Byte-identical for valid specs (factory suite green). Regression:
  `tests/unit/test_factory/test_general_mfg_factory.py::TestLoadFunctionFailLoud1071`.

- **`FPParticleSolver` fails loud on an invalid initial density** (Issue #1071, fail-fast).
  When `m_initial` produced invalid sampling probabilities — NaN/Inf (which routed silently to
  the uniform `else` since `NaN > 1e-9` is False) or negative entries (which made
  `np.random.choice` raise) — the solver silently sampled the initial particles from a
  **uniform** distribution instead of the specified density, solving a different problem with no
  error. Both invalid cases now raise a clear `ValueError`; the documented
  degenerate-but-finite (`sum ≈ 0`) → uniform default is unchanged. No test relied on the
  fallback (all use finite, non-negative densities) so this is byte-identical for valid usage.
  Regression: `tests/unit/test_alg/test_fp_particle_solver.py::TestFPParticleInvalidMInitial1071`.

- **`LaplacianOperator` fails loud on an unhandled / unparseable boundary condition**
  (Issue #1071, fail-fast). Two silent-wrong fallbacks (surfaced by the silent-fallback scan)
  are now errors: (a) an **unhandled `bc_type`** (e.g. Robin) previously emitted a
  boundary-diffusion-free *interior-only* stencil (a silently under-constrained boundary) →
  now raises `NotImplementedError` naming the unsupported type and the supported set; (b) a
  **provided `bc` whose `bc_type` could not be determined** previously degraded silently to
  the periodic default → now raises `ValueError`. The documented **`bc=None` → periodic**
  default is unchanged. No tested path hit either fallback (436-test FP/laplacian sweep green),
  so this is byte-identical for real usage. Regression:
  `tests/unit/test_operators/test_laplacian.py::TestLaplacianBCFailLoud1071`.

- **HJB semi-Lagrangian and WENO now fail loud on a missing Hamiltonian** (Issue #1071,
  fail-fast). Both solvers carried silent fallbacks that substituted a hardcoded LQ
  Hamiltonian when no `hamiltonian_class` / `problem.H` was available — returning a
  plausible-but-wrong solution for any non-LQ problem with no error:
  - SL `_default_hamiltonian` (`H = 0.5*|p|^2 + C*m`) on the per-point path,
  - SL **batch path** silently zeroing `H` (`H_values = np.zeros(Nx)`) — reducing the HJB
    update to pure transport of the terminal data (caught by a comprehensive scan; the
    per-point fix alone was incomplete and the per-point loops' broad `except` would have
    swallowed its raise),
  - WENO `0.5*grad**2 + m*grad`.

  Fixed with a **solve-entry guard** in `HJBSemiLagrangianSolver.solve_hjb_system` (fails
  loud before any path can produce a silent pure-transport solution), a batch-path backstop,
  and the WENO raise; the dead `_default_hamiltonian` method is removed. These paths were
  dead for any normally-constructed `MFGProblem` (construction requires a Hamiltonian), so
  the change is **byte-identical for real usage**; they now raise a clear `ValueError` naming
  the fix instead of silently solving the wrong physics. Regression (incl. the real solve
  path): `tests/unit/test_alg/test_1071_fail_loud_missing_hamiltonian.py`.

### Removed

- **Vestigial `OmegaConfManager.create_pydantic_config` / `_map_omega_to_pydantic`**
  (Issue #1392). A pre-North-Star second OmegaConf→Pydantic bridge that bypassed the
  canonical single crossing (`config.bridge.bridge_to_pydantic`) and silently returned a
  **default** `MFGSolverConfig`: `_map_omega_to_pydantic` mapped flat keys
  (`max_iterations`/`tolerance`/`damping`) while `MFGSolverConfig` is nested
  (`hjb`/`fp`/`picard`/…), so Pydantic (`extra=ignore`) dropped them — the user's config was
  silently discarded. No callers. Per the Config System North Star (Pydantic = single source
  of truth, OmegaConf = transport only, one validation crossing), this was residue. Use
  `bridge_to_pydantic(omega_cfg, MFGSolverConfig)`; `OmegaConfManager`'s load/merge/save
  transport is unchanged. Regression guard:
  `tests/unit/test_config/test_bridge.py::TestNoVestigialOmegaToPydanticBridge1392`.

## [0.20.4] - 2026-06-16

### Fixed

- **HJB source term receives the value function on all couplers** (Issue #1382).
  `source_composition.compose_hjb_source` (the Picard / coupled-Newton couplers)
  passed `v = 0` to `source_term_hjb(x, m, v, t)`, while `graph_mfg_solver` passed
  the real value-function slice `v_t` — so the same problem-level source yielded a
  different result on graph vs grid for any `v`-dependent HJB source (the #1259/#1285
  silent-divergence class). The documented contract (`mfg_problem.py`:
  `Callable(x, m, v, t)`) and the FP source both bind `v_t`; the grid couplers now do
  too. Both couplers build the source + nonlocal terms through a single
  `_problem_hjb_source_terms` primitive, so the fork cannot re-open silently.
  Byte-identical for the existing (all `v`-independent) sources and obstacle/nonlocal
  ordering preserved; the change only affects a future `v`-dependent HJB source.
  Pinned by `test_issue1361_source_composition.py::test_hjb_source_receives_value_function_slice`.

- **1D HJB boundary gradient now BC-aware on the default NumPy path — residual AND Jacobian**
  (Issue #1384). The default single-population `HJBFDMSolver` carries `backend=NumPyBackend`
  (≠ `None`), so both the residual's Hamiltonian momentum `p = ∂u/∂x` and the per-point FD
  Jacobian fell to the legacy per-point stencil, which uses periodic `% Nx` wraparound at the
  boundary — a wrong boundary momentum for Dirichlet/Neumann/no-flux problems, regardless of
  the actual BC. (The Issue #542 diffusion fix made the *Laplacian* BC-aware for this path,
  but the *gradient* was gated more strictly behind `backend is None`.) Both the precomputed
  residual gradient and the FD-Jacobian momentum stencil are now BC-aware for any non-torch
  (NumPy-like) array, mirroring the BC-aware Laplacian gate. **Both had to move together**:
  making only the residual BC-aware while the Jacobian stayed periodic is a severe
  `J ≠ ∂F/∂U` mismatch at the boundary that makes Newton *diverge* for Dirichlet BC with
  steep terminals. This stays finite-difference; the analytic-Jacobian swap is a separate
  question under #1380. Verified: the Dirichlet steep-terminal case that diverged with a
  residual-only fix now converges; for a no-flux wall the fix feeds momentum `0.0`
  (BC-respecting) where the old path fed a spurious `+2.69`; **periodic BC is byte-identical**
  (Δ=0); the full suite is green. Only asymmetric non-periodic 1D FDM solves shift (toward
  correct). Validated in `scripts/validation/hjb_1d_bc_gradient.py`; regression-tested in
  `tests/unit/test_alg/test_hjb_fdm_solver.py::TestBoundaryGradientBCAware1384`.
- **Post-audit hygiene** (session quality audit). Three low-severity consistency
  fixes plus a CHANGELOG cleanup:
  - `GeneralMFGFactory.create_from_hamiltonian` now raises a clear `ValueError`
    naming the missing `Nx` when a legacy 1D `domain_config` supplies
    `xmin`/`xmax`/`Lx` but omits `Nx`, instead of a bare `KeyError` (Issue #1363).
  - Renamed the two `solve_hjb_system` holdouts (`BaseHJBSolver` abstract
    signature, `PenaltyHJBSolver`) from the removed parameter names
    `M_density_evolution_from_FP` / `U_final_condition_at_T` / `U_from_prev_picard`
    to the canonical `M_density` / `U_terminal` / `U_coupling_prev`, matching every
    concrete solver and the [0.20.1] removal (Issue #1355). The penalty solver
    forwards positionally, so the inner solve is unchanged.
  - `create_lions_source` / `create_nonlocal_source` now raise `ValueError` on a
    2-D `(Nt+1, Nx)` density instead of silently using the terminal slice `m[-1]`;
    the composed source pipeline time-slices before calling, so a per-time source
    must receive a 1-D spatial slice (a 2-D array is a caller error that
    reintroduced the Issue #1285 wrong-slice behavior).
  - Merged a duplicate `### Fixed` subsection under `[0.20.0]` into one block
    (Keep a Changelog compliance).

## [0.20.3] - 2026-06-15

### Added

- **Hamiltonian single-source contract** (PR #1377, Refs #1071 — pilot, Phase 0+1). New `HEvalState`,
  `HamiltonianValues`, and a `Regularizer` protocol in `core/hamiltonian.py`; `HamiltonianBase` gains
  granular primitives `evaluate_H` / `evaluate_dp` (the residual/Jacobian hot path), a separate
  `dH_dm`, a thin `evaluate()` convenience, and a physics-only `with_regularizer()` scaffold.
  `base_hjb`'s residual + analytic Jacobian now consume the primitives (byte-identical, atol=0), and
  `h_eval.eval_H_batch` / `eval_dH_dp_batch` delegate to them (single source — no third layer). First
  phase of consolidating per-solver physics re-derivation; later phases migrate WENO/SL/GFDM. (JAX
  lowering #1072 is a separate sub-RFC — not delivered here.)

### Removed (BREAKING)

- **`mfgarchon.core.HamiltonianState`** (PR #1377) — an unused public export (zero usages anywhere in
  the package, tests, or research); superseded by `HEvalState` (reshaped, so no compatible alias was
  possible). Use `HEvalState(x, p, m, t)`.
- **Legacy 1D-geometry `MFGProblem` construct/write surfaces** (PR #1375, Refs #1363) — completes the
  geometry-first migration begun in #1360 (Tier-3a). Removed the `xmin=`/`xmax=`/`Nx=`/`Lx=`
  **constructor** kwargs (deprecated v0.17.1), the `_init_1d_legacy`/`_init_nd` manual grid-construction
  methods (v0.17.0), and the `get_u_final`/`get_u_fin` value-function aliases (v0.17.6). Construct 1D
  problems geometry-first: `MFGProblem(geometry=TensorProductGrid(bounds=[(xmin, xmax)],
  Nx_points=[Nx + 1], boundary_conditions=no_flux_bc(dimension=1)), T=…, Nt=…, sigma=…)` — note `Nx`
  counts **intervals**, so `Nx_points = Nx + 1`. Because `__init__` takes `**kwargs`, the removed
  kwargs now raise `ValueError` (pointing at the geometry-first API) rather than being silently
  swallowed; the removed methods/aliases raise `AttributeError`. The `spatial_bounds=` n-D path and the
  no-arg default are unaffected. ~99 construction sites migrated; the Picard / fictitious-play couplers
  now read terminal data via `get_u_terminal()` (which also removed a latent silent-`zeros`
  terminal-condition fallback). With this, the legacy 1D-geometry API is fully removed (Tier-1→3c).

## [0.20.2] - 2026-06-15

### Added

- **Coupled-Newton coupler solves source/nonlocal/obstacle problems** (PR #1372, Refs #1361). The
  `MFGResidual` / `NewtonMFGSolver` path now solves MFG problems with `source_term_hjb`,
  `source_term_fp`, `nonlocal_operator`, or `obstacle` (previously it raised). The source is composed
  from the `(U, M)` residual arguments so the finite-difference Jacobian differentiates through it,
  and the Newton path converges to the same equilibrium as the Picard `FixedPointIterator` (verified
  to ~1e-10). (refs #1259, #1285, #924)
- **GFDM `joint_socp`: SOCP-infeasibility-triggered adaptive stencil enlargement** (PR #1370, Refs
  #1106). When a stencil is still infeasible after C-bisection, `PrecomputedJointSocpStencils` adds
  next-nearest neighbors (Taylor degrees of freedom) and retries the SOCP, capped by
  `HJBGFDMSolver(socp_max_stencil_enlargements=…, socp_enlargement_step=…)`. This recovers
  geometrically-infeasible wall/corner/obstacle-adjacent stencils that C-bisection and penalty
  pressure cannot. Enlargement candidates respect obstacle visibility (`filter_visible_neighbors`),
  so it never re-adds cross-wall neighbors the base filter removed. Default OFF (`0`) → paper/default
  path is byte-identical; the enlarged set is consumed via the SOCP single-source `neighbor_indices`
  contract on all HJB-GFDM assembly paths (no operator/FP/BC cascade). Documented hook for the
  #1107 (3rd-order Taylor) / #1108 (M-matrix-only) exact fallbacks.
- **Opt-in 2nd-order one-sided FDM boundary stencils** (PR #1368, Refs #1084). `scheme="one_sided"`
  on the finite-difference gradient produces genuinely O(h²) boundary derivatives (verified EOC
  ≈ 2.13); the default `scheme="central"` path is unchanged, so no existing solver path shifts.

### Changed

- **Source/nonlocal/obstacle composition is single-sourced** (PR #1372, Refs #1361). The composition
  logic now lives in one module, `coupling/source_composition.py` (`compose_hjb_source` /
  `compose_fp_source`), consumed by both `FixedPointIterator` (now thin delegates) and `MFGResidual`,
  with a byte-equality pinning test — closing the parallel-private-copy class behind #1259/#1285. The
  `MFGResidual` #1285 fail-loud guard is removed (the path now solves these problems). Obstacle uses
  the same approximate `v=0` penalty in both couplers (`PenaltyHJBSolver` #924 remains the proper route).
- **JAX backend warns on high-order-scheme downgrade** (PR #1367, Refs #1072). Selecting the JAX
  backend with a high-order scheme (WENO/upwind/…) now emits a one-time warning — JAX implements
  only 2nd-order central differences, so results differ from the NumPy high-order path. Interim
  guard (no behavior change to the solve); full JAX high-order support remains open in #1072.
- **Honest accuracy claims for GFDM/FDM/WENO** (PR #1368, Refs #1084). The GFDM ill-conditioning
  threshold (`COND_THRESHOLD=1e12`) is documented as warn-only (≈4 reliable float64 digits at the
  cap; pass a stricter `cond_threshold` for tighter control), and the WENO5 `c_minus` docstring was
  corrected to match the implemented reversed-stencil window. No numerical change.

## [0.20.1] - 2026-06-15

### Changed

- **Vectorized the semi-Lagrangian canonical Carlini-Silva per-point optimization** (PR #1353).
  `HJBSemiLagrangianSolver._canonical_cs_step` (`diffusion_method="canonical_cs"`, Issue #1058) in
  1D previously minimized the per-node DPP objective `phi(alpha)` with a Python loop calling
  `scipy.optimize.minimize_scalar` (Brent) once per grid node. The loop is replaced by a single
  fixed-iteration **vectorized golden-section search** that solves all nodes' independent 1D
  bounded minimizations simultaneously via array ops (one batched interpolation per iteration
  instead of `Nx` separate scalar solves). Iteration count is set to reach the same bracket
  tolerance (`self.tolerance`) as the Brent call it replaces. This changes the optimizer (not
  byte-identical), but accuracy is preserved: the canonical-CS gates pass unchanged and the
  Hopf-Lax analytic L2 error is identical to 7 significant figures. Measured 1D solve speedup of
  13× (Nx=101) to 40× (Nx=401), growing with grid size. nD (vector control, L-BFGS-B per node) is
  unchanged.

- **CI: blocking mypy gate on `mfgarchon/config` + parallel test runs** (PR #1358). Added a
  blocking CI step running `mypy mfgarchon/config --follow-imports=silent` (0 errors — the
  `config` subpackage is now type-clean and gated against regressions). Enabled `pytest-xdist`
  (`-n auto`) for the PR test suite after confirming it reproduces the serial result exactly
  (no test-isolation failures); `pytest-xdist>=3.5` added to the `dev` extra.

### Removed (BREAKING)

- **Removed the deprecated legacy 1D-geometry read-properties on `MFGProblem`** (PR #1360, Tier-3a
  of the geometry-first unification, ADR #417 / #435). The computed properties `xmin`, `xmax`, `Lx`,
  `Nx`, `dx`, `xSpace`, `_grid` (getters + setters, deprecated since v0.17.0) and their `_*_override`
  backing slots are removed; accessing them now raises `AttributeError`. Use the geometry-first API:
  `problem.geometry.get_bounds()[0][0]`/`[1][0]` (xmin/xmax), `get_grid_spacing()[0]` (dx),
  `get_spatial_grid()` (xSpace), `num_spatial_points - 1` (Nx, the number of intervals),
  `problem.geometry` (_grid). The unused `xmin`/`xmax`/`Nx`/`dx`/`xSpace` declarations on the
  `GridProblem`/`DirectAccessProblem` typing Protocols are also removed. The legacy
  `MFGProblem(xmin=, xmax=, Nx=, Lx=)` **constructor** parameters are retained (still deprecated) —
  removing them (Tier-3b) requires migrating ~70 active construction sites and is tracked in #1363.
- **Lowercase grid parameters removed** (deprecated v0.17.0, past 3-minor-version
  window at v0.20.0). `MFGSystemBuilder.domain()` no longer accepts `nx`/`nt` and
  `SparseMatrixOptimizer.create_laplacian_3d()` no longer accepts `nx`/`ny`/`nz`.
  Use the capitalized `Nx`/`Ny`/`Nz`/`Nt` names. Passing the old names now raises
  `TypeError`.
- **Mesh-IO `format_type` parameter removed** (deprecated v0.17.12, past window at
  v0.20.0). `Mesh1D.export_mesh()` and `Mesh3D.export_mesh()` no longer accept
  `format_type`; use `file_format`. Passing the old name now raises `TypeError`.
- **FP-particle solver legacy parameters removed** (deprecated v0.17.0, past window
  at v0.20.0). `FPParticleSolver.__init__` no longer accepts `mode` (use
  `density_mode`), `external_particles` (use `num_particles`), or
  `normalize_kde_output`/`normalize_only_initial` (use `kde_normalization`).
  Passing the old names now raises `TypeError`. The already-broken
  `particle_collocation_dual_mode_demo.py` example (relied on the removed
  collocation mode and the absent `ParticleMode` symbol) was deleted.
- **Removed Tier-2 HJB-solver parameter deprecations (≤ v0.17)** (PR #1355). The legacy keyword
  aliases on the HJB solvers' `solve_hjb_system` methods are gone; pass the canonical names:
  `M_density_evolution` / `M_density_evolution_from_FP` → `M_density`,
  `U_final_condition` / `U_final_condition_at_T` → `U_terminal`,
  `U_from_prev_picard` → `U_coupling_prev`. Affects `HJBFDMSolver`, `HJBGFDMSolver`,
  `HJBSemiLagrangianSolver`, `HJBWenoSolver`, and the network solvers (`NetworkHJBSolver`,
  `NetworkPolicyIterationHJBSolver`). The deprecated `bc_values=` kwarg (adjoint-consistent BC is
  handled via `BCValueProvider` in `BoundaryConditions`) is removed from
  `HJBFDMSolver.solve_hjb_system` and from the internal `_compute_laplacian_1d` helper. The
  deprecated Newton-parameter aliases `NiterNewton` → `max_newton_iterations` and
  `l2errBoundNewton` → `newton_tolerance` are removed from `HJBFDMSolver.__init__` (since v0.16) and
  from the `base_hjb` module functions `solve_hjb_timestep_newton` / `solve_hjb_system_backward`
  (since v0.17). Passing any removed name now raises `TypeError`. The v0.18+ deprecations
  (`damping_factor` → `relaxation`, since v0.19.2; `tensor_volatility_field` → `volatility_field`,
  since v0.18.7) are retained.
- **Removed the Tier-2 FP-solver parameter renames `m_initial_condition` and `diffusion_field`** (PR #1356)
  (deprecated v0.17.0, removed at v0.20 — 3 minor versions past the deprecation window). On the FP
  solvers' `solve_fp_system` (`FPFDMSolver`, `FPParticleSolver`, `FPGFDMSolver`, `FPSLSolver`,
  `FPSLAdjointSolver`) and the `BaseFPSolver` abstract interface, use `M_initial` instead of
  `m_initial_condition` and `volatility_field` instead of `diffusion_field`; passing an old name
  now raises `TypeError`. (`FPGFDMSolver` only deprecated `diffusion_field` — `m_initial_condition`
  is its current first positional parameter and is unchanged.) Internal helpers
  (`solve_fp_nd_full_system`, `FPFDMSolver._solve_fp_1d`) keep `m_initial_condition` /
  `diffusion_field` as their own non-deprecated parameter names. The `tensor_diffusion_field` and
  `volatility_matrix` aliases (also v0.17.0) are intentionally retained: `volatility_field` has no
  equivalent yet for their callable-tensor routing, so they are not removal-ready. The
  `MFGProblem.diffusion_field` property and the `FPNetworkSolver` `m_initial_condition` deprecation
  are out of scope.
- **Removed Tier-2 monitoring-family deprecations** (PR #1357). The deprecated boolean kwargs
  `monitor_convergence` / `auto_progress` / `timing` on the `enhanced_solver_method` decorator are
  removed — pass an `options=` flag instead, e.g.
  `options=SolverMonitoringOptions.CONVERGENCE | SolverMonitoringOptions.PROGRESS`. The deprecated
  convergence-monitor factories `create_default_monitor` / `create_stochastic_monitor` are removed;
  use `create_rolling_monitor` / `create_distribution_monitor`. Passing the old names now raises.

## [0.20.0] - 2026-06-14

### Added

- **Spatial interaction operators for non-local MFG coupling** (Issue #1023, PR #1348). New
  `mfgarchon/operators/interaction/`: a `RadialKernel` ABC with `GaussianKernel` / `TentKernel` /
  `WendlandKernel` (compact-support C²) / `DipoleKernel` / `PowerLawKernel`; a
  `ConvolutionCouplingOperator` computing `(K * m)(x)` via FFT (`scipy.signal.fftconvolve`,
  O(N log N)) with a direct-quadrature O(N²) fallback for validation (1D + 2D, periodic and
  non-periodic); and an `EnergyFunctional` protocol with `QuadraticInteractionEnergy` /
  `PotentialEnergy` / `CombinedEnergy`, each exposing an analytic `lions_derivative()`
  (the functional derivative `δℱ/δm = K * m`). `create_lions_source` recognizes an
  `EnergyFunctional` and uses its analytic derivative as the HJB source. Verified gates:
  FFT-vs-direct agreement 1.7e-16; `lions_derivative` vs finite-difference `δℱ/δm` 2.4e-9;
  Lions-bridge identity 2.2e-16; ring-equilibrium center depletion 2.3× under repulsive
  non-local coupling.

- **Hexahedral FEM element family** (Issue #470, PR #1347). The FEM `mesh_adapter` round-trips
  `element_type='hexahedron'` → skfem `MeshHex`, and `assembly` adds `(MeshHex, 1) → ElementHex1`
  and `(MeshHex, 2) → ElementHex2`, so a 3D hexahedral mesh (built gmsh-free via
  `MeshHex.init_tensor`) runs the FEM solve path at P1 and P2. gmsh-based mesh *generation*
  remains separately blocked (gmsh is not an installed dependency).

- **Conservative Finite Volume Method (FVM) Fokker-Planck solver** (Issue #422). New
  `FPFVMSolver` (`mfgarchon/alg/numerical/fp_solvers/fp_fvm.py` + `fp_fvm_flux.py`) evolves cell
  averages on a structured `TensorProductGrid` (1D + 2D) with a shared interface-velocity flux,
  so mass conserves to machine precision (~1e-15) by flux telescoping — the higher-order
  extension of the divergence-upwind FDM stencil. Reconstructions: 1st-order `upwind` and
  2nd-order `muscl` (minmod-limited, TVD → positivity + O(dx²)). Time stepping is IMEX by Strang
  splitting (CFL-bounded explicit advection + backward-Euler implicit conservative diffusion).
  Interface velocity comes from `drift_field` (velocity, averaged to faces) or `potential_field`
  (value function U → `α = -coupling·∇U`, the MFG-coupling entry point). Boundary conditions:
  no-flux and periodic (exact conservation); Dirichlet for the diffusion operator only. Registered
  as `NumericalScheme.FVM_UPWIND` / `FVM_MUSCL` and wired into `create_paired_solvers` (paired
  with the upwind HJB-FDM solver), so `problem.solve(scheme=NumericalScheme.FVM_MUSCL)` dispatches
  to it. Verified gates: mass drift ~1e-15 (1D/2D, free + advective-diffusive); MUSCL positivity
  (min density 0 on an advected top-hat); convergence slope ≈1.87 (MUSCL) vs ≈0.92 (upwind);
  FVM-vs-FDM max-diff shrinking under refinement (8.9e-4 → 2.4e-4). Deferred: corner handling
  (#663), 3D, WENO/PPM, unstructured meshes, varying/tensor/callable volatility, Dirichlet
  advection inflow.

- **Assembled-matrix DMP verification + runtime guard for joint_socp** (Issue #1074, paper-critical).
  Per-stencil SOCP feasibility (already tested) does **not** imply the *assembled* HJB iteration
  matrix `I/Δt − D·L + α·D_grad` is an M-matrix — the paper's discrete-maximum-principle claim.
  Empirical finding (verify-by-artifact): the assembled interior matrix **is** an M-matrix at zero
  drift for σ∈{0.3,0.5,1.0} (the SOCP-monotone Laplacian gives the diffusion-part DMP), but the
  signed `α·D_grad` term **flips an off-diagonal positive** once `|α| > α_crit = D·min_edge(L_ij /
  ‖D_grad_ij‖)` — a Péclet-like threshold. So **the DMP holds conditionally (diffusion-dominated),
  not unconditionally**. Crucially this off-diagonal *sign* condition is **dt-independent** (a smaller
  Δt restores diagonal dominance but not the sign). Added `verify_assembled_m_matrix(J,
  interior_indices)` and `critical_drift_for_dmp(D_lap, D_grad, D)` to `gfdm_components/
  monotonicity_enforcer.py`, and an **opt-in** `HJBGFDMSolver(check_dmp=True)` runtime guard that
  warns once when a solve's drift exceeds `α_crit`. The guard is **numerically inert** — verified the
  GFDM joint_socp solve is **byte-identical** with `check_dmp` False vs True (`max|Δ|=0`), and it
  defaults off (zero overhead on the paper path). Characterization tests pin both halves (M-matrix at
  zero drift; broken under strong drift) so a future unconditional-DMP claim fails loudly. The formal
  per-stencil→assembled theorem and the paper-claim wording (scope to low-Péclet vs add gradient
  upwinding) remain open for the author.

- **FP mass-conservation regression now covers P2** (Issue #470 follow-up). The FEM FP advection
  operator is assembled as `-C^T`, whose column sums vanish by partition of unity
  (`Σ_i φ_i = 1`) — an *order-agnostic* property. `test_fp_advection_is_mass_conserving` is now
  parametrized over P1 and P2, pinning that the P2 path conserves mass exactly (verified
  `max|col sum| ≈ 1e-16` at P2) so a future P2 assembly change cannot silently break it.

- **Quadrilateral FEM solve path, P1 + P2** (Issue #470, smallest actionable slice). The FEM
  `element_map` had `(MeshQuad, 1) → ElementQuad1` but no order-2 entry, so a quad mesh at
  `order=2` raised `ValueError` even though `ElementQuad2` ships with skfem. Added
  `(MeshQuad, 2)` / `(MeshQuad1, 2) → ElementQuad2` and corrected the error text (it omitted
  Quad despite Quad-P1 already working). The mesh-generation layer (`Mesh2D`/`Mesh3D`) still
  emits only simplex meshes via gmsh — and **gmsh is not an installed dependency** — but the FEM
  *solve* path is element-family-agnostic, so a quad mesh built gmsh-free via
  `skfem.MeshQuad.init_tensor` now runs end-to-end (`mesh_adapter` round-trips `element_type='quad'`).
  New `tests/integration/test_fem_quad_path.py`: quad round-trip + Poisson at P1 and P2 (manufactured
  solution) + a P2-more-accurate-than-P1 check. gmsh-based Quad/Hex/Prism *generation* remains the
  large, separately-blocked part of #470. FEM is a distinct scheme — no paper-path (FDM/GFDM) impact.

- **`FixedPointIterator` supports unstructured-mesh geometry** (coupled-FEM chain, seam 3). The
  coupling iterator was grid-only — it required a `CartesianGrid` and used `get_grid_shape()` /
  `get_grid_spacing()`, so a FEM pair (which needs an unstructured mesh) could never run through
  the standard fixed-point loop, only a hand-rolled Picard. It now accepts `UNSTRUCTURED_MESH`
  geometry: flat per-DOF state `shape = (num_spatial_points,)` and a unit volume element (the L2
  convergence is a *relative* tolerance, so a constant volume weight does not change convergence
  detection). The `CartesianGrid` path is gated and **byte-identical** (verified: an FDM grid
  coupled solve reproduces its pre-change `U`/`M` exactly). `track_measure_field` (grid-only
  `GridMeasureField`) now fails loud on mesh geometry rather than crashing downstream. With this,
  the **full coupled FEM MFG solves through the standard `FixedPointIterator`** (no-flux,
  mass-conserved) — completing the seam-1→2→3 chain. `tests/integration/test_coupling_mesh_geometry.py`.

- **Cross-path convention-agreement guards** (`tests/unit/test_convention_agreement.py`). The
  dominant bug class here is the same convention implemented along parallel code paths with
  private copies and silent divergence. These tests pin the conventions that have *converged*
  so a future private-copy drift fails loudly: (1) `sigma -> D` resolution agrees across the
  canonical converter, `MFGProblem.diffusion`, the GFDM `_get_sigma_value` path, and the backend
  literal `0.5*sigma**2` (Issue #811/#1192); (2) `get_bounds()` is the one uniform bounds
  accessor across `TensorProductGrid` / `Hyperrectangle` / `Hypersphere` / CSG composites
  (the `.bounds` / `get_bounding_box` non-uniformity itself stays tracked in #1056).

- **6-month removal criterion + `deprecated_on` date** for the deprecation policy (CLAUDE.md
  "3 minor versions OR 6 months"). Only the version criterion was implemented; the time
  criterion is now wired through a single source of truth, `_removable_by_policy(since,
  deprecated_on, current_version)` (major-aware minor diff `(Δmajor·100 + Δminor) ≥ 3`, OR
  `≥ 183` days since an optional `deprecated_on="YYYY-MM-DD"` recorded on `@deprecated` /
  `@deprecated_parameter`). `check_removal_readiness` and `audit_all_deprecations` both delegate
  to it, so the two eligibility calcs can no longer disagree. `audit_all_deprecations` now
  defaults `current_version` to the installed package version and dedups deprecations surfaced
  on many inherited subclasses by `(name, type, since)`. Added regression tests for the policy
  (3-minor / major-aware / 6-month-date paths), the audit delegation/dedup, and the
  `deprecated_parameter` user-vs-default fix below.

- **Coupled meshless-Galerkin-vs-FDM regression test** (Issue #1145 acceptance gap). The
  meshless MFG fixed point previously diverged to NaN and had *no* coupled-vs-reference test —
  only isolated-piece tests. `TestMeshlessGalerkinCoupled` runs the full HJB↔FP Picard with the
  opt-in stabilization recipe (`use_newton=True`, streamline-diffusion) and asserts it stays
  finite (the #1145 NaN regression), mass-bounded, and its terminal mean tracks `FDM_UPWIND`
  within 0.05 (slow/integration). The *unstabilized default still NaNs*, so #1145 stays open for
  that (default-stabilization decision) + the clip-limited convergence.
- **Diffusion-magnitude invariant CI gate** (`tests/integration/test_diffusion_magnitude_gate.py`).
  A standing parametrized gate that pins every diffusion-carrying solver to the correct
  magnitude `D = sigma^2/2` (Issue #811) via cosine-eigenmode decay (`exp(-D*sum k^2*T)`).
  It FAILS on the wrong-coefficient bug class that finiteness/mass/self-consistency tests
  miss (#1152 sigma->D, #1178 ADI dt/dimension, #1183 sigma->mean) — verified discriminating
  (a halved magnitude gives relerr 0.46 vs the 0.03 threshold). Covers ADI (1D/2D/3D, tier1),
  the FP-FDM explicit + implicit paths (tier2), and the production HJB-GFDM per-point Newton
  path (`joint_socp` + `precompute`) in isolation via MMS source-cancellation (slow/tier3;
  correct D -> field relerr ~0.012, halved/doubled D -> 0.105/0.295, 0.05 threshold).
- **Canonical `diffusion_from_volatility(sigma, *, kind=None)` converter**
  (`mfgarchon/utils/pde_coefficients.py`, Issue #811). Single source of truth for the
  SDE-volatility -> PDE-diffusion conversion `D = (1/2) Sigma Sigma^T` (scalar `sigma^2/2`),
  per NAMING_CONVENTIONS "Volatility vs Diffusion": tensor-first (volatility is a tensor in
  general), `Sigma Sigma^T` not `Sigma^T Sigma`. Fail-loud, no silent guess: a scalar `sigma`
  is unambiguous, but an array requires `kind` (`"field"` = isotropic per-point `sigma^2/2`
  elementwise; `"tensor"` = trailing `(d,k)` is Sigma, `D = 0.5*Sigma@Sigma.T`, `ndim>=2`) —
  an array with `kind=None` raises rather than guessing `(d,d)`-tensor vs `(Nx,Ny)`-field.
  Replaces the dominant silent-convention bug class (#1152/#1178/#1183) at the root by making
  `D` come from one place. First consumer: `hjb_sl_adi.py`; remaining ~36 ad-hoc
  `0.5*sigma**2` sites tracked in #1189 (byte-identical migration).

### Changed

- **Five un-instantiable solvers demoted to clearly-experimental** (Issue #1342, PR #1351).
  `SinkhornMFGSolver`, `WassersteinMFGSolver`, `VariationalMFGSolver`, `PrimalDualMFGSolver`, and
  `MFGDGMSolver` were missing required abstract-method implementations (broken since 2025-09-29)
  and raised a cryptic `Can't instantiate abstract class` error. They now raise a clear
  `NotImplementedError` naming them experimental and referencing #1342, are excluded from the
  production factory / `NumericalScheme` dispatch, and are documented experimental. Their solve
  logic is preserved but unvalidated; completing it remains open (#1342).

- **Solver-selection performance guidance** (PR #1349). `NewtonMFGSolver` is documented as
  research-grade for d ≥ 2 — ~135× slower than the fixed-point (Picard) coupler in 2D and scaling
  poorly with dimension and grid size; prefer `FixedPointIterator` or Howard for production
  coupled solves.

- **Robin / Periodic FEM boundary conditions now fail loud** (Issue #1237). `apply_bc_to_fem_system`
  previously *warned and silently fell back to natural (Neumann) BC* for `BCType.ROBIN` — a
  fail-silent anti-pattern that ran the wrong physics quietly — and warn-only (no fallback) for
  `BCType.PERIODIC`. Both now raise `NotImplementedError` with a pointer to the use-an-FDM/GFDM-solver
  workaround. A correct Robin FEM BC needs a diffusion-coefficient-scaled `FacetBasis` boundary term
  threaded through the weak-form assembly (the adapter is coefficient-blind — it only sees the
  already-assembled matrix); Periodic needs DOF identification across paired boundaries. Both are
  tracked in #1237. Nothing exercised these stubs through the FEM path, so this only converts a silent
  wrong-answer into a loud error. New parametrized fail-loud tests in `test_fem_solver_path.py`.

- **Single-sourced the tensor-diffusion operator** (Issue #1228). `∇·(Σ∇u)` was implemented twice:
  `operators/differential/diffusion.py` (`DiffusionOperator` private `_tensor_diffusion_1d/2d/nd`)
  and `utils/numerical/tensor_calculus.py` (`diffusion`) — two independent copies, the repo's
  dominant "parallel implementation, no single owner" bug class. Verified the two are **bit-identical**
  (`max|Δ| = 0` across 1D/2D/3D, scalar / diagonal / full / spatially-varying tensors, and
  periodic / no-flux / Dirichlet BCs), then routed `DiffusionOperator._apply_tensor_diffusion`
  through the lower-level `tensor_calculus.diffusion` (layering: operators → utils) and deleted the
  ~190 lines of private duplicate (incl. the now-dead `_pad_array`). Byte-identical, so the
  paper-adjacent `fp_fdm_time_stepping` consumer is unchanged (87 diffusion/tensor/FP-FDM tests
  pass). Added a single-source guard in `tests/unit/test_convention_agreement.py` pinning the two
  diffusion entry points to agree. The scalar-isotropic path (`laplacian_with_bc`) is unchanged.

- **Single-sourced the σ→D conversion in the weak-form / FEM family** (Issue #811 / #1192; FEM
  audit). The weak-form HJB and FP solvers had three inline `0.5 * sigma**2` copies
  (`weak_form_fp_solver._diffusion_coefficient`, `solve_fp_step_adjoint_mode`, and
  `weak_form_hjb_solver.solve_hjb_system`) that bypassed the canonical `diffusion_from_volatility`
  and had already diverged for an array volatility (silently collapsing the field to a scalar
  mean). Added one `scalar_diffusion_from_volatility(volatility_field, fallback_sigma)` helper in
  `pde_coefficients` that all three delegate to; it routes the formula through the single source
  and makes the scalar-D field-collapse **loud** (warns that a spatially-varying field is reduced
  to its mean, since these solvers assemble `D * K` with one scalar `D`). Byte-identical to the
  prior copies (`None→0.5σ²`, `scalar→0.5v²`, `array→0.5·mean(v)²`); also dedups the
  `solve_fp_step_adjoint_mode` copy onto `_diffusion_coefficient`. EOC-safe (weak-form/FEM is off
  the paper EOC path; scalar paths bit-identical).

- **Single-sourced boundary on-wall tolerances** (Issue #1101). The scattered `1e-6` / `1e-8` /
  `1e-10` / `1e-12` magic literals across the boundary / geometry on-wall classifiers are now
  named, documented, tunable constants in `mfgarchon/geometry/boundary/tolerances.py`
  (`BOUNDARY_TOL=1e-6`, `ONWALL_TOL=1e-10`, `SDF_BOUNDARY_TOL=1e-8`, `BOUNDARY_REL_TOL=1e-12`).
  Routed ~25 sites across `conditions`, `base`, `types`, `point_cloud`, `implicit_domain`,
  `tensor_grid`, `network_geometry`, GFDM `boundary_handler` and `hjb_gfdm`. **Byte-identical**:
  each literal maps to a same-valued constant, so every path (incl. the GFDM `1e-6` paper path
  and the analytic-geometry `1e-10` exact-membership path) is unchanged — verified by the full
  geometry + GFDM suites (764 tests). The distinct values are preserved on purpose (grid-exact
  `1e-10` vs scattered-cloud `1e-6` vs SDF `1e-8` are different *measures*); they are **not**
  collapsed to one (that would loosen analytic boundary detection by four orders of magnitude).
  Non-on-wall `1e-X` (FD-step `eps`, degenerate-normal guards, iterative-projection convergence,
  Lipschitz validation, point-equality, QP active-set) were left untouched. `protocol.py`'s
  `typing.Protocol` stub defaults are also left as literals (not executed; concrete impls route
  through the constants). Pinned by `tests/unit/test_convention_agreement.py`.

- **Canonical FP drift contract: explicit `_drift_convention` trait + weak-form `potential_field`
  rename** (Issue #1043). The FP equation only ever sees the advective velocity `α` in
  `div(α m)`; most solvers (FDM, GFDM) take that velocity as `drift_field`, but the weak-form
  family (`WeakFormFPSolver` + FEM/meshless) and the network/SL solvers take the value function
  `U` and recover `α = -coupling·∇U` internally. Added a `DriftConvention` enum + a
  `_drift_convention` class trait (default `VELOCITY`) so this distinction is explicit and
  machine-readable, and set `VALUE_FUNCTION` on the U-taking solvers. Renamed the weak-form
  family's misnamed `drift_field` U-input to **`potential_field`** (the name the SL solvers +
  couplers already prefer, #919), with a `drift_field` deprecation alias (byte-identical;
  equivalence + fail-loud-on-both tests added). **Deferred** (paper-number risk, like #1071
  steps 4-5): the coupler trait-dispatch that would *enforce* the contract (the smooth-H common
  case currently passes raw `U` and the solver differentiates it on its own grid — moving that
  into the coupler changes where the gradient is taken → not byte-identical), and the
  `FPNetworkSolver`/`FPParticleSolver` param renames (network is internally inconsistent; particle
  is bivalent via a flag). `Refs #1043`.
- **Single-source HJB residual/Jacobian assembly** (`hjb_solvers/h_eval.py`, Issue #1071 Layer B).
  Added `assemble_hjb_residual` (`-u_t + H(+running_cost) - D·lap_u`) and
  `assemble_hjb_jacobian_diag` (`(1/dt)I + Σ_d diag(∂H/∂p_d)@D_grad[d] - D·D_lap`), so the
  diffusion-term convention (`D = σ²/2`, #1073/#811) and the assembly skeleton live in one place;
  callers supply their own discrete operators (`D_grad`, `D_lap`) and time-derivative. Folded
  `hjb_gfdm`'s `_compute_hjb_residual_hamiltonian` + `_compute_hjb_jacobian_hamiltonian` onto them,
  **byte-identical** (112 gfdm tests unchanged). Scoped to the gfdm implicit-residual framing; the
  `base_hjb` (`Phi_U += -D·U_xx`) and WENO (explicit `-H + D·Δu`) framings are intentionally not
  folded (irreducibly different). Most of the diffusion-convention centralization value was already
  delivered by the #1189 converter sweep + Layer A.
- **Single-source batch Hamiltonian evaluation** (`hjb_solvers/h_eval.py`, Issue #1071 Layer A).
  Every HJB solver inlined the same `np.asarray(H_class(x, m, p, t=t), dtype=float)` batch call
  (6 value + 5 `dp` sites across `base_hjb`/`hjb_fdm`/`hjb_gfdm`/`hjb_semi_lagrangian`); these now
  route through `eval_H_batch` / `eval_dH_dp_batch`, so a change to the batch contract (dtype,
  shape, NaN policy) happens in one place. **Byte-identical** — callers keep their own `.ravel()`
  / reshape / `alpha* = -dp` sign conventions and discrete operators (369 HJB tests unchanged).
  Foundation for the residual/Jacobian assembly harness (Layer B); the hardcoded-LQ-default
  elimination (steps 4-5) is deferred (it is not byte-identical and could shift paper EOC numbers).
- **Upgraded two "plumbing-only" tests to run-and-compare (retrospect rec ③).**
  `test_qp_optimization_levels` only asserted constructor *labels* (`hjb_method_name`) — it
  would pass even if a QP monotonicity level corrupted the solution; added
  `test_qp_optimization_levels_agree_on_smooth_problem`, which solves at all three levels on
  a smooth problem and asserts they agree to <1e-4 (a real regression guard). And
  `test_fp_matrix_conservation`'s `TestFPMatrixConservation` asserted column-sum=1/dt on a
  *test-local re-implementation* of the FP matrix (a shadow that passed even if the production
  assembly diverged); replaced by `TestFPProductionConservation`, which runs the real
  `FPFDMSolver` and asserts machine-precision (<1e-12) mass conservation — the tight signal a
  wrong boundary stencil would break. Removed a no-op placeholder test.
- **~30 numerical-core diffusion sites now route through `diffusion_from_volatility`**
  (Issue #1189, the sweep follow-up to the #1190 converter). The literal `0.5*sigma**2`
  / `sigma**2/2` is replaced by the single-source converter across the problem core
  (`mfg_problem._volatility_to_diffusion` now delegates its scalar/diagonal/tensor shape
  dispatch; `.diffusion` property routes through it), all 8 FP solvers, 5 HJB solvers
  (`base_hjb`/`hjb_fdm` use `kind="field"` for their array-capable sigma; the 7 gfdm,
  3 weno, 1 SL sites are scalar), and the adjoint ADI/BC-coupling builders. Byte-identical
  (verified against the diffusion-magnitude gate + the FP/HJB suites). The wrong-coefficient
  bug class (#1152/#1178) is now unrepresentable in these paths, not merely caught downstream.
  Out of scope (documented): the PINN/torch-backend sites (torch tensors, not numpy), the
  code-generation template, and `flux_diagnostics`.
- **Spatially varying volatility on the explicit-drift / strict-adjoint FP paths now warns**
  instead of silently approximating (Issue #1183). Those paths use a constant-coefficient
  diffusion (scalar `D = mean(sigma)^2/2`), so a genuinely non-uniform `volatility_field`
  silently solved a different PDE than the per-point implicit path. They now emit a
  `UserWarning` when a non-uniform array sigma is collapsed (uniform arrays collapse exactly
  and do not warn). The full per-point variable-coefficient diffusion on these paths remains
  tracked in #1183 (it needs an operator that matches the existing conservative no-flux
  discretization).

### Deprecated

- **`mfgarchon.operators.nonlocal_ops` renamed to `mfgarchon.operators.integro_diff`**
  (Issue #1024). The subpackage held two unrelated notions of "non-local": Lévy
  integro-differential operators (non-local *PDE structure*) and graphon coupling, which
  collided in name with the *game-coupling* non-local operators in `operators/interaction/`
  (#1023). "Integro-differential" is the standard literature term (Jakobsen-Karlsen,
  Barles-Imbert) and is unambiguous. The old path is retained as a deprecation shim that
  re-exports the public API, aliases the submodules (`levy_integro_diff`, `levy_measures`,
  `graphon_coupling`, `graphon_kernels`) so existing dotted imports keep resolving, and emits
  a `DeprecationWarning` on import. An equivalence test asserts both paths return identical
  class objects. Shim removal scheduled for v0.22.0. Update imports:
  `from mfgarchon.operators.integro_diff import ...`.

### Removed (BREAKING)

- **`BoundaryConditions.default_bc` no longer silently defaults to `PERIODIC`** (Issue #1100).
  The dataclass field is now `default_bc: BCType | None = None` ("unspecified"). Previously any
  boundary point that matched no segment (coverage gap, geometric-tolerance edge) silently fell back
  to periodic wrapping — wrong physics for the non-periodic 2D geometries that dominate real use, and
  the proximate cause of silent zero-Jacobian rows (#1098). Now, resolving an unmatched point with
  `default_bc` unset raises a clear `ValueError` ("...matched no BC and default_bc was not specified
  ...; set default_bc=BCType.NO_FLUX / PERIODIC / DIRICHLET explicitly (Issue #1100)") at the point→BC
  resolution sites (`BoundaryConditions.get_bc_at_point` / `get_bc_type_at_boundary` via the new
  `_resolve_default_bc` helper, and the implicit/meshfree applicators, `fp_gfdm`, and
  `base_solver` constraint/env-config builders). **Migration**: pass `default_bc=BCType.NO_FLUX`
  (safe, mass-conserving) — or `BCType.PERIODIC` if you genuinely want wrapping — whenever your
  segments may not cover every boundary point; the `*_bc()` factories (`no_flux_bc`, `periodic_bc`,
  …) already set `default_bc` explicitly and are unaffected. The `mixed_bc_from_regions` factory's
  no-`"default"`-segment fallback changed from `PERIODIC` to `NO_FLUX`. Global-property reads
  (`__str__`, `validate()`, `validate_boundary_conditions`) treat `None` as "not specified" and do
  not crash. Explicit-BC paths (no-flux / Dirichlet / explicit-periodic) are byte-identical.

- **Deprecated `fdm_bc_1d` 1D BC factory functions** `periodic_bc`, `dirichlet_bc`, `neumann_bc`,
  `no_flux_bc`, `robin_bc` (deprecated since v0.14.0 — past the 3-minor-version removal window at
  v0.19.8). They were dead code: every live import resolves the same names from the geometry-first
  `conditions.py` (re-exported by `mfgarchon.geometry` / `…geometry.boundary`); nothing imported the
  factories from `fdm_bc_1d` (only its `BoundaryConditions` dataclass is still used). Use
  `from mfgarchon.geometry import periodic_bc; bc = periodic_bc(dimension=1)` (the documented
  replacement). The legacy `fdm_bc_1d.BoundaryConditions` dataclass is unchanged (still imported by
  the FDM/applicator layer; itself deprecated since v0.14.0 and kept until its consumers migrate).

### Fixed

- **FP-particle drift gradient ignored boundary conditions: O(1/h) wrong-sign drift at
  non-periodic walls** (silent-divergence bug-hunt). `FPParticleSolver._compute_gradient_nd`
  computed `∇U` for the drift `α = -∇U/λ` via the **periodic-wrap** central difference
  (`stencils.gradient_nd`) regardless of BC, while the HJB uses the BC-aware
  `geometry.get_gradient_operator`. At a non-periodic wall the periodic stencil takes
  `(U[1] − U[N-1])/(2h)` — wrapping to the *far* wall — giving an **O(1/h) wrong-sign** drift that
  pushes mass away from the wall (verified: for `U=x²` on `[0,1]` no-flux at N=51, `g[0]=−24.99`
  vs exact 0; end-to-end ~2.5× wrong wall density, ~15% mass misplaced). It now routes through the
  same BC-aware operator the HJB uses, unifying the two gradient conventions (ghost-padded /
  one-sided at non-periodic boundaries; periodic BC reproduces the wrap **byte-identically** —
  verified `max|Δ|=0`). The GPU/backend path round-trips through host numpy for the operator. Not a
  byte-identical paper path (those are FP-FDM / FP-GFDM); the precomputed-drift path
  (`drift_is_precomputed=True`, e.g. the GFDM→particle handoff) bypasses this gradient entirely.
  **FP-*particle* experiments on non-periodic domains were silently wrong at the boundary and
  should be re-validated.** Replaced the obsolete `test_compute_gradient_zero_dx` (it pinned the
  now-advisory passed-spacing behavior) with a boundary-BC-aware refinement regression test.

- **Regime-switching / graph MFG iterators declared convergence on the value function only**
  (silent-divergence bug-hunt). `RegimeSwitchingIterator` and `GraphMFGSolver` gated
  `converged=True` on `max_k|U^k_{n+1} − U^k_n| < tol` with **no density term** — half of the
  canonical `(u, m)` criterion (`fixed_point_utils.check_convergence_criteria` requires both). When
  a regime/node's value function stabilizes faster than its density (different timescales across
  regimes, strong mass-transfer coupling), the iterator returned a solution whose density was still
  evolving — reproduced for a two-volatility-regime problem: `solve()` reported converged at iter 12
  with `error_U = 7e-5 < tol` but `error_M = 3.3·tol` (M reached tol only at iter 15). Both now gate
  on `max(error_U, error_M)` (with an iteration-0 shape guard: the 1D initial density vs the 2D
  trajectory makes the M-change undefined → treated as not-converged). Verified the fix still
  converges (both U and M below tol, monotone) and the existing integration tests stay green.
  Not on any byte-identical paper path (Phase-2 institutional-MFG features). Docstrings corrected
  (tolerance was documented as U-only).

- **Multi-population K=1 converged to a non-fixed-point: BoundHamiltonian defeated the FP drift
  dispatch** (Issue #1043 follow-up). After the FP-drift single-sourcing below, a K=1 multi-pop
  solve *still* diverged from single-pop (`||F_FP|| ≈ 6.9`, 19% density / 44% U) — and the
  multi-pop solution was **not a coupled fixed point** (`||F_HJB|| ≈ 0` but `||F_FP|| = O(1)`; a
  Picard step from it moved M by 11%). Root cause: the iterator binds `H_bound =
  H.bind_cross_density(...)`, a `BoundHamiltonian` wrapper that fails
  `isinstance(SeparableHamiltonian)`, so `resolve_fp_drift_kwargs` took the *velocity* path while
  single-pop (unbound H) took the *potential* path — two different (self-consistent-but-distinct)
  fixed points. The wrapper's `optimal_control` delegates to the inner H, so a bound
  smooth-separable H still has the momentum-only optimal control that makes `potential_field=U`
  correct (cross-coupling enters via the HJB, not the FP drift). The resolver now unwraps
  `H._inner` for the smoothness dispatch; K=1 multi-pop now matches single-pop **exactly** (`U`/`M`
  diff `< 1e-4`, bounded by the Picard tolerance; was 116%/19% across the two stacked bugs).
  No-op for single-pop (no `_inner` → byte-identical; Newton-Picard + convention guards pass).
  Tightened `test_k1_matches_single_population_fp_convention` to assert the exact match.

- **Multi-population FP drift now single-sourced with single-pop** (Issue #1043). The
  `MultiPopulationIterator` computed its own *node*-centered velocity (`np.gradient`) and always
  passed it as an explicit `drift_field`, while the single-population `FixedPointIterator`/
  `MFGResidual` resolve drift through `resolve_fp_drift_kwargs` — *face*-centered velocity (#919,
  matched to the divergence-upwind stencil) for non-smooth H, or `potential_field=U` for
  smooth-separable H. Two divergent conventions on parallel paths (the repo's dominant bug class):
  feeding the *same* U through the same FP solver, the node-centered velocity gave a **~68%**
  different density, and a **K=1** multi-population solve was **~116%** off the equivalent
  single-population solve. The iterator now routes its continuous-domain FP drift through the
  shared `resolve_fp_drift_kwargs` (new `h_class=` param threads the cross-density-bound
  Hamiltonian so the velocity still sees other populations' density, #1157); the private
  `_compute_velocity_field` copy is removed and the network branch (`spatial_dimension == 0`) is
  preserved. K=1 now matches single-pop to **~19%** (residual is the iterators' other non-FP
  differences — single-pop damps U+M and runs Anderson/source/BC machinery; the multi-pop loop is
  a leaner Picard — not the drift convention). `resolve_fp_drift_kwargs(h_class=None)` is
  byte-identical for single-pop. New `test_k1_matches_single_population_fp_convention`; 18 existing
  multi-pop tests still pass. (`_compute_drift_field`, the separately-deprecated node-centered
  sibling, keeps its own v0.25.0 removal timeline.)

- **`NewtonMFGSolver` diverged ~99.5% from Picard** (Issue #1233). The Newton coupling residual
  `F = [HJB(M) - U, FP(U) - M]` was inconsistent with the Picard fixed point, so the two solvers
  converged to different roots. Two causes: **(1) stale FP drift convention** — `MFGResidual`
  passed the value function as `drift_field`, but the v0.18.6 rename redefined `drift_field` as
  the *velocity* $\alpha^*$ (the old U-potential entry point became `potential_field`); the
  Picard `FixedPointIterator` was updated (#896/#919) while `MFGResidual` was not, so
  `||F_FP(Picard soln)||` was O(10) instead of ~0. The drift/potential resolution is now
  **single-sourced** in `resolve_fp_drift_kwargs` + `compute_fp_velocity_field`
  (`fixed_point_utils`), shared by both `FixedPointIterator` and `MFGResidual` — the
  `FixedPointIterator` FDM path is **byte-identical** (verified `max|Δ|=0` on 1D-LQ `U`/`M`).
  **(2) basin selection** — even with a consistent residual, a too-short Picard warmup (3) left
  the iterate in the basin of a spurious near-trivial discrete fixed point; Newton (a local
  root-finder, line search cannot escape a basin) locked onto it. Documented the warmup-basin
  requirement; the comparison test now warms up adequately. Also **defaulted
  `use_jax_autodiff=False`** for `NewtonMFGSolver`: the residual wraps black-box numpy/scipy
  solvers that JAX cannot trace, so `'auto'` was a guaranteed `TracerArrayConversionError` +
  FD-fallback warning on every run (the option is retained for a hypothetical jnp-native
  residual). Added fast, non-`slow` guards (`tests/integration/test_newton_picard_agreement.py`):
  the realistic-grid comparison is `slow` (skipped on PRs), which is why this stayed red on
  `main` undetected. (`multi_population_iterator` keeps its own *node*-centered velocity copy —
  a different convention — and is left as a tracked third copy, not folded in here.)

- **FEM named-region BC silently unresolved: no facet boundary tags** (Issue #607; coupled-FEM
  chain, seam 2). `meshdata_to_skfem` produced a skfem mesh with `mesh.boundaries = None`, so a
  `BCSegment(boundary="x_min")` could not be resolved — the FEM `bc_adapter` either crashed
  (`argument of type 'NoneType' is not iterable`) or fell back to the *entire* boundary. Now
  `meshdata_to_skfem` tags axis-aligned wall facets as named boundaries (`x_min`/`x_max`/`y_min`/…,
  matching `BoundaryFace.to_string()` / `BCSegment.boundary`) via skfem `with_boundaries`, keyed off
  the mesh bounding box with `BOUNDARY_TOL`. A Dirichlet segment now resolves to exactly its wall's
  DOFs (verified: `x_min` → only `x=xmin` DOFs, a strict subset of the boundary), and an FP solve
  with a Dirichlet BC runs. `_find_segment_dofs` also guards `mesh.boundaries is None` (untagged
  meshes fall back cleanly instead of raising). Curved/SDF region markers remain out of scope. The
  last coupled-FEM seam is `FixedPointIterator` mesh-geometry support. Refs #607.

- **`UnstructuredMesh.get_boundary_conditions()` returned boundary-handler metadata, not a
  `BoundaryConditions`** (coupled-FEM chain, seam 1). It returned `get_boundary_handler()` — e.g.
  `{"type": "unstructured_mesh", "boundary_faces": None}` — which, as the higher-priority accessor
  in the solver BC-resolution order, **shadowed the real problem-level `BoundaryConditions`** and
  crashed the FEM `bc_adapter` with `'dict' object has no attribute 'segments'`. Now returns the
  `BoundaryConditions` attached to the geometry, or `None` (callers then fall back to the
  problem-level BC; `is_pure_neumann(None)` → natural Neumann). With this, the FEM solver receives
  a real `BoundaryConditions` (when attached to the geometry — note `MFGProblem(boundary_conditions=)`
  is dropped for mesh geometries, so attach to the geometry), and the **first coupled FEM MFG solve
  through the real solver classes** runs (no-flux, mass-conserved — `test_fem_solver_path.py`). The
  next links remain: the Dirichlet segment→facet resolver (#607) and `FixedPointIterator`
  mesh-geometry support. Consolidated tests: removed `test_fem_coupled_mfg.py` (3 hand-rolled raw
  `K`/`M` tests that bypassed the solver classes, misleadingly named "coupled") — fully subsumed by
  `test_fem_solver_path.py` (real-class coupled) + `test_fem_mfg_solve.py` (forward/backward heat,
  assembly).

- **FEM solver path was broken at three seams** (FEM-readiness audit; Issues #773, #580). (1) The
  FP advection operator was assembled as the raw convective form `+C` (`C[i,j]=∫φ_i(v·∇φ_j)`),
  which is **not** mass-conserving — column sums `≈∫v·∇φ_j≠0`, giving ~20%+ mass drift on a
  non-divergence-free drift. Now assembled as `-C^T` (integration-by-parts form, zero column
  sums since `Σφ_i=1`), matching the meshless-Galerkin sibling and the adjoint identity
  `A_FP=A_HJB^T` (verified: `max|col sum|≈6e-17`). (2) `HJBFEMSolver`/`FPFEMSolver` inherited
  `_scheme_family=GENERIC`, so the documented factory entry point `create_paired_solvers(..., FEM_P1)`
  **raised** (duality VALIDATION_SKIPPED); both now carry `SchemeFamily.FEM`, which is registered as
  a Type-A (exact-discrete-transpose) Galerkin family in the duality validator. (3) `Mesh2D`/`Mesh3D`
  `generate_mesh()` now returns a pre-populated `mesh_data` as-is — a gmsh-free path for callers
  that inject a mesh (e.g. via `skfem_to_meshdata`), since gmsh is not in the default install. Added
  `tests/integration/test_fem_solver_path.py` exercising the real solver classes (the prior
  `test_fem_coupled_mfg.py` hand-rolled raw `K`/`M` and never touched them). NOTE: coupled FEM
  *through* `FixedPointIterator` still cannot run — the iterator requires a `CartesianGrid` while
  FEM requires an unstructured mesh, and `fem/bc_adapter` expects a `BoundaryConditions` object
  where the solver receives a `dict`; those two seams are the remaining coupled-FEM prerequisites.

- **`BoundaryConditions.get_outward_normal` mis-fired on outer walls of Difference-style domains**
  (Issue #1114). It checked `domain_sdf` *first*, so for a domain that is an outer box minus an
  obstacle (both `domain_bounds` and `domain_sdf` set), a point on the outer wall received the
  *obstacle's* SDF gradient — pointing toward the obstacle, tens of degrees off the wall normal —
  instead of the axis-aligned wall normal. Unified the two normal sources into one classifier:
  an axis-aligned outer-box wall now returns the exact face normal (via `outward_normal_for_face`),
  and the SDF gradient is used only for genuinely curved boundaries (the obstacle surface, or a
  pure-SDF domain). The box-wall branch is guarded by an explicit axis-bound match (not
  `identify_boundary_face`, whose SDF Method-2 also classifies curved points) so obstacle-surface
  normals keep the SDF gradient. The GFDM paper path was unaffected either way (it uses the
  `identify_boundary_face` + `outward_normal_for_face` Path-1, with `get_outward_normal` only a
  legacy fallback). Added `TestOutwardNormalSourceAgreement`.

- **HJB-SL reflecting-BC characteristic-foot fold was wrong on asymmetric domains**
  (Issues #1161, #1048, #1054). Three sibling code paths in `hjb_semi_lagrangian.py` folded
  out-of-bounds characteristic feet for no-flux/Neumann BC, each with a private copy of the
  rule: the deterministic explicit/rk2 fast path used `np.clip` (clamping feet onto the wall
  node, biasing toward the wall value — #1161), while the stochastic 1D (#1048) and nD (#1054)
  paths used `xmin + |((x-xmin) mod 2L) - L|`, which is a point-inversion about the domain
  *center* (`x -> xmin+xmax-x`), **not** a boundary reflection — it displaced even in-bounds
  feet. All three shipped because their tests used a symmetric `[0,1]` domain/data where the
  center-flip is a symmetry of the solution (silent divergence). Replaced all three (plus the
  duplicated formula) with a single vectorized `reflect_into_domain(x, xmin, xmax)` helper in
  `hjb_sl_characteristics.py`, the closed-form triangle wave `xmin + L - |((x-xmin) mod 2L) - L|`
  verified equal to the trusted iterated scalar `apply_boundary_conditions_1d` on asymmetric
  domains and identity in-bounds. Numerics are unchanged on symmetric-domain runs and on the
  paper EOC paths (FDM / GFDM, not SL); asymmetric-domain reflecting-BC SL runs now reflect
  correctly. Added `TestReflectIntoDomain` (asymmetric-domain reflected-value tests — the
  coverage gap that let the bug ship).

- **Polymorphic reads of the non-uniform `.bounds` geometry attribute** (Issue #1056). The ad-hoc
  `.bounds` attribute returns four incompatible shapes across geometry classes (`(d,2)` ndarray,
  `(min,max)` tuple, `list[(min,max)]`, or absent), while `get_bounds() -> (mins, maxs)` is the
  uniform `Geometry` ABC accessor present on every class. Migrated the eight cross-type `.bounds`
  readers to `get_bounds()` (`projection`, `hjb_weno`, `gfdm_strategies` periodic, three
  `hjb_semi_lagrangian` sites, `fp_particle`, `base_pinn`). Paper paths already consumed
  `get_bounds()` and are untouched (byte-identical). The migration also fixes four latent bugs the
  shape-divergence caused: `hjb_semi_lagrangian` nD clip (`bounds[0][d]` mis-indexed for `d>=1`),
  `fp_particle` PointCloud bounds transpose, `projection` tuple/ndarray mismatch, and `base_pinn`
  reading a nonexistent `self.problem.domain` (so it always fell back to default bounds).
  1177 affected-suite tests pass. The per-class `.bounds` attributes remain as internal detail
  (deprecating/uniformizing them is optional follow-up); `get_bounds()` is now the canonical
  cross-geometry accessor, pinned by `tests/unit/test_convention_agreement.py`.

- **`FPParticleSolver._drift_convention` trait was dishonest** (Issue #1043). It inherited the
  base `VELOCITY` default, but the solver body is `VALUE_FUNCTION` by default — the 1D path always
  takes the value function `U` via `drift_field` and computes `alpha = -coupling*grad(U)`, and the
  nD default does the same (`drift_is_precomputed=True`, nD only, flips it to `VELOCITY`). Set the
  class trait to `VALUE_FUNCTION` to match the default behavior, documenting the per-call
  precomputed exception. Pure metadata: nothing dispatches on `_drift_convention` (verified — the
  only readers are the contract test), so this is byte-identical on every path. Also added
  `FPParticleSolver` to `test_drift_contract.py`, which previously silently omitted it from both
  assertion lists — the one bivalent solver was the only untested one.

- **Silent collapse of a spatially-varying diffusion tensor to a constant in the FDM tensor
  path** (Issue #1079, partial). When `HJBFDMSolver` is given a spatially-varying *diagonal*
  tensor `volatility_field` (shape `(*grid, d, d)`), it extracts the per-axis diagonal and
  averages it to a single constant per axis — discretizing a constant-coefficient Laplacian and
  silently dropping the spatial variation of `D(x)`. The existing #1169 warning only fired for
  *non-diagonal* tensors, so this varying-diagonal reduction was silent. Added a warning that
  fires only when the diagonal is actually spatially-varying; scalar / constant-tensor inputs
  (incl. the paper EOC paths) are byte-identical (no new warning). The two other #1079 sub-sites
  are **not** addressed here: the GFDM "tensor → trace" drop does not reproduce (`MFGProblem`
  rejects an array `sigma` at construction, and a callable `sigma` is not trace-reduced), and the
  `hjb_sl_adi` diagonal-vs-off-diagonal scaling mismatch needs a `sigma`-vs-`D` convention
  decision — so #1079 stays open for the ADI convention item.

- **`geometry/collocation.py` silently swallowed real `TypeError`s in SDF fallbacks**
  (Issue #1069). Nine `except (AttributeError, TypeError): pass` blocks guard optional
  `geometry.signed_distance(...)` calls in the mesh-optimization / interior-filter paths.
  Catching `AttributeError` is the correct optional-capability signal (TensorProductGrid /
  network geometries legitimately lack `signed_distance`), but also catching `TypeError`
  masked genuine dtype/shape bugs inside geometries that *do* implement it. Narrowed all nine
  to `except AttributeError` so the optional-capability fallback is preserved while real
  failures surface (fail-fast). The four `_select_strategy` capability probes (already
  `except AttributeError` + terminal `raise ValueError`) and the analytic→FD gradient fallback
  are correct idiom and untouched.

- **`@deprecated_parameter` false-positive warning + dead removal-readiness lister**. The
  decorator called `bound.apply_defaults()` and warned whenever the resolved value was
  non-`None`, so any deprecated parameter with a **non-`None` default** emitted a deprecation
  warning on *every* call even when the user never passed it. Now it binds without applying
  defaults and warns iff the parameter was actually supplied (positionally or by keyword).
  Separately, `scripts/audit_deprecated_symbols.py` listed removable symbols via fragile AST
  source-parsing (broken — listed nothing); it now reads the runtime `@deprecated` registry via
  `audit_all_deprecations`, so the READY/NOT-READY/ACTIVE report reflects the actual decorated
  metadata and the shared age policy.

- **Per-point spatially-varying volatility on the explicit-drift & strict-adjoint FP paths**
  (Issue #1183). Both paths collapsed a non-uniform `volatility_field` to its spatial **mean**
  (a scalar `D`), silently solving a different PDE than the per-point implicit path (a low-σ
  region was over-diffused, a high-σ region under-diffused — ~14–21% L2 error). Added a
  variable-coefficient mode to `LaplacianOperator` (`coefficient_field=`): the conservative
  finite-volume no-flux stencil now bakes in the **face-averaged** diffusion
  `D_{i+1/2} = ½(D_i + D_{i+1})`, so `∇·(D(x)∇·)` stays column-conservative (`1ᵀL = 0`) even
  for varying `D` — a point-value `D_i·Δ` would leak. The explicit-drift
  (`solve_timestep_explicit_with_drift`) and strict-adjoint (`solve_fp_step_adjoint_mode`) FP
  steps now build this per-point matrix for an array σ (replacing the mean-collapse + the interim
  warning), so a non-uniform σ is honored per point **and** mass is conserved; low-σ regions
  correctly under-diffuse. Scalar (and uniform-array) σ takes the existing scalar path unchanged
  — **byte-identical**, so no EOC change for the common case. `coefficient_field=None` leaves
  `LaplacianOperator` byte-identical for all other consumers. (The point-value implicit reference
  `solve_timestep_full_nd` is itself non-conservative for varying σ, and the strict-adjoint
  *scalar* path is not yet mass-conservative — both tracked as a follow-up.) `Fixes #1183`.
- **Nonzero Neumann flux in the linear-reflection ghost path** (Issue #1186 sibling, FDM/SL).
  `PreallocatedGhostBuffer._apply_linear_reflection` (the order<=2 ghost path used by FDM/SL)
  filled Neumann/no-flux ghosts with a pure mirror, silently dropping a nonzero
  `neumann_bc(value=g)` (it always encoded du/dn=0). It now adds the linear flux offset with the
  Robin-branch sign (`-dx*g` at the low wall, `+dx*g` at the high wall), so the prescribed
  `du/dn = g` is recovered at both walls. `g = 0` (and `NO_FLUX`/`REFLECTING`, definitionally
  zero-flux) keeps the pure mirror -> byte-identical for the no-flux case, so existing FDM/SL
  solves are unchanged. Companion to the WENO/poly-path fix (#1186); regression tests added.
- **WENO HJB spatial scheme rebuilt: correct gradient + Lax-Friedrichs numerical Hamiltonian**
  (Issue #1200). The previous scheme reconstructed WENO interface *values* and then took a bogus
  central difference `(u_right − u_left)/(2·dx)`, so the nodal gradient fed to the Hamiltonian was
  `≈ −0.25 · du/dx` (wrong sign **and** magnitude) — the solver had never computed a correct
  gradient in any dimension, and the 2D/3D off-axis sweeps were `np.gradient` placeholders. With no
  numerical Hamiltonian, oscillatory terminal data amplified high-frequency modes and blew up
  (`1e26+`), CFL-independent; the whole WENO suite only asserted `isfinite`/shape so this was never
  caught. Replaced with the Osher–Shu HJ-WENO5 one-sided nodal derivatives `p_minus`/`p_plus`
  (undivided-difference stencils, ghost depth 2→3) and a global Lax-Friedrichs numerical Hamiltonian
  `Ĥ = H((p_minus+p_plus)/2) − (α/2)(p_plus−p_minus)`, `α = max|∂_pH|`, whose viscosity damps the
  unstable modes while staying O(h⁵) on smooth data (measured EOC ≈ 6, polynomial-exact to degree 3).
  All dimensions now route through a single vectorized axis operator (`_compute_hjb_rhs_axis`); the
  duplicated/placeholder 2D/3D direction methods are removed (−232 lines net). New tests assert the
  gradient sign/magnitude, polynomial exactness, convergence order, bounded oscillatory solve, and 2D
  axis-symmetry; the `#1200` `xfail` is removed. `Fixes #1200`.
- **High-order ghost extrapolation fixed at the high boundary** (`PreallocatedGhostBuffer`, surfaced
  by #1200). `_extrapolate_boundary_1d` paired the nearest-first `x_interior = [-dx, -2dx, ...]` with
  `interior_indices` ordered farthest-first at the high boundary, so the Vandermonde fit produced
  badly wrong order>2 ghosts there (the low boundary, where both orderings coincide, was correct).
  Latent because WENO5 is the only order-5 consumer and nothing had consumed the high-boundary ghost
  derivative correctly before; the prior order-5 test checked the low boundary only. Now ordered
  nearest-first to match; smooth high-boundary ghosts recover machine-level accuracy (a `cos(pi x)`
  ghost went from ~1.6e-1 error to ~1e-9), with a both-boundaries regression test added.
- **High-order ghost extrapolation honours a nonzero Neumann flux** (Issue #1186). The order>2
  ghost BC row hardcoded `p'(0) = 0`, so a `neumann_bc(value=g)` with `g != 0` was silently
  dropped (WENO5 HJB is the order-5 consumer). The row now encodes the prescribed `du/dn = g`
  with the outward-normal sign (`p'(0) = -g` at the low boundary, `+g` at the high boundary,
  matching the Robin branch); `NO_FLUX`/`REFLECTING` stay zero-flux. A quadratic with `du/dn = g`
  at both ends is now reproduced to machine precision (regression test added). The order<=2 linear
  reflection path (shared by FDM/SL) is fixed in the companion entry above (#1186 sibling).
- **Conservative no-flux Laplacian for the implicit FP diffusion solve** (Issue #1184, diffusion
  half). `LaplacianOperator.as_scipy_sparse()` no-flux branch had zero *row* sums (2nd-order
  Neumann accuracy) but nonzero *column* sums at the walls (`≈1/h²`); the implicit FP system
  `(I/dt − D·L)` conserves mass iff `1ᵀL = 0`, so it leaked at no-flux walls (a wall-touching
  density lost ~0.84% per solve even at zero drift). Added `LaplacianOperator(mass_conservative=…)`
  (default `False` keeps the byte-identical 2nd-order stencil for HJB/elliptic matvec consumers);
  the explicit-drift FP diffusion path now requests the finite-volume zero-flux stencil
  (`mass_conservative=True`, both row- and column-conservative), so pure diffusion at no-flux walls
  conserves mass to machine precision (was 0.84% → 1.9e-15). The explicit *advection* sub-step's
  non-conservation under strong drift is a separate follow-up (#1184 step 4); `Refs #1184`.
- **Conservative finite-volume advection at no-flux walls** (Issue #1184, advection half). The
  divergence-form upwind advection applied a *node-based* `gradient_upwind` to the flux `v·m`,
  which telescopes in the interior but leaks `±(v·m)` through a no-flux wall — under strong drift
  into a wall the explicit-drift FP solve lost mass catastrophically (a density piled against the
  wall reached mass `0.79` at drift −0.3 and went **negative** (`−0.57`) at drift −0.8). Added
  `AdvectionOperator(mass_conservative=…)` (opt-in; mirrors `LaplacianOperator`): a finite-volume
  flux-difference with velocity-upwinded face fluxes and **zero flux through no-flux/periodic
  boundary faces**, so `1ᵀA = 0` exactly (column-conservative). The explicit-drift FP paths
  (`solve_timestep_explicit_with_drift`, `solve_timestep_tensor_explicit`) opt in: mass is now
  conserved to machine precision under arbitrary wall-directed drift, with no negative densities;
  the scheme self-converges (L1 rate ≈ 1.2–1.6). Default `False` keeps the byte-identical
  node-based divergence for all other consumers (verified bit-for-bit over 19 cases), so there is
  no change to the HJB/geometry advection paths. `Fixes #1184`.
- **WENO HJB now sub-steps each backward interval to cover the full `dt`** (Issue #1180). The
  1D/2D/3D/nD backward sweeps advanced only one CFL/diffusion-stable `dt_stable` per interval
  (often `dt_stable << dt`) while recording it as a full `dt` step, so in the common
  diffusion-limited regime the value function was silently near-frozen at the terminal
  condition (integrated ~2% of the horizon at `sigma=0.3`). Added a shared
  `_advance_full_interval` substep loop (recomputes `dt_stable` per sub-step, fails loud at
  `max_substeps`) wrapping the full directional-split sequence in each branch; happy path
  (`dt_stable >= dt`) is byte-identical. Also fixed the 3-D branch's reference to an unset
  `self.dt` (it raised `AttributeError`). **Behavior change**: the WENO solver now actually
  integrates, so it surfaces (as non-finite, fail-fast) the pre-existing spatial instability
  for oscillatory terminal data tracked in **#1200** — previously masked by the under-integration.
  Cost: per-interval work scales `~dt/dt_stable` in diffusion-limited regimes.
- **Roll-based finite-difference stencils now work on the torch backend** (Issue #1194).
  `finite_difference.py` and `tensor_calculus.py` called `xp.roll(u, k, axis=...)`, but
  `torch.roll` uses `dims=` not `axis=`, so every roll-based stencil (gradient/laplacian/
  divergence/hessian) raised `TypeError` on the torch backend (e.g.
  `test_particle_gpu_pipeline::test_boundary_conditions_gpu`). Added a backend-aware `_roll`
  shim (`dims=` for torch, `axis=` for numpy/cupy); the numpy/cupy path is byte-identical.
- **GFDM QP monotonicity path no longer emits deprecated OSQP kwargs** (Issue #1196). The
  `qp_m_matrix` schemes solve an OSQP problem per collocation point per Newton iteration via
  `polish=False` + bare `prob.solve()`; OSQP 1.0 renamed `polish`→`polishing` and warns
  `raise_error`-default-change on every solve (tens of thousands per run, log-flooding, and a
  latent break when `polish` is removed). Switched to `polishing=False` + explicit
  `raise_error=False` (byte-identical: same fail-soft fallback on non-`solved` status); bumped
  the floor to `osqp>=1.0`.
- **`DualHamiltonian`/`DualLagrangian` Legendre transform now returns the supremum under
  `OptimizationSense.MAXIMIZE`** (Issue #1185). The 1-D `__call__` flipped to the *infimum*
  for MAXIMIZE (the value at a control bound), disagreeing with its own `dp` argmax and the
  d>1 scipy branch — e.g. `L=½·2·α²`, `p=1` returned `-110.0` instead of the correct
  `sup_α{p·α−L}=0.25`. The convex conjugate is a supremum by definition, independent of
  optimization sense; the 1-D branch now always takes the `sup` (matching the d>1 branch).
  Reachable via a non-separable `LagrangianBase` with `sense=MAXIMIZE` passed to
  `MFGComponents(lagrangian=...)`.
- **Callable/tensor explicit-drift FP advection now honors the domain boundary conditions**
  (Issue #1181). `solve_timestep_explicit_with_drift` and `solve_timestep_tensor_explicit`
  called `compute_advection_from_drift_nd` without `bc=`, so on a no-flux domain the
  advection defaulted to periodic and mass exiting one wall silently re-entered at the
  opposite wall (a #1151-class wall leak; the U-derived sibling already passed the BC).
  Reachable via `solve_fp_system(drift_field=<callable>)` or any tensor/anisotropic
  `volatility_field`. Fixed to pass the in-scope `boundary_conditions`; regression test
  asserts a leftward drift on a no-flux domain leaves the far-right (near-wall) region empty.

- **nD ADI diffusion now applies the full prescribed diffusion** (Issue #1178). The
  semi-Lagrangian `adi_diffusion_step` split the time step across dimensions
  (`dt/dimension` per directional Crank-Nicolson sweep), applying only `1/dimension`
  of the diffusion — a silent 2x under-diffusion in 2D, 3x in 3D — on the default
  `diffusion_method='adi'` path for 2D/3D SL HJB and the SL-adjoint FP. Sequential
  (Lie) splitting requires the full `dt` per directional solve (the directional
  Laplacians commute on a tensor grid). Fixed to full `dt`; added a magnitude-pinning
  regression test (cosine-mode decay vs analytic, 2D + 3D) — no prior test constrained
  the diffusion magnitude, which is how it shipped.

## [0.19.8] - 2026-06-04

### Added

- **Coupled-MFG-vs-reference MMS test** (`tests/integration/test_coupled_mfg_mms.py`).
  Closes a critical verification gap: every prior coupled-MFG test checked only Picard
  self-consistency residual (→0 for *any* fixed point) and mass conservation (~1 for any
  conservative scheme) — neither verifies the converged `(u_h, m_h)` is the *correct*
  solution, which is how the σ→D factor bug (#1152) and the no-flux wall-leak (#1151)
  shipped. The test manufactures a smooth periodic pair with active bidirectional
  coupling, injects analytic `S_HJB`/`S_FP` (3-way independently derived + cross-checked)
  so `(u*, m*)` is the exact source-augmented solution, runs the real `FixedPointIterator`,
  and asserts the empirical convergence order of *both* fields — a wrong diffusion factor,
  coupling sign/coefficient, or non-conservative flux breaks the rate even though Picard
  still converges and mass is conserved. Marked `slow` (~2 min; runs on merge/release).

- **`HJBGFDMSolver(inner_solver='howard')` now supports adjoint-consistent Robin BC**
  (Issue #1118 PR2b). `BCType.ROBIN(alpha=0, beta=1)` — the `AdjointConsistentProvider`
  pattern whose resolved scalar is `g = -sigma^2/2 * d ln(m)/dn` — is routed through the
  shared `_build_neumann_bc_row` (the equation reduces to `n.grad u = g`), so both the
  Newton and Howard inner solvers honor it from a single coefficient source. The Howard
  guard admits `"robin"`; the alpha/beta check lives only in the row builder. Combined
  with the per-solve BC refresh, the per-Picard resolved value now reaches the Howard
  value-form rows. Reachable-but-unsupported forms fail loud: `ROBIN(alpha != 0)` and
  `ROBIN(beta != 1)` raise `NotImplementedError` (the normal-derivative row cannot
  represent `alpha*u` and does not apply a `1/beta` scaling); an unresolved
  `BCValueProvider` reaching the solver raises `AssertionError` (the coupling layer must
  resolve it via `using_resolved_bc` first).

- **`HJBGFDMSolver(inner_solver='howard')` honors prescribed Dirichlet values and the
  real Neumann normal-derivative stencil** (Issue #1118 PR2a). The Howard inner solver
  previously hardcoded the Dirichlet RHS to 0 and approximated Neumann by a
  nearest-interior copy; both now flow through the shared `_value_form_bc_rows` /
  `_bc_row_for_point` single coefficient source, so a nonzero `g_D` and the true
  `n.grad u = g` stencil are applied (and the Newton/Howard paths share one BC-row builder).

### Changed

- **`MFGProblem` rejects non-positive `T`** (Issue #1077 case 4). `T <= 0` previously
  produced a degenerate time grid silently; it now raises at construction.

- **Anisotropic non-diagonal σ-tensor warning strengthened** (Issue #1079). The warning
  now states the dropped off-diagonal cross-derivatives are an O(1) error (not
  higher-order) in the affected HJB-FDM path. The diagonal-approximation behavior is
  unchanged (deliberate, codified by `test_non_diagonal_tensor_warning`).

### Fixed

- **Parameter sweep no longer crashes on macOS `spawn`** (Issue #1080). `ParameterSweep`
  carried a `threading.Lock` (via its logger) into the pickled payload for
  `ProcessPoolExecutor`; `__getstate__`/`__setstate__` now drop the logger, and a
  pre-flight `pickle.dumps` probe fails loud with a clear message instead of an opaque
  spawn error.

- **Multi-population HJB now sees cross-population densities** (Issue #1157).
  `MultiPopulationIterator` computed the cross-density bound Hamiltonian but never
  passed it to `solve_hjb_system`, so each population's HJB solved against the
  uncoupled `problem.hamiltonian_class` — the coupling reached the FP drift but not
  the value function, a silently wrong half-coupled equilibrium. `solve_hjb_system`
  now accepts a `hamiltonian_override`; `HJBFDMSolver` threads it through the batch
  Hamiltonian path (forcing `backend=None`, which is numerically equivalent to the
  per-point path) so the stacked density field reaches the Hamiltonian. The
  iterator sends the override only to backends that honor it (FDM) and **fails loud**
  for K>1 on backends that do not, rather than silently decoupling; single-population
  (K==1) runs are byte-identical (no override is sent). `BoundHamiltonian` now
  time-indexes a `(Nt+1, K*N)` density trajectory by the evaluation time
  `t -> n = round(t/dt)`, so each backward step sees the cross-density at that step.

- **`HJBGFDMSolver` now re-reads geometry boundary conditions per solve** when
  the BC was sourced from the geometry (Issue #1118). GFDM previously snapshotted
  the BC and its preclassified per-point segment map at construction, so the
  coupling layer's per-Picard `using_resolved_bc` swap of
  `geometry.boundary_conditions` never reached the solver — freezing any resolved
  value (e.g. `AdjointConsistentProvider`'s `g = -sigma^2/2 * d ln(m)/dn`) at its
  construction-time value, a silent error in the boundary-stall regime. FDM
  already re-read BC each solve; GFDM now matches via
  `_refresh_boundary_conditions_if_changed()` at the top of `solve_hjb_system`.
  No-op for explicitly-passed `boundary_conditions=` (static) and for
  provider-free BC (object identity unchanged). Prerequisite for the
  Robin/adjoint-consistent Howard inner-solver support that follows.

## [0.19.7] - 2026-06-03

### Added

- **`HJBGFDMSolver(inner_solver='howard')`** — opt-in delegation of the backward
  HJB sweep to `HJBHowardSolver` (Issue #1118, PRs #1165/#1166). The default
  `inner_solver='newton'` is byte-identical; `'howard'` avoids the Armijo
  `MIN_ALPHA` stall (policy iteration has no line search) by deriving the optimal
  control `α* = -hamiltonian_class.dp`. Restricted to `joint_socp`+`precompute`
  stencils, unit control cost, and homogeneous no-flux BC; fail-loud otherwise.
  Dirichlet-value / nonzero-Neumann / Robin BC parity is deferred to a follow-up.

- **`HJBHowardSolver`** — Howard's policy iteration inner solver for HJB on
  GFDM clouds (Issue #1118). Replaces the Newton inner loop of
  `HJBGFDMSolver._solve_timestep` when the Hamiltonian is strictly convex
  in `p`. Resolves the temporal-plateau symptom where Armijo backtracking
  bottoms out at `MIN_ALPHA = 1e-6` and Newton makes no net progress past
  the first few backward steps.

  Graduates the 5-fork research-side `howard_patch_*` family
  (`mfg-research/experiments/gfdm_monotonicity_audit/minors/{exp08, exp09,
  exp11}/`) into a single peer class to `HJBGFDMSolver`. Composition-based:
  takes a constructed `HJBGFDMSolver` with `monotonicity_scheme="joint_socp"`
  + `monotonicity_application="precompute"` as `stencil_provider`, plus a
  `alpha_star(x, p, m, t) -> alpha` Legendre callable. Three discretisation
  options: `upwind_projection` (default, projection onto α direction),
  `upwind_per_axis` (per-axis sign-aware Dpos/Dneg pair), `central` (bare
  central, accurate on smooth problems; not monotone for advection-dominant).

  Convergence hypothesis: H strictly convex in p (Bokanowski-Maroso-Zidani
  2009). Separability is neither necessary nor sufficient.

  Reported ~57× speedup over Newton inner on irregular 2D clouds with
  11-15% k-NN-fallback stencils (per exp09 Phase 7 readme). Eight unit
  tests cover construction validation, 1D LQ Riccati closed-form
  (`P(0)/P(T) = 0.5` per mfgarchon HJB convention — see
  `mfg-research/docs/archon-notes/development/guides/NAMING_CONVENTIONS.md`
  § HJB Equation Conventions), Newton-stall reproducer, each discretisation
  option, and 2D integration with running-cost callable. Howard is a peer
  to `HJBGFDMSolver`; the outer `FixedPointIterator` (Picard / fictitious
  play) is unchanged — Howard replaces inner Newton only.

- **`compute_geodesic_distance` and `build_geodesic_field`** in
  `mfgarchon/geometry/cloud_geodesic.py` (Issue #1093). Geodesic distance
  on meshfree clouds via Dijkstra on a k-NN graph with segment-sample
  obstacle filtering, companion to the structured-grid Eikonal solvers
  at `mfgarchon/geometry/level_set/eikonal/`.

  Graduates `mfg-research/experiments/gfdm_monotonicity_audit/minors/exp09_obstacle_navigation_full/geodesic_distance.py`
  to library status. Two functions:

  - `compute_geodesic_distance(points, sources_idx, obstacles_sdf=None, k_neighbors=25, ...)`
    returns `(N,)` geodesic distances. Edges crossing obstacles are
    excluded via `n_segment_samples` SDF samples per edge. `obstacles_sdf`
    follows the mfgarchon SDF convention (`sd <= 0` inside obstacle, see
    NAMING_CONVENTIONS.md § Geometry SDF Convention). `np.inf` for points
    unreachable through navigable region. `O(N · k)` graph build +
    `O((N + E) log N)` Dijkstra.
  - `build_geodesic_field(points, d_geodesic, unreachable_penalty=1.5)`
    wraps per-point distances into a callable `g(x)` via
    `LinearNDInterpolator` with `NearestNDInterpolator` fallback for
    out-of-hull queries. Unreachable points (`np.inf`) are substituted
    with `unreachable_penalty × max(finite d)` so the field stays finite
    for downstream HJB use.

  Primary use case: terminal cost `u(T, x) = 0.5 · G_s · g(x)²` for
  obstacle-navigation problems. Bakes routing into `g(x)` so the HJB
  solver does not need visibility-aware gradient operators or soft-wall
  potentials to encode it.

  12 unit tests cover obstacle-free correctness (Euclidean within 15%
  for k=24), triangle inequality on the graph, multiple sources →
  `min_s d(., s)`, obstacle inflates geodesic above Euclidean,
  unreachable points return `np.inf`, argument validation, and
  field-builder round-trip + out-of-hull fallback + penalty substitution.

- **`preserve_indices=False` flag on `FPParticleSolver`** (Issue #1119).
  When `True`, absorbed particles are NaN-marked in the per-step trajectory
  rather than compact-removed, so `particle_history[t].shape == (num_particles, d)`
  is constant across all timesteps and original particle indices are preserved
  across absorption events. Enables follow-individual-particle trajectory plots
  in evacuation experiments without NN-matching artifacts. Default `False`
  preserves the legacy compact-array behavior bit-for-bit. Currently supported
  only in the callable-drift n-D path with segment-aware (Dirichlet) BC; other
  paths raise `NotImplementedError` when the flag is set.

- **HJB-FP volatility consistency check in `FixedPointIterator`** (Issue #1082).
  Warns when `volatility_field=X` is passed AND `problem.sigma=Y` with
  `X != Y` (scalar case). HJB sees Y, FP sees X — Picard fixed point not
  a coherent MFG. Same trap pattern as #811. Silent for callable / matched.

- **Empirical per-stencil M-matrix verification tests for joint_socp**
  (Issue #1074, partial). New `tests/unit/test_alg/test_socp_m_matrix_property.py`
  verifies the 4 stencil-level invariants the paper claim depends on, across
  σ ∈ {0.5, 1.0, 1.5}: Laplacian consistency `sum(L)=0`, off-diagonal
  non-negative `L[off] ≥ 0`, center non-positive `L[center] ≤ 0`, and the
  per-edge cone bound `‖D[:,j]‖ ≤ (C/h_i) · L[j]` (the non-trivial constraint
  that closes the discrete comparison principle proof). Full assembled-matrix
  M-matrix verification deferred (depends on dt + advection regime).

### Changed

- **Fail-loud config & coupling guards** (Issues #1081/#1154/#1156). `FixedPointIterator`
  warns on an HJB–FP volatility mismatch and when the HJB Newton tolerance is looser than
  the Picard tolerance (a convergence floor); the silent `σ=0.1` default in `fp_particle`
  was removed. `load_solver_config` raises on unknown top-level keys; `solve()` raises when
  `config.hjb`/`config.fp` cannot be honored by the Safe/Auto factory paths.

- **`gradient_*` FP advection schemes documented + warned as non-conservative** (Issue
  #1075). They leak mass at no-flux walls even at zero drift (boundary diffusion
  discretization, not advection); a `UserWarning` steers callers to the conservative
  `divergence_*` default. Corrected the false "`gradient_upwind` is conservative via row
  sums" docstrings.

- **`FPSLSolver` warns once when its positivity clip injects mass** (Issue #1153),
  mirroring the weak-form FP clip warning.

- **`TaylorOperator` now accepts `obstacle_sdf=` / `visibility_samples=` /
  `visibility_margin=`** (Issue #1124). When provided, stencil edges
  crossing the obstacle region are excluded at operator-construction
  time, so the pre-assembled `D_lap` / `D_grad` sparse matrices respect
  domain connectivity. Same convention as `NeighborhoodBuilder.obstacle_sdf`:
  `obstacle_sdf(x) < 0` means inside the obstacle region (pass the SDF
  of the obstacle, not the navigable domain). Wired through
  `HJBGFDMSolver.__init__` so `HJBGFDMSolver(obstacle_sdf=...)` now
  filters both layers (operator-level `D_lap` / `D_grad` AND the
  post-adaptive `NeighborhoodBuilder.neighborhoods` view).

  Pre-this-fix, `obstacle_sdf=` reached only `NeighborhoodBuilder`. The
  bulk linear operator was constructed without it, so wall-crossing
  edges remained in `D_lap` / `D_grad` regardless of the documented
  visibility-filter behavior. Symptom in 2D thin-wall geometries
  (`delta` exceeds wall thickness): HJB-backward-integrated `U(t=0)`
  inverts in dead corners versus door bands even with correct geodesic
  terminal cost (see issue body §Reproducer).

  Counter `op._visibility_filtered_count` records how many stencil
  edges were blocked; constructor emits a `UserWarning` when filtering
  starves any stencil below `n_derivatives` (Taylor LSQ falls through
  to `None` at those points — consider increasing `delta` or relaxing
  `visibility_margin`).

- **CFL diagnostic logging now emits at INFO once per solver instance**, then
  DEBUG on subsequent calls (Issue #1052). Previously every Picard iteration
  emitted the same "CFL diagnostic" line at INFO, spamming user logs and
  causing researchers to blanket-suppress warnings (which masked unrelated
  DeprecationWarnings — the Tier-C silent-semantic-shift bugs in #1043 went
  unnoticed for weeks partially for this reason). Applies to `HJBFDMSolver`
  and `FPFDMSolver`.

- **`SeparableHamiltonian.potential` docstring** now explicitly documents the
  "potential as reward" sign convention (Issue #1057, gotcha G-001). For an
  attractive potential at `x_c`, write `V(x, t) = -0.5*C*(x-x_c)**2`
  (inverted parabola, peak at `x_c`); for repulsive, write `+0.5*C*(x-x_c)**2`
  (bowl). This is opposite to standard MFG literature where V is "cost to
  avoid"; mfgarchon's convention is reward, agents concentrate at V_max.

### Changed

- **`JAXBackend` JIT cache uses explicit None-init pattern** (Issue #1068, partial).
  Replaced 4 `hasattr(self, "_jit_*")` duck-typing checks with explicit
  `is None` initialization in `__init__`. Per CLAUDE.md "Object Shape
  Stability". Other #1068 hasattr clusters (core/mfg_components, types/protocols)
  deferred — those need Protocol/ABC design.

### Removed (BREAKING)

- **`FixedPointIterator` legacy `damping_*` kwargs and attribute aliases**
  (Issue #1070, v0.25.0 milestone). The 7 ctor kwargs
  (`damping_factor`, `damping_factor_M`, `adaptive_damping`,
  `adaptive_damping_decay`, `adaptive_damping_min`, `damping_schedule`,
  `damping_schedule_M`) and their 7 read-only `@property` aliases were
  deprecated in v0.19.2 and are now removed per the 3-version deprecation
  window. Migration: rename to the canonical `relaxation_*` /
  `adaptive_relaxation_*` names (one-to-one mapping).

  Enforced at construction by `mfgarchon.utils.deprecation.validate_kwargs`
  with a class-level `_REMOVED_KWARGS` migration map (matching the
  `MFGProblem._validate_kwargs` pattern). Passing any removed kwarg raises
  `ValueError` with a curated "Use 'X' instead (v0.25.0 removal, Issue #1070)"
  message rather than Python's generic "unexpected keyword argument".

  Unaffected: users on the high-level `PicardConfig(damping_factor=...)`
  path — `PicardConfig` has its own independent deprecation map at the
  config layer (`mfgarchon/config/core.py`) which translates to canonical
  names before passing to `FixedPointIterator`. That config-layer
  deprecation continues to warn (separate removal track).

  Migration test file: `tests/unit/test_alg/test_fixed_point_iterator_relaxation_alias.py`
  rewritten from "verify the redirect" to "verify removal" — 22 gate
  tests lock in the `ValueError` raise + `AttributeError` on attribute
  read + canonical kwargs still accepted.

  Cluster B of Issue #1070 (`HJBGFDMSolver` deprecated `NiterNewton` /
  `l2errBoundNewton` / `qp_optimization_level`) deferred to a follow-up
  PR — that cluster requires migrating ~20 mfg-research scripts that pass
  `qp_optimization_level=` directly.

- **`PrecomputedMonotoneStencils` / `PrecomputedJointSocpStencils` legacy
  ctor signature** (Issue #1102, dual-source stencil bug class). The
  `operator: TaylorOperator` parameter is removed; `neighborhoods=`,
  `points=`, `delta=` (for `PrecomputedMonotoneStencils`) and
  `neighborhoods=` (for `PrecomputedJointSocpStencils`) are now required
  kwargs. The legacy fallback path that read pre-adaptive
  `op.get_derivative_weights()` / `op.get_neighborhood()` is deleted.

  Motivation: the bug class recurred twice (#1099 → JointSocp,
  #1102/#1121 → Monotone). Both incidents were the same shape: stencil
  weights computed against pre-adaptive `op.neighborhoods`, contracted at
  runtime against `b = u_neighbors - u_center` built from post-adaptive
  `NeighborhoodBuilder.neighborhoods`, raising
  `ValueError: matmul: size N is different from K` at corner buffer
  points where adaptive-δ enlargement modified the stencil. After this
  change the bug class is statically impossible: there is no way to
  construct a stencil object that silently drifts from runtime
  neighborhoods.

  Migration (production callers — already done in
  `HJBGFDMSolver.__init__` lines 920, 978):
  ```python
  # before (v0.24)
  PrecomputedMonotoneStencils(operator=op, is_boundary=mask, ...)
  PrecomputedJointSocpStencils(operator=op, points=pts, interior_indices=..., delta=δ, ...)

  # after (v0.25.0)
  PrecomputedMonotoneStencils(is_boundary=mask, neighborhoods=nh, points=pts, delta=δ)
  PrecomputedJointSocpStencils(points=pts, interior_indices=..., delta=δ, neighborhoods=nh, ...)
  ```

  Tests at `tests/unit/test_alg/test_precomputed_monotone_stencils.py`
  rewritten: legacy-path test deleted, TypeError gates added for each
  required kwarg, matched + enlarged paths kept, integration regression
  unchanged. 7/7 pass.

### Fixed

- **Weak-form FP `volatility_field` is σ, not D** (Issue #1152). FEM/meshless FP and HJB
  treated a scalar/array `volatility_field` as the diffusion coefficient directly, skipping
  `D = σ²/2` (~6.7× too-large diffusion); both branches now square it.

- **`divergence_centered` FP conserves mass at no-flux walls** (Issue #1151). The boundary
  handler evaluated the shared-face velocity one-sided while the interior used a central
  stencil → double-valued face flux → leak; now uses the central face velocity (one-sided
  only at corners). Verified to ~1e-15.

- **`Hypersphere` fails fast on a non-finite radius** (Issue #1077). `inf`/`nan` slipped past
  the `radius <= 0` check, producing an infinite bounding box → NaN rejection sampling.

- **Duality-claim honesty** (PR #1158). `check_solver_duality` / `_create_fdm_pair` no longer
  claim the default FDM pair (gradient_upwind HJB + divergence_upwind FP) is a bit-exact
  transpose; the exact transpose remains opt-in via `adjoint_mode='jacobian_transpose'`.

- **`PrecomputedMonotoneStencils` accepts `neighborhoods=` parameter**
  (Issue #1102). Pre-fix, the class built stencils on
  `op.get_derivative_weights(i)` (pre-adaptive, e.g. 53 neighbors on a corner
  buffer point) while the runtime override site in
  `HJBGFDMSolver.approximate_derivatives` contracted against
  `self.neighborhoods[i]["indices"]` (post-adaptive, e.g. 522 after
  `adaptive_neighborhoods=True` enlargement). The size divergence raised
  `ValueError: matmul: size N is different from K` and forced the b-rebuild
  workaround in commit `67fa5ad8`, which silently dropped adaptive-enlarged
  neighbors from the Phase-2 Laplacian correction.

  Fix: the constructor now accepts `neighborhoods`, `points`, and `delta`
  (mirroring `PrecomputedJointSocpStencils`, joint_socp.py:491). When
  provided, stencils are built on post-adaptive indices with Wendland-LSQ
  unconstrained Laplacian weights recomputed on the enlarged stencil, then
  passed through the existing M-matrix QP. The wired call in
  `HJBGFDMSolver` (hjb_gfdm.py:985) now passes `neighborhoods=self.neighborhoods`.
  Closes the second instance of the [[feedback_mfgarchon_dual_stencil_bug]]
  recurrence (first instance: `PrecomputedJointSocpStencils`, #1099).
  Adds the first dedicated test file for `PrecomputedMonotoneStencils`
  (audit 2026-05-10 D.1 gap), exercising legacy / matched-indices /
  enlarged-indices / integration regression paths.

- **`HJBGFDMSolver` diffusion-term arithmetic in `scheme="none"` path**
  (Issue #1073). Four sites in `hjb_gfdm.py` (residual_vectorized at L1840,
  residual_hamiltonian at L1889, jacobian_hamiltonian at L1926,
  jacobian_vectorized at L2056) sourced σ via the chain
  `getattr(self.problem, "diffusion", 0.0) or getattr(self.problem, "sigma", 0.0)`.
  Because `problem.diffusion` returns `σ²/2` (the PDE coefficient `D`) and
  is truthy whenever σ > 0, this resolved σ to `D`, then computed
  `0.5 · D² · Δu = (σ⁴/8) · Δu` instead of `D · Δu = (σ²/2) · Δu`.

  Ratio of buggy/correct = `σ²/4`:

  | σ | ratio | severity |
  |---|---|---|
  | 0.3 | 0.022 | 44× too small (paper Stage 3 high-Pe regime) |
  | 0.5 | 0.063 | 16× too small |
  | 1.0 | 0.250 | 4× too small |
  | 1.414 | 0.500 | 2× too small |
  | 2.0 | 1.000 | accidentally correct |

  Fix: replace all 4 sites with `self._get_sigma_value(None)` (same pattern
  already used correctly by `_compute_hjb_residual_with_cache` at L2024).

  **Active path**: only when `monotonicity_scheme="none"` (the default).
  QP/SOCP modes (`joint_socp`, `qp_m_matrix`) take a different code path
  via `_compute_hjb_residual_with_cache` and were always correct. So:
  - Tutorial / default-scheme users at σ ≠ 2 were getting wrong diffusion
  - Production paper experiments using `joint_socp` were unaffected
  - σ=2 is the only value where the bug coincidentally cancels

  Same trap pattern as Issue #811 (`MFGProblem(diffusion=...)` vs
  `sigma=`); cross-references same docstring at `core/mfg_problem.py:1306-1317`.
- **Picard NaN/Inf diagnostic now identifies HJB vs FP source** (Issue #1078).
  Previously `fixed_point_iterator.py:804` (Issue #688 fix) emitted a generic
  "NaN/Inf detected" warning when terminating early on non-finite iterates,
  with no indication of which side blew up. Now examines `U_new` / `M_new`
  (still in scope from earlier in the loop iteration) and labels the source
  as `HJB (Newton divergence)`, `FP (density blow-up)`, `both`, or
  `post-damping (likely Anderson acceleration)`. Five-line change, no new
  control flow.
- **`FPParticleSolver` meshfree KDE now uses reflection ghosts on reflecting
  BC axes** (Issue #1083). Previously `fp_particle.py:2026-2029` constructed
  `ParticleDensityQuery` without `reflect_bounds`, so boundary cells were
  underestimated by ~50% (per `particle_density_query.py:558` known limit).
  For Towel-on-Beach Gaussian with stall near the wall, this biased the next
  Picard iteration's drift, producing a wrong fixed point.

  New helper `_infer_reflect_bounds()` examines `self.boundary_conditions`
  and returns the bounds list when at least one segment is `NO_FLUX` /
  `REFLECTING` / `NEUMANN`. Per-axis disambiguation is deferred until BC
  framework exposes segment→axis mapping.- **`enforce_obstacle_boundary` no longer captures particles past the outer
- **`np.linalg.inv()` → `np.linalg.solve()` in 2 hot paths** (Issue #1066,
  partial — neighborhood_builder cache deferred). `joint_socp.py:193`
  computed `ATA_inv` then matmul'd with `e_grad[d]` in a Python loop;
  now uses a single `solve(ATA, [e_lap|e_grad].T)` (kills Python loop +
  squares fewer condition numbers). `sampling.py:736` Mahalanobis used
  `inv(cov)` (silently wrong for cond > 1e10); now uses `solve(cov, ...)`
  + `einsum`. Third site `neighborhood_builder.py:744` deferred
  (long-lived cache needs `lu_factor`/`lu_solve`).- **`enforce_obstacle_boundary` no longer captures particles past the outer  bounding box** (Issue #1064). When `FPParticleSolver` is configured with
- **`TensorProductGrid` validates `Nx_points >= 1` + finite/ordered bounds**
  (Issue #1077, partial). `Nx_points=[10, 0, 5]` and `bounds=[[1, 0]]` (lo > hi)
  now raise `ValueError`. N=1 (single-point grid, zero spacing) preserved.
  Other input-validation cases in #1077 deferred.- **`enforce_obstacle_boundary` no longer captures particles past the outer  bounding box** (Issue #1064). When `FPParticleSolver` is configured with  both `implicit_domain` (for obstacle reflection) and a `BoundaryConditions`
  containing a Dirichlet (absorbing) segment on the outer boundary,
  `enforce_obstacle_boundary` was projecting **all** particles outside the
  navigable region back inside — including those that had crossed the
  Dirichlet exit segment. The segment-aware BC then never saw them, so
  `total_absorbed` stayed at 0 and absorbing exits were silently disabled.

  The fix discriminates by bounding-box membership: only particles **inside
  the outer bbox** but in an obstacle interior get re-projected. Particles
  **past the outer bbox** are an outer-boundary concern and are left for
  the caller's segment-aware BC (which handles reflect / absorb / wrap per
  segment). Composes correctly with #1042 (callable-drift segment-aware
  routing).

- **`HJBSemiLagrangianSolver._stochastic_sl_step_nd` companion fixes**
  (Issue #1054): apply the analogous trio of correctness fixes to the nD
  stochastic SL path:
  1. **Monotone interpolation**: when `interpolation_method ∈ {"cubic",
     "quintic"}`, route through `RegularGridInterpolator(method="pchip")`
     (tensor-product monotone Hermite, scipy ≥ 1.10) instead of the
     non-monotone tensor-product cubic. Mirrors 1D Issue #1033.
  2. **Per-axis BC handling on Brownian feet**: apply iterated mirror
     reflection (`reflect`) or modular wrap (`wrap`) per axis to
     `y_plus`/`y_minus` before interpolation. Previously the nD path
     silently extrapolated via `bounds_error=False, fill_value=None`,
     producing values dependent on the nearest interior cell rather than
     respecting the SDE's reflection/periodicity. Mirrors 1D Issue #1048.
  3. **Vectorized batch interpolation**: replace per-(node, axis)
     `_interpolate_value` calls (which rebuilt the interpolator each call)
     with a single `RegularGridInterpolator` built once and queried on the
     full `(2*d*N_total, d)` departure batch. Linear interpolation
     continues to work alongside stochastic dispatch (Issue #1049 carries
     through).

- **`HJBSemiLagrangianSolver._stochastic_sl_step_1d` trio of fixes** (Issues
  #1033, #1048, #1049):
  1. **#1033**: replace `scipy.interpolate.CubicSpline` (non-monotone, blew up
     on stiff problems with `max|∇u|` exponential growth 6 → 100 → 10⁶ → NaN
     on 1D Towel-on-Beach in 17 Picard iters) with `PchipInterpolator`
     (monotone Hermite). Linear interpolation now uses `np.interp` directly
     when `interpolation_method="linear"`.
  2. **#1048**: replace `np.clip(y, xmin, xmax)` boundary handling with
     iterated mirror reflection `xmin + |((y − xmin) mod 2L) − L|`. Clamping
     collapsed all out-of-bounds characteristic feet onto the boundary node,
     biasing toward wall values and breaking upwind property near reflective
     boundaries. Reflection matches the underlying SDE's behavior for Neumann.
  3. **#1049**: remove the validation that **rejected** `interpolation_method
     ="linear"` with `diffusion_method="stochastic"` — that combination IS the
     proven-stable Carlini-Silva 2014 canonical scheme; the previously-required
     `cubic` is non-monotone and outside the stability proof. Now `linear` is
     the unwarned default for stochastic; cubic/quintic emit a `UserWarning`
     pointing to the proof status.

  See `mfg-research/docs/mfgarchon_gotchas.md` G-008 / G-009 / G-010 for the
  research-side audit. The dim-agnostic refactor (unifying `_stochastic_sl_step_1d`
  and `_stochastic_sl_step_nd` per the project's "dimension as parameter, not
  constraint" principle) is tracked separately as Issue #1050; this PR fixes
  the 1D path only.
- **`FPParticleSolver._get_grid_params` now fails fast on geometries that
  expose neither `.bounds`, `.xmin`/`.xmax`, nor `.coordinates`** (Issue #1053).
  Previously fell through to a silent `[(0.0, 1.0)] * dimension` fallback
  (the unit hypercube), which corrupted FP particle simulation on any
  non-standard geometry without a clear error. Now raises `TypeError` with
  a diagnostic pointing at the missing API.
- **`ImplicitDomain.project_to_domain(method='simple')` now uses Newton-on-SDF
  as a fallback** when the original line-search-toward-bbox-center fails
  (Issue #1047). Previously the line-search exhaustion path silently teleported
  particles to the bounding-box center — geometrically incorrect (e.g. for a
  navigable region with an off-center obstacle, every failure-to-project
  collapsed particles to one point, producing KDE singular covariance
  downstream with no clear diagnostic). Now: line search first; on failure,
  Newton iteration `x ← x − φ(x)·∇φ(x)/|∇φ|²` (uses `sdf_gradient` from
  `sdf_utils`); if both fail (degenerate gradient or non-converging), raises
  `RuntimeError` with diagnostic instead of silent corruption. Fail-fast.- **`FPParticleSolver._solve_fp_system_callable_drift` now honors segment-aware
  Dirichlet absorbing boundary conditions** (Issue #1042). Previously the
  callable-drift path always routed through `_apply_boundary_conditions_nd`
  (uniform topology BC) and ignored `boundary_conditions=BoundaryConditions(segments=[...])`
  with Dirichlet exit segments. Particles approaching exits piled up indefinitely
  instead of being absorbed; the grid-based-drift path correctly routed through
  `_apply_boundary_conditions_segment_aware`, but the callable-drift path bypassed
  it. The fix mirrors the grid-drift segment-aware branching: per-step variable
  particle count via list storage, Dirichlet absorption applied via
  `_apply_boundary_conditions_segment_aware`, exit-flux tracking populated
  (`exit_flux_history`, `total_absorbed`). Verified by trajectory storage type
  (now `list` for segment-aware vs `ndarray` for uniform).

- **CSG composite domains (`UnionDomain`, `IntersectionDomain`, `DifferenceDomain`,
  `ComplementDomain`) now expose `.bounds`** (Issue #1041), mirroring the
  `Hyperrectangle.bounds` API. Previously they had only `get_bounding_box()`,
  causing `FPParticleSolver._get_grid_params` to silently fall back to the
  unit hypercube `[(0, 1)] * d` when reading `geom.bounds`. On non-unit
  domains (e.g., `[0, 18] × [0, 8]`) particles got reflected/clipped against
  the wrong domain after every FP step → KDE singular covariance downstream.
  After the fix, FPParticleSolver reads the actual domain bounds end-to-end.
  `ComplementDomain.bounds` is a property delegating to `get_bounding_box()`
  (raises if not manually set, since `ComplementDomain` is unbounded by
  default — fail-fast is correct here).

### Added

- **`HJBGFDMSolver` now emits a `UserWarning` when `monotonicity_scheme` is
  unspecified** (Issue #1034). The default resolves to `"none"` (no QP
  correction), producing bare Wendland-Taylor LSQ stencils whose M-matrix
  structure is not enforced. On long-time-horizon problems (e.g. 1D
  Towel-on-Beach at T=8) this destabilizes FP-Particle coupling and produces
  catastrophic boundary oscillation. The warning surfaces the trap and points
  users to `monotonicity_scheme='joint_socp'` (paper-canonical) or
  `'qp_m_matrix'` (cheaper). Users intentionally using the bare scheme can
  pass `monotonicity_scheme='none'` explicitly to suppress the warning.
  Validated in
  `mfg-research/.../exp08_towel_2d_validation/_preflight_1d/post_mortem_1d_tob_debug.md`.

### Changed

- **Documented `HJBGFDMSolver.obstacle_sdf` sign convention** (Issue #1038).
  Convention: ``obstacle_sdf(x) < 0`` means "x is INSIDE the obstacle (to be
  filtered)". This matches a single-obstacle ``Hypersphere``/``Hyperrectangle``
  ``.signed_distance`` natively but is **inverted** for a CSG composite like
  ``DifferenceDomain.signed_distance`` (which uses the standard navigable-region
  convention). Pass ``obstacle.signed_distance`` directly, not
  ``domain.signed_distance``. Docstring example added in both
  ``HJBGFDMSolver.__init__`` and ``NeighborhoodBuilder.__init__``.

### Fixed

- **`ImplicitDomain.num_spatial_points` now caches the result** (Issue #1037).
  Previously the property recomputed from an unseeded Monte-Carlo volume
  estimate on every call, returning slightly different values across calls
  within one process. Downstream callers like
  `MFGComponents._setup_custom_initial_density` that pre-allocate based on
  the value and then iterate the spatial grid would overrun and surface an
  unhelpful `IndexError: index N out of bounds for size N` (Issue #1036,
  obsoleted by this cache fix).

## [0.19.6] - 2026-05-06

### Fixed

- **`HJBGFDMSolver.approximate_derivatives` now consults precomputed
  monotonicity-corrected weights (J/r consistency)** when the slow path is
  taken. Before this fix, when `monotonicity_scheme="joint_socp"` (or legacy
  `qp_optimization_level="precompute"`), the per-point HJB Newton path used
  inconsistent stencil weights:
    - **Jacobian**: assembled from `_cached_derivative_weights[i]` — populated
      with SOCP / M-matrix-QP weights at __init__ (PR #1030 fix).
    - **Residual**: computed by `approximate_derivatives` slow path, which
      used the bare Wendland-Taylor LSQ (`taylor_data["AtWA_inv"]` etc.),
      *bypassing* any precomputed monotonicity correction.

  Newton then solved `J · δu = -r` with `J` and `r` derived from different
  stencil weights, converging to a stationary point of the mongrel system
  rather than the true discrete-HJB fixed point. Empirically: at the exp08
  step 4 2D Towel-on-Beach validation N=100, raw `‖U_HJB,centered‖₂` at
  iter 1 was 244 with the inconsistency vs ~130 after the fix (47%
  reduction). At N=75 the inconsistency was tolerable (28% of nodes had
  bare W-T in BOTH J and r since SOCP was infeasible there); at finer h
  with higher SOCP coverage, the inconsistency dominated.

  The fix overrides gradient and Laplacian-trace entries in the multi-index
  derivative dict with values computed from the precomputed weights, only
  for nodes that have a precomputed stencil. Behavior is unchanged for
  nodes without a precomputed stencil and for `monotonicity_scheme="none"`
  (which routes through the fast path of `approximate_derivatives`).

### Notes

- This is a correctness fix, orthogonal to the user-visible API. No
  deprecation, no parameter changes. The 16 equivalence tests for the
  v0.18.0 `qp_optimization_level` rename continue to pass with bit-identical
  weights.

## [0.19.5] - 2026-05-06

### Added

- **Two-axis monotonicity API on `HJBGFDMSolver`** (PR #1030):
  - `monotonicity_scheme: "none" | "qp_m_matrix" | "joint_socp"` — what kind of
    constraint to enforce on stencil weights.
  - `monotonicity_application: "adaptive" | "always" | "precompute" | None` —
    when/how it is enforced (per-point QP at runtime vs. precomputed at
    construction).
  - Replaces the legacy `qp_optimization_level=` bundled parameter; equivalence
    is bit-identical, covered by 12 tests in
    `tests/unit/test_alg/test_hjb_gfdm_monotonicity_scheme_rename.py`.
- **First-class `monotonicity_scheme="joint_socp"` option** — precomputes
  joint SOCP-constrained weights (M-matrix on $-\Delta_h$ + per-edge cone
  $\|D_j\|_2 \le C\,h_i\,L_j$) at construction. Includes a Wendland-LSQ
  fast-path (paper Theorem `thm:joint_socp_feasibility`) and a CLARABEL
  CVXPY fallback. Replaces the research-side `patch_operator` monkey-patch
  workflow used through v0.19.4.
- **New module `mfgarchon/alg/numerical/gfdm_components/joint_socp.py`** with
  `PrecomputedJointSocpStencils`, mirroring `PrecomputedMonotoneStencils`.

### Deprecated

- **`qp_optimization_level=`** parameter on `HJBGFDMSolver`. Still accepted
  via `@deprecated_parameter` alias (3 minor versions / 6 months removal
  timeline per `DEPRECATION_LIFECYCLE_POLICY.md`). Emits `DeprecationWarning`
  and translates to the new two-axis API internally with bit-identical
  results.

### Fixed

- **HJB Newton Jacobian now consults precomputed SOCP / M-matrix-QP weights.**
  The lazy fill of `_cached_derivative_weights` (around line 2006 of
  `hjb_gfdm.py`) previously read directly from `_gfdm_operator.get_derivative_weights`,
  bypassing precomputed-stencil overrides — so `_D_lap` / `_D_grad` (used by
  the batch Hamiltonian path) saw SOCP-corrected weights, but the per-point
  Newton Jacobian saw bare Wendland-Taylor. This caused a 12× `u_err` gap
  in the exp08 2D Towel-on-Beach validation between the research-side
  `patch_operator` workflow and the new first-class `joint_socp` scheme; the
  fix restores numerical equivalence (`u_err = 2.115` to 4 sig figs at iter 1
  in both paths).
- **`monotonicity_scheme="joint_socp"` now aliases internal
  `qp_optimization_level` to `"precompute"`** (previously `"none"`). The
  legacy value silently gates HJB Newton path selection: `"none"` selects
  the batch Hamiltonian path, anything else selects per-point. SOCP weights
  must be consumed by the per-point path to match the legacy patch_operator
  workflow.

## [0.19.4] - 2026-04-18

### Removed (BREAKING)

- **`mfgarchon.config.structured_schemas`** module deleted (Issue #1010 B4).
  It defined 13 OmegaConf dataclass schemas (`MFGSchema`, `BeachProblemSchema`,
  `NewtonSchema`, `HJBSchema`, etc.) that encoded a tree shape different from
  the canonical Pydantic `MFGSolverConfig` (nested `solver.hjb.method` vs
  flat `hjb.method`). These schemas were used only by test code and by a
  handful of loader methods on `OmegaConfManager`; no production code loaded
  them. Keeping them alongside the canonical Pydantic hierarchy was the
  dual-schema smell the v0.19.0–v0.19.3 renovation was eliminating elsewhere.
- **Dataclass-tied methods removed from `OmegaConfManager`** (Issue #1010 B3):
  `load_structured_config`, `load_mfg_config`, `load_beach_config_structured`,
  `create_default_mfg_config`, `validate_structured_config`. All returned
  `TypedMFGConfig` / `TypedBeachConfig` dataclass-shaped objects that are
  now gone. The `TypedMFGConfig` / `TypedBeachConfig` type aliases were
  removed along with them.
- **Module-level dataclass wrappers removed**: `load_structured_mfg_config`,
  `load_structured_beach_config`, `create_default_structured_config` (all in
  `mfgarchon.config.omegaconf_manager`).

### Kept (no change)

- **Generic OmegaConf functionality**: `OmegaConfManager.{load_config,
  compose_config, create_pydantic_config, save_config, create_parameter_sweep,
  validate_config, get_config_template}`. These operate on plain YAML /
  DictConfig and do not depend on dataclass schemas. Parameter sweeps and
  CLI overrides keep working via these methods.
- **`bridge_to_pydantic`**: the one-way gate between OmegaConf DictConfig and
  Pydantic `MFGSolverConfig` remains the canonical validation point. Users
  wanting validated configs should call `OmegaConf.load(...)` then pipe the
  result through `bridge_to_pydantic`.
- **YAML example files** in `configs/*.yaml`: kept as user-facing examples.
  These use the OmegaConf-style tree (`problem.T`, `solver.hjb.method`);
  users who want Pydantic validation should transform to the flat Pydantic
  shape first or use them as OmegaConf-only loads.

### Tests

- **Removed** `tests/unit/test_config/test_structured_configs.py` (247 lines)
  and `tests/unit/test_config/test_structured_schemas.py` (632 lines). These
  tested the dataclass tree that no longer exists. Pydantic-side coverage
  lives in `test_core.py`, `test_mfg_methods.py`, and `test_bridge.py` added
  in v0.19.3.

### Migration

User code that called the removed APIs (unlikely — internal audit found zero
production callers outside `mfgarchon/config/` itself) should migrate to the
OmegaConf + bridge pattern:

```python
# Old (removed):
from mfgarchon.config.omegaconf_manager import load_structured_mfg_config
config = load_structured_mfg_config("config.yaml")
# config was a DictConfig with MFGSchema tree shape

# New:
from omegaconf import OmegaConf
from mfgarchon.config import MFGSolverConfig
from mfgarchon.config.bridge import bridge_to_pydantic

raw = OmegaConf.load("config.yaml")
config = bridge_to_pydantic(raw, MFGSolverConfig)  # Pydantic validation at this point
```

Note that the YAML file's tree shape may need adjustment to match Pydantic's
flat `{hjb, fp, picard, backend, logging}` structure; the legacy YAMLs use
`{problem, solver, experiment}` nesting.

### Context

Closes the B3+B4 items of Issue #1010. With this release, the config system
has **one canonical schema authority** (Pydantic models in `core.py`,
`mfg_methods.py`, `array_validation.py`) and **one validation crossing**
(`bridge_to_pydantic`). OmegaConf handles YAML transport only — no schemas.
The North Star design from v0.19.0 is now fully realized.

## [0.19.3] - 2026-04-18

### Changed

- **Internal cleanup** (B1.5b follow-up): `create_network_mfg_solver` now
  forwards the canonical `relaxation` kwarg to `FixedPointIterator` internally
  instead of the legacy `damping_factor`. The legacy-forwarding was a
  deliberate temporary measure to keep the B1.5b.3 PR mergeable before B1.5b.1
  (FixedPointIterator rename) landed. With both now on main, the factory no
  longer emits a self-generated `DeprecationWarning` from its own codebase.

### Fixed

- **`ExperimentConfig` forward-ref crash under pydantic 2.12.5** (Issue #1010 B5):
  `NDArray` was imported under `TYPE_CHECKING` in `mfgarchon/config/array_validation.py`
  but used as an annotation on the `MFGArrays.U_solution` / `M_solution` fields.
  Pydantic 2.12.5+ resolves field annotations at model-build time and rejected the
  unresolved forward reference with `PydanticUserError: class-not-fully-defined`,
  breaking any instantiation of `ExperimentConfig`, `MFGArrays`, or
  `CollocationConfig`. Fixed by importing `NDArray` at runtime (not under
  `TYPE_CHECKING`), with a `noqa: TC002` on the import explaining why the
  runtime import is deliberate. The `MFGArrays.model_rebuild()` and
  `ExperimentConfig.model_rebuild()` workaround calls in `test_array_validation.py`
  are no longer needed and have been removed.

### Tests

- **Canonical config module coverage** (Issue #1010 B2):
  Added dedicated unit tests for `mfgarchon/config/core.py` and
  `mfgarchon/config/mfg_methods.py`, which previously had only indirect
  coverage via factory tests and integration tests. Two new files (58 tests):
  - `tests/unit/test_config/test_core.py` — 23 tests covering
    `LoggingConfig`, `BackendConfig`, canonical-path `PicardConfig`, and
    `MFGSolverConfig` (defaults, range validators, `@model_validator` hooks,
    `save_intermediate`-requires-`output_dir`, `numpy`-cannot-use-`gpu`,
    `anderson_memory <= max_iterations`, `model_dump` round-trip).
  - `tests/unit/test_config/test_mfg_methods.py` — 35 tests covering the 14
    method configs: default instantiation, Literal/enum rejection of
    invalid values, range-bound enforcement, `@model_validator` hooks
    (e.g., `wind_dependent_bc` requires `ghost_nodes`, FEM auto-quadrature).

## [0.19.2] - 2026-04-18

### Changed — B1.5b series (solver ctor kwargs damping_* → relaxation_*)

Incremental rename propagating the naming change landed in v0.19.1 (`PicardConfig`)
through the solver constructors. Four sub-PRs shipped together in this release,
each adding `@deprecated_parameter` decorators + silent `@property` aliases for
backward compatibility. Removal of legacy names scheduled for v0.25.0 per the
3-version deprecation window.

- **B1.5b.1** (PR #1012): `FixedPointIterator` — 7 ctor kwargs renamed
  (`damping_factor`, `damping_factor_M`, `adaptive_damping`, `adaptive_damping_decay`,
  `adaptive_damping_min`, `damping_schedule`, `damping_schedule_M`). Legacy kwargs
  accepted via `@deprecated_parameter` + body redirect. Silent `@property` aliases
  preserve `iter.damping_factor` attribute reads without warning-flooding Picard
  hot loops. 16 equivalence tests in
  `tests/unit/test_alg/test_fixed_point_iterator_relaxation_alias.py`.
- **B1.5b.2** (PR #1013): Block iterators — `BlockIterator` (base),
  `BlockJacobiIterator`, `BlockGaussSeidelIterator`. Renames `damping_factor` →
  `relaxation` and `damping_factor_M` → `relaxation_M` on all three. Legacy kwargs
  accepted via `@deprecated_parameter`. Silent `@property` aliases on base
  class. Plus `SolverResult.metadata` key `"damping_factor"` → `"relaxation"`
  (narrow break; no bridge available for dict-key reads; impact limited to code
  that inspects `result.metadata["damping_factor"]`). 18 equivalence tests in
  `tests/unit/test_alg/test_block_iterators_relaxation_alias.py`.
- **B1.5b.3** (PR #1014): `NetworkMFGSolver` factory functions + `MultiPopulationIterator`.
  `MultiPopulationIterator.__init__(damping_factor=)` → `relaxation=` with
  `@deprecated_parameter` + silent `@property` alias for attribute access.
  `create_network_mfg_solver(damping_factor=)` → `relaxation=`, and
  `create_simple_network_solver(damping=)` → `relaxation=` (same `@deprecated_parameter`
  pattern applied to factory functions, not just classes). Internal
  `self.damping_factor` attribute reads rewritten to `self.relaxation`;
  single-letter aliases (`omega = self.damping_factor`) removed in favor of direct
  `self.relaxation` use per internal style preference. 7 equivalence tests in
  `tests/unit/test_alg/test_network_multipop_relaxation_alias.py`.
- **B1.5b.4** (PR #1015): `HJBFDMSolver` + `FixedPointSolver` (utils/numerical).
  Same pattern: `damping_factor` → `relaxation` ctor kwarg, silent `@property`
  alias. HJBFDMSolver's internal construction of `FixedPointSolver` now forwards
  `relaxation=...` (canonical), and docstring "recommend 0.5-0.8" numerical
  guidance removed per the no-opinionated-numerical-recommendations style. 12
  equivalence tests in
  `tests/unit/test_alg/test_hjb_fdm_fp_solver_relaxation_alias.py`.

## [0.19.1] - 2026-04-17

### Changed

- **`PicardConfig` field rename** (naming abstraction): the five damping-related
  fields are renamed from `damping_*` to `relaxation_*`:

  | Legacy field (deprecated) | Canonical field |
  |---|---|
  | `damping_factor` | `relaxation` |
  | `damping_factor_M` | `relaxation_M` |
  | `damping_schedule` | `relaxation_schedule` |
  | `damping_schedule_M` | `relaxation_schedule_M` |
  | `adaptive_damping` | `adaptive_relaxation` |

  `relaxation` is the more abstract name — it extends cleanly to over-relaxation
  (omega > 1) if the range constraint is loosened in future work, whereas
  "damping" is conceptually under-relaxation only. `FixedPointIterator` reads
  of `config.picard.*` updated to canonical names.

### Deprecated

- Legacy `damping_*` kwargs on `PicardConfig(...)` still accepted via
  `@model_validator(mode="before")` translation, with `DeprecationWarning`
  emitted. Removal scheduled for **v0.25.0** per standard 3-version window.
  Passing both legacy and canonical names for the same concept raises
  `ValueError` immediately (e.g. `PicardConfig(damping_factor=0.5, relaxation=0.8)`).

### Tests

- New `tests/unit/test_config/test_picard_relaxation_alias.py` (15 tests)
  provides the mandatory equivalence tests per CLAUDE.md deprecation policy:
  each legacy kwarg produces an instance `==` to the canonical kwarg.

### Out of scope (future patches)

- Solver constructor kwargs (`FixedPointIterator(damping_factor=...)`,
  `HJBFDMSolver(damping_factor=...)`, etc.) are **not** renamed in this
  release. The runtime-layer rename via `@deprecated_parameter` is B1.5b,
  tracked in #1010.

## [0.19.0] - 2026-04-17

### Removed (BREAKING)

- **`mfgarchon.config.pydantic_config`** — the legacy parallel config hierarchy.
  All 7 exported classes (`NewtonConfig`, `PicardConfig`, `GFDMConfig`, `ParticleConfig`,
  `HJBConfig`, `FPConfig`, `MFGSolverConfig`) are now available exclusively from
  `mfgarchon.config`. See `docs/user/migration_v0.19.md` for import updates and
  field-by-field mapping (legacy defaults differed from canonical by up to 1000x in
  some fields, e.g. `PicardConfig.tolerance: 1e-3 -> 1e-6`).
- Phantom factory functions removed from user docs: `create_fast_config`,
  `create_accurate_config`, `create_research_config`, `create_enhanced_config`
  (never existed as public API; docs referenced them in error). Use
  `create_fast_solver` / `create_accurate_solver` / `create_research_solver` from
  `mfgarchon.factory` for preset patterns, or `MFGSolverConfig()` for direct config.
- **`hydra-core>=1.3`** dependency (declared but unused — zero `@hydra.main`,
  `from hydra`, or `HydraConfig` references in the codebase). Can be reintroduced
  deliberately if HPC sweep workflows or config-group based solver selection
  become priorities.

### Fixed

- **GraphMFGSolver source_term alignment** (Issue #1006): `_get_time_slice` in
  `graph_coupling.py` had a hardcoded `dt=0.05` default. Any problem with
  `dt != 0.05` silently indexed wrong time slices — invisible to tests that
  happened to use `dt=0.05`. `dt` is now threaded through
  `compute_hjb_source` / `compute_fp_source` and required at the indexing site.
- **GraphMFGSolver source composability**: per-node `problem.source_term_hjb`,
  `problem.source_term_fp`, and `problem.nonlocal_operator` were ignored when
  combined with graph coupling (only the graph source was injected into the
  HJB/FP solvers). New `_compose_hjb_source` / `_compose_fp_source` methods
  layer problem-level sources on top of the graph coupling source, matching
  the Layer 1 design's composability promise.
- `GraphMFGSolver.__init__` now validates that all nodes share the same `dt`
  (required for coupling to be well-defined); raises `ValueError` otherwise.

### Changed

- `pyproject.toml`: version bumped `0.18.19` -> `0.19.0`.
- `mfgarchon/config/omegaconf_manager.py`: `MFGSolverConfig` now imported from
  canonical `.core` module (was `.pydantic_config`).
- User docs (`plugin_development.md`, `migration.md`, `usage_patterns.md`):
  updated 6 import statements to canonical `from mfgarchon.config import ...`
  path.
- `GraphCouplingOperator.compute_hjb_source` / `compute_fp_source`: signature
  changed from `(..., t: float)` (vestigial, never used) to `(..., dt: float)`
  (load-bearing, used for time indexing). Callers of the protocol need to
  update their kwarg name.

### Audit context

Driven by a dual-config-system audit that revealed every legacy/canonical class
pair had diverged — both in schema (different fields) and in defaults (up to 1000x,
e.g. `PicardConfig.tolerance`). A simple deprecation-redirect was impossible since
the hierarchies were different APIs rather than versions of one API. v0.19.0 is a
hard break; subsequent v0.19.x patches will complete the internal consolidation
(canonical-module tests, YAML loader migration, removal of the remaining OmegaConf
dataclass mirrors, and `ExperimentConfig` NDArray forward-ref fix). Umbrella
tracking issue: #1010.

## [0.18.0] - 2026-03-29

### Added

- **Geometry trait compliance** (PR #872)
  - `ImplicitDomain`: `manifold_dimension`, `get_tangent_space_basis()`, `compute_christoffel_symbols()`, `validate_lipschitz_regularity()`
  - `GraphGeometry`: `mark_region()`, `get_region_mask()`, `get_region_names()`, `intersect_regions()`, `union_regions()`
  - Region predicate factories: `box_region()`, `sphere_region()`, `sdf_region()`, `halfspace_region()` in `geometry.predicates`
- **Periodic BC model compatibility** warning in user guide (quadratic potential on periodic domain pitfall)

### Changed

- `GhostCellConfig` relocated from `_compat.py` to `ghost_cells.py` (canonical location)
- Common noise MFG test updated for modern API (Issues #670, #673)

### Removed

- `HybridFPParticleHJBFDM` coupling solver (-757 lines) — deprecated since v0.9.0, 9 versions past policy

### Fixed

- Stale `See Also` references in boundary conditions user guide (docs migrated to mfg-research)

## [0.17.16] - 2026-03-28

### Added

- **BC resolution layer** — `MathBCType`, `BCResolver`, `HJBResolver`, `FPResolver` (PR #856, Issue #848)
- **Periodic BC for SL diffusion** — Sherman-Morrison circulant solver (PR #865, Issue #858)
- **Jupyter Book v2 (MyST)** configuration for docs (PR #864)
- Runtime `DeprecationWarning` for legacy `fdm_bc_1d.BoundaryConditions` (PR #869)

### Changed

- FP FDM boundary assembly decoupled — dict dispatch, Dirichlet nD, fail-fast (PR #868, Issue #859)
- Boundary module restructured — split monolith, slim exports (PR #849, Issue #848)
- BC design docs consolidated from 14 to 10 files (PR #850)
- ~111 raw deprecation warnings migrated to structured decorators (PR #847, Issue #841)
- All FP FDM tests migrated to modern BC API (PR #870)
- All remaining tests migrated from legacy fdm_bc_1d (PR #871, -706 lines)

### Removed

- Duplicate FEM BC system (PR #851, Issue #848)
- Theory docs moved to mfg-research (-7,567 lines) (PRs #853-855, Issue #852)
- Stale `docs/theory/` references (-429 lines) (PR #863)
- 28 RL placeholder test files (-9,051 lines) (PR #867, Issue #833)

## [0.17.13] - 2026-03-26

### Added

- **True adjoint mode** (Issue #707, PR #829)
  - `HJBFDMSolver.build_linearized_operator(U, M, time)` — builds linearized HJB Jacobian
  - `adjoint_mode="jacobian_transpose"` in `BlockIterator` for true adjoint FP coupling
  - `LinearizedOperatorCapable` protocol for type-safe solver integration
  - 8 unit tests + 4 integration tests validating convergence, mass conservation, analytical correctness

### Removed

- Dead deprecated shim modules: `grid_operators.py`, `tensor_operators.py`, `differential_utils.py` (-484 lines, PR #828)

### Changed

- `tensor_calculus.py` internalized — no longer re-exported from `utils.numerical`, no deprecation warning on import (PR #825)

## [0.17.12] - 2026-03-26

### Changed

- Complete `mfg_pde` → `mfgarchon` rename across archives and docs (follow-up to #821)
- README updated: fix API examples, remove stale version tag, fix tutorial links (PR #822)
- CITATION.cff: general description, no specific solver list (PR #822)
- Deprecation guide updated for v0.17.11 changes (PR #826)

### Fixed

- `SpatialCoordinates`/`TemporalCoordinates` deprecation warnings during test collection (PR #823)
- Stale `DOMAIN_2D`/`DOMAIN_3D` references in docs (PR #823)

### Removed

- Orphan `test_issue_557_fix.py` from project root
- 3 stale `.venv` editable install artifacts from old `mfg_pde` package

### Infrastructure

- Rename `_GmshMeshBase` → `_MeshGeneratorBase` for backend-agnostic naming (PR #824)
- Bump `actions/upload-artifact` 6→7, `actions/download-artifact` 7→8, `docker/setup-buildx-action` 3→4 (#815-817)

## [0.17.7] - 2026-02-06

### Fixed

- **Thread-safe global singletons** (Issue #759)
  - Added `threading.Lock()` with double-check locking to 4 global managers:
    - `plugin_system.get_plugin_manager()`
    - `workflow.get_workflow_manager()`
    - `network_backend.get_backend_manager()`
    - `general_mfg_factory.get_general_factory()`
  - Prevents race conditions in multi-threaded environments

- **Visualization type annotations** (Issue #758)
  - Removed 41 `type: ignore[assignment]` suppressions
  - Used proper `Any` typing for optional dependency fallbacks

- **Import patterns** (Issues #756, #757)
  - Replaced wildcard imports with explicit imports in `acceleration/`
  - Removed `sys.path` manipulation anti-pattern in solver modules

### Changed

- **Test suite cleanup** (Issue #761)
  - Reduced unconditional skips from 24 to 15 (37% reduction)
  - Fixed ghost buffer tests (incorrect assertions)
  - Deleted obsolete tests for deprecated patterns
  - Created tracking issues for remaining skips (#762, #763)

- **Deprecation timelines standardized** to v1.0.0 for all deprecated APIs

### Removed

- Deleted `tests/integration/test_coupled_hjb_fp_2d.py` (tested deprecated inheritance pattern)

## [0.17.6] - 2026-02-06

### Changed

- **Renamed `u_final` to `u_terminal`** (Issue #670, PR #755)
  - All APIs now use `u_terminal` for HJB terminal condition (MFG literature standard)
  - Deprecated `u_final` parameter in `MFGComponents`, redirects to `u_terminal`
  - Deprecated `get_u_final()`, `get_final_u()` methods, redirect to `get_u_terminal()`
  - Deprecated `validate_u_final()`, redirects to `validate_u_terminal()`
  - **Migration**: Replace `u_final=` with `u_terminal=` in all code

- **Unified `volatility_field` API** (Issue #717, PR #755)
  - Single `volatility_field` parameter handles all volatility specifications:
    - Scalar `σ` → isotropic diffusion `D = σ²/2`
    - Diagonal `[σ₀, σ₁, ...]` → anisotropic `D = diag(σᵢ²)/2`
    - Matrix `Σ (d×d)` → tensor diffusion `D = ΣΣᵀ/2`
    - Spatially varying `Σ(x)` → `D(x) = Σ(x)Σ(x)ᵀ/2`
    - Callable `σ(t,x,m)` or `Σ(t,x,m)` → state-dependent
  - Auto-detection by input shape (no separate parameters needed)

### Deprecated

- **`diffusion_field` parameter** → Use `volatility_field` instead
- **`tensor_diffusion_field` parameter** → Use `volatility_field` with `(d,d)` array
- **`volatility_matrix` parameter** → Use `volatility_field` with `(d,d)` array

### Documentation

- Updated `docs/NAMING_CONVENTIONS.md` with volatility vs diffusion terminology
- Added SDE-PDE relationship: `dX = μdt + σdW` → `∂ₜm = -∇·(μm) + DΔm` where `D = σ²/2`

## [0.17.5] - 2026-02-06

### Added

- **Adaptive Picard damping** (Issue #583, PR #745)
  - `adapt_damping()` function detects error oscillation and dynamically reduces damping
  - Opt-in via `FixedPointIterator(adaptive_damping=True)` (default off, backward compatible)
  - Independent U/M adaptation with cautious recovery toward initial damping
  - Damping history recorded in `SolverResult.metadata["adaptive_damping"]`
  - Gradient clipping warning now directs users to adaptive damping as primary fix

### Removed

- **`bc_mode` parameter from `HJBFDMSolver`** (Issue #703, #625)
  - Removed deprecated `bc_mode` parameter from `__init__` signature
  - Removed adjoint-consistent BC logic block from `solve_hjb_system`
  - **Migration**: Use `AdjointConsistentProvider` in `BCSegment.value` instead
  - Callers passing `bc_mode=` will now get `TypeError`

### Fixed

- **Mass Conservation in FP FDM Solver** (Issue #615)
  - Fixed catastrophic mass conservation failure (99.4% error → 2.3%)
  - Changed default advection scheme from `gradient_upwind` to `divergence_upwind`
  - Removed confusing `conservative: bool` parameter

## [0.17.4] - 2026-02-06

**Validation Initiative Release: Comprehensive Input Validation (Issue #685)**

### Added

- **Callable signature detection and adaptation** (Issue #684, PR #738)
  - New `adapt_ic_callable()` in `mfgarchon/utils/callable_adapter.py`
  - Auto-detects and wraps IC/BC callables: `f(x)` scalar, `f(x)` array, `f(x,t)`, `f(t,x)`, `f(x,y)`, `f(x,y,z)`
  - Zero-overhead passthrough for the common `f(x_scalar)` case
  - Detailed error messages listing all attempted calling conventions on failure
  - Expanded-coordinate signatures `f(x,y)` emit `DeprecationWarning`
- **Custom function validation** (Issue #686, PR #733)
  - `validate_hamiltonian()`, `validate_drift()`, `validate_running_cost()` in validation module
  - Probing-based signature detection for Hamiltonian, drift, running cost functions
  - Wired into `MFGProblem._initialize_functions()`
- **Array/field validation** (Issue #687, PR #735)
  - `validate_array_dtype()`, `validate_array_shape()`, `validate_field_dimension()`
  - Shape and dtype validation for solver arrays wired into MFGProblem
- **Runtime safety validation** (Issue #688, PR #736)
  - `check_finite()`, `check_bounds()`, `validate_solver_output()`
  - NaN/Inf detection wired into `FixedPointIterator`
- **IC/BC validation wiring** (Issue #681, PR #728)
  - `validate_components()` checks m_initial/u_final at problem construction
  - NDArray and callable IC/BC validated against geometry shape
- **Newton-to-Value-Iteration adaptive fallback** (Issue #669, PR #727)
  - HJB solver automatically falls back from Newton to value iteration on divergence

### Fixed

- **Backend device selection tests on Apple Silicon** (PR #737)
  - Fixed MPS backend detection tests that failed on Apple Silicon

### Changed

- **Validation module fully wired** into `MFGProblem._initialize_functions()` pipeline

## [0.17.2] - 2026-01-18

**Maintenance Release: Legacy Parameter Deprecation + Codebase Cleanup**

This release completes two important maintenance priorities:
1. **Legacy parameter deprecation** (Issue #544) - Deprecates old MFGProblem parameters, migrates all internal code to modern Geometry API
2. **Solver mixin cleanup** (Issue #545) - Removes dead code from completed refactoring

Both changes are 100% backward compatible with clear migration paths for users.

### Added

- **DeprecationWarning for Legacy Parameters** (Issue #544, Phase 1) 🎯
  - Warns users when using deprecated parameters: `Nx`, `xmin`, `xmax`, `Lx`, `spatial_bounds`, `spatial_discretization`
  - Clear migration instructions in warning message pointing to `docs/migration/LEGACY_PARAMETERS.md`
  - Respects `suppress_warnings=True` flag for gradual migration
  - **Timeline**: 6-12 month deprecation period before v1.0.0 removal

- **Comprehensive Migration Guide** (Issue #544):
  - `docs/migration/LEGACY_PARAMETERS.md` - 180-line guide with 5 common patterns
  - Before/after examples for each migration pattern
  - Nx → Nx_points conversion explained (Nx=100 intervals → Nx_points=[101] grid points)
  - Troubleshooting section for common issues

- **Documentation** (Issue #544, #545):
  - `docs/development/PRIORITY_8_PHASE_2_STATUS.md` - Complete deprecation plan (112 lines)
  - Updated `docs/development/PRIORITY_LIST_2026-01.md` - Priority 7 & 8 marked complete
  - Updated `docs/development/NEXT_STEPS_2026-01-18.md` - Next development priorities

### Changed

- **All Tests Migrated to Geometry API** (Issue #544, Phase 2) 🎯
  - Migrated 7 test files with 23 MFGProblem/StochasticMFGProblem calls
  - Integration tests: test_lq_common_noise_analytical.py, test_mass_conservation_1d*.py, test_particle_gpu_pipeline.py, etc.
  - Unit tests: test_common_noise_solver.py (12 calls)
  - Fixed SimpleMFGProblem1D mock for Geometry API compatibility
  - **Test results**: 79 + 23 + 12 passing, zero regressions

- **All Examples Verified** (Issue #544):
  - All files in `examples/` already use modern Geometry API
  - Zero migration needed (modern API adopted early)

### Deprecated

- **MFGProblem legacy parameters** (Issue #544) - **DEPRECATED, will be removed in v1.0.0**
  - `Nx`, `xmin`, `xmax`, `Lx` - Use `geometry=TensorProductGrid(...)` instead
  - `spatial_bounds`, `spatial_discretization` - Use `geometry=TensorProductGrid(...)` instead
  - DeprecationWarning provides migration guidance
  - See `docs/migration/LEGACY_PARAMETERS.md` for complete migration guide

### Removed

- **Dead Code Cleanup** (Issue #545) 🎯
  - Deleted `hjb_gfdm_monotonicity.py` (28KB) - MonotonicityMixin no longer used
  - Updated 5 outdated comments in hjb_gfdm.py referencing removed mixin
  - Verified: All 11 solvers use composition or simple inheritance (zero mixins)

### Fixed

- Documentation consistency in solver architecture references

## [0.17.1] - 2026-01-17

**Feature Release: Adjoint-Consistent Boundary Conditions + Three-Mode Solving API**

This release adds two major features:
1. **Adjoint-consistent boundary conditions** for HJB solver (Issue #574) - fixes equilibrium inconsistency at reflecting boundaries
2. **Three-mode solving API** (Issue #580) - prevents non-dual solver pairings

Both features include comprehensive documentation, validated testing, and are 100% backward compatible.

### Added

- **Three-Mode Solving API** (Issue #580, PR #585) 🎯
  - **Safe Mode**: `problem.solve(scheme=NumericalScheme.FDM_UPWIND)` - Guaranteed dual pairing
  - **Expert Mode**: `problem.solve(hjb_solver=hjb, fp_solver=fp)` - Manual control with validation
  - **Auto Mode**: `problem.solve()` - Intelligent defaults (backward compatible)
  - Prevents non-dual solver pairings that break Nash equilibrium convergence
  - Educational warnings guide users toward correct pairings
  - 121 tests validate correctness, 100% backward compatible

- **New Types** (Issue #580):
  - `NumericalScheme` enum: User-facing scheme selection (FDM_UPWIND, FDM_CENTERED, SL_LINEAR, SL_CUBIC, GFDM)
  - `SchemeFamily` enum: Internal classification (FDM, SL, FVM, GFDM, PINN, GENERIC)
  - `DualityStatus` enum: Validation status (DISCRETE_DUAL, CONTINUOUS_DUAL, NOT_DUAL, VALIDATION_SKIPPED)
  - `DualityValidationResult` dataclass: Rich validation result object

- **New Utilities** (Issue #580):
  - `check_solver_duality()`: Validates HJB-FP adjoint relationship
  - `create_paired_solvers()`: Factory for validated solver pairs with config threading
  - `get_recommended_scheme()`: Intelligent scheme selection (Phase 3 TODO - currently returns FDM_UPWIND)

- **New Examples** (Issue #580):
  - `examples/basic/three_mode_api_demo.py`: Comprehensive three-mode demonstration (246 lines)

- **New Documentation** (Issue #580):
  - `docs/development/issue_580_adjoint_pairing_implementation.md`: Technical guide (578 lines)
  - `docs/user/three_mode_api_migration_guide.md`: User migration guide (448 lines)

- **Adjoint-Consistent Boundary Conditions** (Issue #574, PR #588) 🎯
  - **`bc_mode` parameter** in `HJBFDMSolver`: `"standard"` | `"adjoint_consistent"`
  - Fixes equilibrium inconsistency at reflecting boundaries when stall points occur at domain boundaries
  - Mathematical formula: `∂U/∂n = -σ²/2 · ∂ln(m)/∂n` (Robin-type BC coupling HJB to FP density gradient)
  - **2.13x convergence improvement** validated in boundary stall configuration (703 → 330 max error)
  - Automatic BC computation from density gradient each Picard iteration
  - Negligible overhead (<0.1%), often reduces total iterations due to better consistency
  - 100% backward compatible (default `bc_mode="standard"` preserves classical Neumann BC)
  - 11 tests passing (smoke, integration, validation)

- **New Utilities** (Issue #574):
  - `compute_boundary_log_density_gradient()`: Computes ∂ln(m)/∂n at boundaries
  - `compute_coupled_hjb_bc_values()`: Converts to HJB BC values for adjoint-consistent mode

- **New Tutorial** (Issue #574):
  - `examples/tutorials/06_boundary_condition_coupling.py`: Comprehensive tutorial (266 lines)
  - Step-by-step comparison of standard vs adjoint-consistent BC modes
  - 4-panel visualization (density, value function, differences, convergence history)

- **New Documentation** (Issue #574):
  - `docs/development/issue_574_robin_bc_design.md`: Mathematical derivation and design (339 lines)
  - `docs/development/TOWEL_ON_BEACH_1D_PROTOCOL.md`: BC consistency solution section
  - `CLAUDE.md`: Boundary condition coupling patterns

### Changed

- **MFGProblem.solve()** (Issue #580):
  - Added `scheme` parameter for Safe Mode
  - Added `hjb_solver` and `fp_solver` parameters for Expert Mode
  - Mode detection and validation implemented
  - Fully backward compatible (existing code uses Auto Mode)

- **Solver Traits** (Issue #580):
  - All HJB and FP solvers now have `_scheme_family` class attribute
  - Used for refactoring-safe duality validation
  - Trait-based classification survives renames and inheritance changes

- **Renamed** `OneDimensionalAMRMesh` → `OneDimensionalAMRGrid` (Issue #466)
  - The class is a structured grid, not an unstructured mesh
  - Backward compatibility alias `OneDimensionalAMRMesh` remains (deprecated)
- **Renamed** `create_1d_amr_mesh()` → `create_1d_amr_grid()`
  - Backward compatibility alias remains (deprecated)

### Deprecated

- **`create_solver()`** (Issue #580) - Use three-mode API instead
  - Replacement: `problem.solve(scheme=...)` (Safe Mode) or `problem.solve(hjb_solver=..., fp_solver=...)` (Expert Mode)
  - Will be removed in v1.0.0
  - Deprecation warning guides migration with examples

- `OneDimensionalAMRMesh` - use `OneDimensionalAMRGrid` instead
- `create_1d_amr_mesh()` - use `create_1d_amr_grid()` instead

### Fixed

- **Critical BC Type Recognition Bug** (Issue #574, PR #588):
  - Fixed BC type recognition for `'no_flux'` string in HJB solver (`base_hjb.py`)
  - Previously, `neumann_bc()` objects were misinterpreted as periodic boundaries
  - **Impact**: Affects ALL Neumann BC usage throughout codebase (not limited to Issue #574)
  - Solver now correctly recognizes `'no_flux'`, `'neumann'`, `'dirichlet'`, `'periodic'`, and `'robin'` BC types

- **Scientific Correctness** (Issue #580):
  - Prevents accidental mixing of incompatible discretizations (e.g., FDM + GFDM)
  - Ensures L_FP = L_HJB^T relationship for Nash gap convergence
  - Type A (discrete dual) vs Type B (continuous dual) distinction enforced

- **HJB Boundary Equilibrium Consistency** (Issue #574):
  - Adjoint-consistent BC mode fixes 2.65x error increase at boundary stall configurations
  - Enables correct convergence to Boltzmann-Gibbs equilibrium

## [0.16.2] - 2025-12-12

**Patch Release: Grid Interpolator Batched Points Fix**

### Fixed

- **Grid-to-grid interpolation** now works correctly (Issue #444)
  - `TensorProductGrid.get_interpolator()` supports batched points (2D array of shape `(N, dim)`)
  - Single point evaluation remains backward compatible (returns `float`)
  - Projection between grids of different resolutions now works in 1D, 2D, and 3D

### Changed

- Fixed test assertions for 1D grids that used incorrect array shapes

## [0.16.1] - 2025-12-12

**Patch Release: Nx/Nx_points Naming Consistency**

This release introduces consistent naming for spatial and temporal discretization:
- `Nx` = number of intervals (consistent with `Nt`)
- `Nx_points` = number of grid points (`Nx + 1`)
- `Nt_points` = number of time points (`Nt + 1`)

### Added

- **`Nx` property** to `TensorProductGrid` - returns intervals per dimension
- **`Nx_points` property** to `TensorProductGrid` - returns grid points per dimension
- **`Nt_points` property** to `MFGProblem` - returns `Nt + 1`

### Changed

- `TensorProductGrid` constructor now accepts:
  - `Nx=` for intervals (like `Nt`)
  - `Nx_points=` for points
  - `num_points=` (deprecated, use `Nx_points`)
- Updated all codebase usages from `num_points=` to `Nx_points=`
- Updated `NAMING_CONVENTIONS.md` to document the new convention

### Deprecated

- **`num_points` parameter and property** in `TensorProductGrid`
  - Use `Nx_points` instead
  - Will be removed in v1.0.0

## [0.16.0] - 2025-12-11

**Feature Release: Geometry-First API Unification**

This release completes the geometry-first API unification for `MFGProblem`. The `geometry` attribute is now always non-None and serves as the single source of truth for all spatial information. Legacy attributes emit deprecation warnings but remain functional for backward compatibility.

### Changed

**Geometry-First API (Issue #435, PRs #436-#443)**

- **`MFGProblem.geometry` is now always non-None** after initialization
  - All four init paths (`_init_1d_legacy`, `_init_nd`, `_init_geometry`, `_init_network`) set geometry
  - Legacy parameters (`xmin`, `xmax`, `Nx`) automatically create `TensorProductGrid`
  - Network problems create appropriate `NetworkGeometry` subclass

- **Legacy attributes converted to computed properties**
  - `xmin`, `xmax`, `Lx`, `Nx`, `dx`, `xSpace`, `_grid` now derive from `self.geometry`
  - Properties emit `DeprecationWarning` when accessed
  - Setters allow backward-compatible assignment (stores to `_*_override`)
  - Internal code uses helper methods to avoid triggering warnings

- **Helper properties for geometry type dispatch**
  - `problem.is_cartesian` - True for `TensorProductGrid`
  - `problem.is_network` - True for `NetworkGeometry`
  - `problem.is_implicit` - True for implicit/SDF geometries

**OmegaConf Configuration (Issue #429, PRs #431-#432)**

- **Renamed OmegaConf classes to `*Schema` suffix** for clear naming convention
  - `MFGConfig` → `MFGSchema`
  - `SolverConfig` → `SolverSchema`
  - `HJBConfig` → `HJBSchema`
  - `FPConfig` → `FPSchema`
  - etc.

- **Added Pydantic-OmegaConf bridge utilities**
  - `bridge_to_pydantic()` - Generic adapter for OmegaConf → Pydantic conversion
  - `save_effective_config()` - Save resolved config for reproducibility
  - `load_effective_config()` - Load previously saved config

### Deprecated

- **Legacy attribute access** (`problem.xmin`, `problem.xmax`, `problem.Nx`, `problem.dx`, `problem.xSpace`)
  - Use `problem.geometry.get_bounds()`, `problem.geometry.num_spatial_points`, `problem.geometry.get_spatial_grid()` instead
  - Will be removed in v1.0.0

### Documentation

- Updated `GEOMETRY_FIRST_API_GUIDE.md` with migration table and v0.16.0 patterns
- Updated `DEPRECATION_MODERNIZATION_GUIDE.md` with Phase 7 completion status
- Updated `quickstart.md` to use geometry-first API in all examples
- Updated `migration.md` with v0.16.0 current API section

## [0.14.1] - 2025-12-06

### Changed

- **Rename `PerfectMazeGenerator` → `MazeGeometry`**: Better reflects role as geometry class for MFG problems

### Fixed

- **MazeGeometry now satisfies GeometryProtocol**: Can be used directly with `MFGProblem`
  - `generate()` returns `self` instead of `Grid`
  - `dimension=2` (spatially embedded) instead of `0` (abstract graph)
  - `geometry_type=MAZE` instead of `NETWORK`
- **MAZE/NETWORK geometry handlers in MFGProblem**: Properly extracts `spatial_bounds` from graph geometries

### Deprecated

- **`fdm_bc_1d` module**: Migrated examples to unified BC API (`periodic_bc()` from `mfgarchon.geometry.boundary`)

## [0.14.0] - 2025-12-01

### Removed - API Simplification (2025-11-23)

**BREAKING CHANGES**: Removed unnecessary API layers to enforce clean 2-level architecture (Factory vs Expert).

- **Removed `ExampleMFGProblem`** (deprecated since v0.12.0)
  - Migration: Use `MFGProblem` directly
  - Old: `problem = ExampleMFGProblem(dimension=2, X=X, t=t, g=g, H=H)`
  - New: `components = MFGComponents(hamiltonian_func=H, final_value_func=g); problem = MFGProblem(spatial_bounds=..., components=components)`

- **Removed `MFGProblemBuilder`**
  - Redundant builder pattern that added cognitive load without benefit
  - Migration: Use `MFGProblem` with `MFGComponents` directly
  - Old: `problem = MFGProblemBuilder().hamiltonian(H, dH).domain(0,10,100).build()`
  - New: `components = MFGComponents(hamiltonian_func=H, hamiltonian_dm_func=dH); problem = MFGProblem(xmin=0, xmax=10, Nx=100, components=components)`

- **Removed `create_mfg_problem()` convenience function**
  - Redundant wrapper around `MFGProblem` constructor
  - Migration: Use `MFGProblem` with `MFGComponents` directly
  - Old: `problem = create_mfg_problem(H, dH, xmin=0, xmax=10, Nx=100)`
  - New: `components = MFGComponents(hamiltonian_func=H, hamiltonian_dm_func=dH); problem = MFGProblem(xmin=0, xmax=10, Nx=100, components=components)`

**Rationale**: Enforces clear 2-level architecture:
- **Level 1 (Factory)**: Pre-configured problems via `create_*_problem()` functions
- **Level 2 (Expert)**: Direct `MFGProblem` + `MFGComponents` for full control

See `docs/development/API_SIMPLIFICATION_PROPOSAL.md` for details.

### Fixed - 2D/nD Support (2025-11-23)

- **Fixed Gap 1**: `H()` and `dH_dm()` now handle 2D/nD tuple indices `(i,j)` correctly
  - No longer crashes with `TypeError: 'NoneType' object is not subscriptable`
  - Proper multi-dimensional indexing via `np.ravel_multi_index()`

- **Fixed Gap 2**: `_setup_custom_final_value()` now works for nD problems
  - Uses `geometry.get_spatial_grid()` for nD instead of assuming 1D `xSpace`
  - Custom terminal conditions work in 2D and higher dimensions

## [0.12.1] - 2025-11-11

**Patch Release: API Consistency Improvements (Week 1)**

This patch release implements Week 1 quick wins from Issue #277 (API Consistency Audit), converting boolean pairs to enums and tuple returns to dataclasses for improved API clarity and type safety.

### Changed

**API Modernization (Issue #277 Phase 2 Week 1)**

- **HamiltonianJacobians dataclass** replaces tuple return in `MFGProblem.get_hjb_hamiltonian_jacobian_contrib()`
  - Self-documenting API: `jacobians.diagonal` instead of `result[0]`
  - Type-safe structured return with named fields
  - Updated HJB solver to use dataclass attributes

- **ProfilingMode enum** replaces `enable_profiling`/`verbose` boolean pair in `StrategySelector`
  - Three clear states: `DISABLED`, `SILENT`, `VERBOSE`
  - String support: `profiling_mode="verbose"`
  - Full backward compatibility with deprecation warnings

- **MeshVisualizationMode enum** replaces `show_edges`/`show_quality` boolean pair in `visualize_mesh()`
  - Four visualization modes: `SURFACE`, `WITH_EDGES`, `QUALITY`, `QUALITY_WITH_EDGES`
  - String shortcuts for quick usage
  - Applies to both `base_geometry.py` and `base.py`

### Fixed

- Docstring examples now use correct lowercase `.dx`/`.dt` convention (2 violations fixed in `problem_protocols.py`)

### Deprecated

- `StrategySelector(enable_profiling=..., verbose=...)` → Use `profiling_mode=ProfilingMode.SILENT` instead
- `visualize_mesh(show_edges=..., show_quality=...)` → Use `mode=MeshVisualizationMode.WITH_EDGES` instead
- Old APIs remain functional with deprecation warnings until v2.0.0

## [0.12.0] - 2025-11-11

**Feature Release: Advanced Projection Methods & API Modernization**

This release adds advanced particle-to-grid projection methods (GPU KDE, multigrid operators), completes the Dx/Dt→dx/dt migration, implements adaptive hybrid CPU/GPU strategies, and introduces enum-based configuration with full backward compatibility.

### Added

**Advanced Projection Operators (PRs #269, #270, Issue #265)**

- **Multi-dimensional GPU KDE** for particle-to-grid projection
  - GPU-accelerated kernel density estimation for 1D/2D/3D
  - Scott's rule and Silverman's rule for automatic bandwidth selection
  - Memory-efficient implementation for large particle systems
  - Fallback to CPU implementation when GPU unavailable
  - Significantly improves accuracy over histogram-based projection

- **Conservative restriction and prolongation operators** for multigrid methods
  - Conservative restriction: Fine → coarse grid with exact mass conservation
  - High-order prolongation: Bilinear/bicubic interpolation for coarse → fine
  - Supports 1D/2D/3D grids with arbitrary refinement ratios
  - Essential for multigrid acceleration of MFG solvers

**Adaptive Hybrid Strategies (PR #268, Issue #262)**

- **Intelligent CPU/GPU backend selection** for particle methods
  - Automatic threshold-based selection (10,000 particles)
  - Performance-optimized decision making based on problem size
  - Graceful handling of backend=None (automatic selection)
  - Reduces GPU overhead for small problems, leverages GPU for large ones

**GFDM Gradient Operators (PR #267, Issue #261)**

- **Full drift computation** in particle FP solver: `α = -∇U`
  - Implements GFDM-based gradient operator for arbitrary grids
  - Replaces zero-drift placeholder with proper physics
  - All 36 particle FP tests pass with realistic dynamics

**Modern Configuration with Enums (PR #283, Issue #277 Phase 2)**

- **AdaptiveTrainingMode** enum for PINN adaptive training strategies
  - Values: `BASIC`, `CURRICULUM`, `MULTISCALE`, `FULL_ADAPTIVE`
  - Replaces boolean triplet: `enable_curriculum`, `enable_multiscale`, `enable_refinement`
  - Backward compatible via `__post_init__` deprecation handling

- **NormalizationType** enum for PINN normalization methods
  - Values: `NONE`, `INPUT`, `LOSS`, `BOTH`
  - Replaces boolean pair: `normalize_input`, `normalize_loss`

- **VarianceReductionMethod** enum for DGM variance reduction
  - Values: `NONE`, `BASELINE`, `CONTROL_VARIATE`, `BOTH`
  - Replaces boolean pair: `use_baseline`, `use_control_variates`

**Enhanced Dependency Management (PR #279, Issue #278)**

- Improved error messages when optional dependencies missing
- Better diagnostics for installation issues
- User-friendly guidance for installing GPU backends

**Examples Reorganization (PR #275)**

- Elevated `tutorials/` to peer level with `basic/` and `advanced/`
- Hierarchical organization: `applications/`, `notebooks/`, `plugins/`
- Enhanced tutorial content (tutorials 04 and 05)
- Cleaner examples directory structure

### Deprecated

- **`Dt` attribute**: Use lowercase `dt` instead (Issue #245, PR #259, #274). Backward compatibility maintained via deprecated property that emits `DeprecationWarning`. Will be removed in v1.0.0.
- **`Dx` attribute**: Use lowercase `dx` instead (Issue #245, PR #259, #274). Backward compatibility maintained via deprecated property that emits `DeprecationWarning`. Will be removed in v1.0.0.
- **Boolean configuration parameters**: Replaced with enums (Issue #277, PR #283). Old parameters still work with deprecation warnings. Will be removed in v1.0.0.
- **GridBasedMFGProblem**: Removed. Use `MFGProblem` with `spatial_bounds` and `spatial_discretization` for nD problems.

### Changed

- **Primary time step attribute**: Changed from `Dt` to `dt` throughout codebase (46 files, ~102 references) following official naming conventions (`docs/NAMING_CONVENTIONS.md` lines 24, 262)
  - Core: `mfgarchon/core/mfg_problem.py`, `mfgarchon/types/problem_protocols.py`
  - Solvers: All HJB, FP, and coupling solvers updated
  - Utilities: `experiment_manager.py`, `hjb_policy_iteration.py`
  - Tests: 15 test files (59 references)
  - Examples: 5 example files (8 references)
  - Benchmarks: 3 benchmark files (4 references)

- **Primary spatial spacing attribute**: Changed from `Dx` to `dx` for 1D problems (same scope as above)

### Fixed

- Test collection errors (PR #266): Removed 2,280 lines of obsolete test code
- Flaky TD3 test (Issue #237): Made `test_soft_update_all_target_networks` deterministic
- Parameter migration system: Added missing `max_iterations → max_picard_iterations` mapping

### Documentation

- API violations audit (PR #282, Issue #277 Phase 1)
- Array-based notation standard (Issue #243 Phase 1)
- Comprehensive dual geometry example

### Issues Closed

9 issues closed: #278, #277 (Phase 2), #273, #265, #262, #261, #260, #259, #243 (Phase 1), #237

### Migration Guide

**For users**: Update your code to use lowercase attributes:
```python
# OLD (deprecated but works with warnings in v0.12.0)
dt = problem.Dt
dx = problem.Dx

# NEW (recommended)
dt = problem.dt
dx = problem.dx
```

**For enum configurations**:
```python
# OLD (deprecated but works with warnings)
config = AdaptiveTrainingConfig(
    enable_curriculum=True,
    enable_multiscale=True,
    enable_refinement=True
)

# NEW (recommended)
from mfgarchon.alg.neural.pinn_solvers.adaptive_training import AdaptiveTrainingMode
config = AdaptiveTrainingConfig(
    training_mode=AdaptiveTrainingMode.FULL_ADAPTIVE
)
```

**For developers**: The deprecated properties and parameters will be completely removed in v1.0.0.

## [0.11.0] - 2025-11-10

**Major Release: Dual Geometry Architecture**

This release introduces complete dual geometry support, enabling HJB and FP solvers to use different discretizations. This enables multi-resolution methods (4-15× speedup), FEM meshes with obstacles, hybrid particle-grid methods, and network-based agent models.

### Added

**Dual Geometry Infrastructure (PR #258, Issues #257 & #245 Phase 4)**

- **GeometryProjector** class (`mfgarchon/geometry/projection.py`, 706 lines)
  - Automatic projection method selection based on geometry types
  - `project_hjb_to_fp()`: Maps HJB solution values to FP geometry
  - `project_fp_to_hjb()`: Maps FP density values to HJB geometry
  - Supports grid-to-grid, grid-to-particles, particles-to-grid (KDE)
  - Vectorized implementations for 1D/2D/3D

- **ProjectionRegistry** pattern
  - Decorator-based registration: `@ProjectionRegistry.register(SourceType, TargetType, direction)`
  - Hierarchical fallback: exact type → category match → generic
  - O(N) custom projectors (not O(N²))
  - User-extensible for custom geometry types

- **MFGProblem Dual Geometry Integration** (`mfgarchon/core/mfg_problem.py`)
  - New parameters: `hjb_geometry` and `fp_geometry`
  - Automatic `GeometryProjector` creation when geometries differ
  - Unified attribute access: `problem.hjb_geometry`, `problem.fp_geometry`
  - Full backward compatibility with single `geometry` parameter

- **FEM Mesh Support**
  - Automatic Delaunay interpolation for `UnstructuredMesh` ↔ `CartesianGrid` (requires scipy)
  - Nearest neighbor fallback when scipy unavailable
  - Works with Mesh2D, Mesh3D, TriangularAMRMesh
  - Graceful extrapolation handling (fills NaN with nearest neighbor)

- **Vectorized Grid Interpolators**
  - `SimpleGrid1D.get_interpolator()`: Binary search-based 1D interpolation
  - `SimpleGrid2D/3D.get_interpolator()`: RegularGridInterpolator wrapper
  - Accepts array of query points for batch interpolation
  - Used by projection system for efficient grid-to-mesh operations

### Documentation

**Comprehensive Dual Geometry Documentation** (5,000+ lines)

- **Theory**: `docs/theory/geometry_projection_mathematical_formulation.md` (556 lines)
  - Mathematical formulation of all projection methods
  - Error analysis (interpolation, KDE, nearest neighbor)
  - Performance complexity analysis (O(N log N), O(N), etc.)
  - Pseudocode for all algorithms

- **Developer Guide**: `docs/development/GEOMETRY_PROJECTION_IMPLEMENTATION_GUIDE.md` (797 lines)
  - Adding new geometry types and projections
  - Registry pattern usage and best practices
  - Debugging tips and performance optimization
  - Complete code examples for custom projections

- **User Guide**: `docs/user_guide/dual_geometry_usage.md` (679 lines)
  - Complete workflow examples
  - Use cases: multi-resolution, hybrid methods, network agents
  - Performance tips and FAQ
  - Best practices for choosing projection methods

- **FEM Mesh Guide**: `docs/user_guide/fem_mesh_projection_guide.md` (352 lines)
  - FEM mesh support levels (basic + optimized)
  - Comparison of nearest neighbor vs Delaunay
  - Use cases: complex domains, obstacles, CAD import
  - Complete examples with performance tips

- **Migration Guide Update**: `docs/migration/unified_problem_migration.md`
  - Updated with dual geometry integration
  - Examples showing unified API + dual geometry together
  - Updated deprecation timeline with v0.11.0 milestone

- **Completion Summary**: `docs/development/ISSUE_257_COMPLETION_SUMMARY.md` (379 lines)
  - Complete implementation details for all 5 phases
  - Performance impact and testing results
  - Known limitations and future enhancements

### Examples

- **Multi-Resolution MFG**: `examples/basic/dual_geometry_multiresolution.py` (323 lines)
  - Fine HJB grid (100×100) + coarse FP grid (25×25)
  - Demonstrates 4× speedup with minimal accuracy loss
  - Complete visualization of projections
  - Performance comparison with unified geometry

- **FEM Mesh with Obstacles**: `examples/advanced/dual_geometry_fem_mesh.py` (330 lines)
  - Complex domain with circular obstacle using Gmsh
  - Automatic vs manual Delaunay registration
  - Accuracy comparison of projection methods
  - Working example with 495 vertices, 884 elements

### Testing

- **Projection Tests**: `tests/unit/geometry/test_geometry_projection.py` (439 lines)
  - 20 unit tests covering all projection methods
  - Shape verification, accuracy tests, conservation tests
  - Tests for 1D, 2D, 3D projections
  - Registry pattern tests

- **Integration Tests**: `tests/unit/test_core/test_mfg_problem.py` (+131 lines)
  - 7 new tests for dual geometry MFGProblem integration
  - Backward compatibility verification
  - Error handling and validation tests

### Use Cases Enabled

| Use Case | HJB Geometry | FP Geometry | Benefit |
|----------|--------------|-------------|---------|
| Multi-resolution | Fine grid | Coarse grid | 4-15× speedup, 46% memory savings |
| Complex domains | Regular grid | FEM mesh | Fast HJB, handles obstacles naturally |
| Hybrid methods | Grid | Particles | Grid-based value, particle density |
| Network agents | Grid | Network graph | Spatial value, network-constrained agents |

### Performance

- Multi-resolution: 4-15× speedup (depending on resolution ratio)
- Projection overhead: <1% of solve time
- Memory savings: Up to 46% for 4× resolution ratio
- Grid→Points: O(N) with RegularGridInterpolator
- Particles→Grid KDE: GPU-accelerated available (1D)

### Changed

- README updated with v0.11.0 features and dual geometry examples
- Citation updated to v0.11.0

### Backward Compatibility

- ✅ Fully backward compatible
- Existing code using single `geometry` parameter continues to work
- `hjb_geometry` and `fp_geometry` are optional
- No breaking changes

### Closes

- Issue #257: Dual geometry architecture (5 phases complete)
- Issue #245 Phase 4: Documentation for unified MFG problem

---

## [0.10.0] - 2025-11-05

**Major Release: Geometry-First API**

This release introduces the geometry-first API, a new recommended pattern for constructing MFG problems using geometry objects. This provides better type safety, clearer separation of concerns, and unified support for diverse geometry types.

### Added

**PR #244: Phase 2 Array Notation - Backward Compatible Implementation**
- Added `_normalize_to_array()` helper method in `MFGProblem` (`mfg_problem.py:79-122`)
  - Automatically converts scalar inputs to arrays
  - Emits `DeprecationWarning` for scalar usage
  - Points users to `MATHEMATICAL_NOTATION_STANDARD.md`
- Updated `MFGProblem.__init__` signature to accept both scalar and array inputs:
  - `Nx`: `int | list[int]` (deprecated scalar, standard array)
  - `xmin`, `xmax`: `float | list[float]` (deprecated scalar, standard array)
- Both scalar and array inputs produce identical results with 100% backward compatibility
- Migration path for Phase 3 (v1.0.0): Remove deprecated scalar API

**PR #247: GeometryProtocol Foundation**
- Created `GeometryProtocol` runtime-checkable Protocol (`mfgarchon/geometry/geometry_protocol.py`)
  - Minimal interface for all geometry objects
  - Four required properties: `dimension`, `geometry_type`, `num_spatial_points`, `get_spatial_grid()`
- Created `GeometryType` enum with 7 types:
  - `CARTESIAN_GRID`: Regular tensor product grids
  - `NETWORK`: Graph/network geometries
  - `MAZE`: Maze environments
  - `DOMAIN_2D`, `DOMAIN_3D`, `DOMAIN_1D`: Cartesian/unstructured meshes
  - `IMPLICIT`: Level sets and signed distance functions
  - `CUSTOM`: User-defined geometries
- Added helper functions:
  - `detect_geometry_type()`: Self-aware type detection via attribute inspection
  - `is_geometry_compatible()`: Compatibility checking
  - `validate_geometry()`: Validation with informative error messages
- Implemented GeometryProtocol for 6 core geometry classes:
  - `Domain1D`: 1D Cartesian grids with grid caching
  - `BaseGeometry`: Abstract base for Domain2D/Domain3D meshes
  - `TensorProductGrid`: Arbitrary-dimension structured grids
  - `NetworkGeometry`: Graph-based geometries (Grid/Random/ScaleFree networks)
  - `ImplicitDomain`: Meshfree domains via signed distance functions (`Hyperrectangle`, `Hypersphere`)
  - `Grid` (mazes): Maze-based geometries from PerfectMazeGenerator
- Comprehensive design documentation (`docs/development/UNIFIED_GEOMETRY_PARAMETER_DESIGN.md`, 844 lines)

**Geometry-First API Implementation**
- Updated `MFGProblem._init_geometry()` to accept any GeometryProtocol-compliant object (`mfg_problem.py:647-768`)
  - Automatic geometry type detection via `geometry.geometry_type` enum
  - Specialized handling for CARTESIAN_GRID, IMPLICIT, DOMAIN_2D/3D, MAZE, NETWORK types
  - Generic fallback for CUSTOM geometries
- Added deprecation warnings for manual grid construction (`mfg_problem.py:350-363, 430-450`)
  - Warns users to migrate to geometry-first API
  - Points to migration guide with code examples
  - 100% backward compatibility maintained
- Created `docs/migration/GEOMETRY_FIRST_API_GUIDE.md` (400+ lines)
  - Quick start examples for all geometry types
  - Migration strategy from old to new API
  - Performance considerations and FAQ
- Created `examples/basic/geometry_first_api_demo.py` (350+ lines)
  - Demonstrates 8 geometry patterns (TensorProductGrid, Domain1D, Hyperrectangle, Hypersphere, Maze, 4D, reuse, refinement)
  - All examples tested and working
- Fixed normalization bug for implicit geometries (`mfg_problem.py:1158-1172`)
  - Handles `None` spatial_bounds for SDF-based geometries
  - Uses uniform approximation when structured grid info unavailable

### Changed

**API Improvements**
- `MFGProblem` now accepts both scalar and array notation for spatial parameters
- `MFGProblem` now accepts geometry objects via `geometry=` parameter (NEW recommended API)
- Array notation is the standard for manual construction (following `MATHEMATICAL_NOTATION_STANDARD.md`)
- Scalar inputs and manual grid construction trigger deprecation warnings

**Code Quality**
- Unified geometry interface across all geometry types via GeometryProtocol
- Protocol-based design enables duck typing without explicit inheritance
- Self-aware geometry types for automatic type detection in MFGProblem
- Enhanced type safety and consistency across geometry module
- Separation of concerns: geometry construction vs. problem temporal/diffusion parameters

### Deprecated

**API Patterns** (will be restricted in v1.0.0, removed in v2.0.0)
- Manual grid construction in `MFGProblem` (passing `spatial_bounds`, `spatial_discretization`, `xmin`, `xmax`, `Nx`)
  - Use geometry-first API instead: create geometry object, pass to `MFGProblem(geometry=...)`
  - Deprecation warnings provide migration examples
  - See `docs/migration/GEOMETRY_FIRST_API_GUIDE.md` for complete guide
- Scalar `Nx`, `xmin`, `xmax` parameters (if still using manual construction)
  - Use arrays instead: `Nx=[100]`, `xmin=[-2.0]`, `xmax=[2.0]`
  - Warnings guide users to `MATHEMATICAL_NOTATION_STANDARD.md`

**Deprecation Timeline**:
- v0.10.x: Warnings emitted, old API fully functional
- v0.11.x - v0.99.x: Continued warnings
- v1.0.0: Manual construction requires explicit `allow_manual_construction=True` flag
- v2.0.0: Complete removal of manual construction

### Documentation

- Array-Based Notation Migration plan (`docs/development/ARRAY_BASED_NOTATION_MIGRATION.md`)
- Mathematical Notation Standard (`docs/development/MATHEMATICAL_NOTATION_STANDARD.md`)
- Unified Geometry Parameter Design (`docs/development/UNIFIED_GEOMETRY_PARAMETER_DESIGN.md`)
- Geometry-First API Guide (`docs/migration/GEOMETRY_FIRST_API_GUIDE.md`)

### Future Work (Planned for 0.10.x series)

**v0.10.1** (Planned):
- Add GeometryProtocol compliance to AMR classes (OneDimensionalAMRMesh, AdaptiveMesh, TriangularAMRMesh, TetrahedralAMRMesh)
- Enable AMR meshes to be used directly in `MFGProblem(geometry=amr_mesh)`

**v0.10.2** (Planned):
- Design and implement dimension-agnostic boundary condition system (`BoundaryConditionND`)
- Support for nD boundary conditions (d > 3) with per-axis BC specification

**v0.10.3** (Planned):
- Rename `BaseGeometry` → `MeshGeometry` for clarity (breaking change with deprecation)
- Update all documentation and examples to reflect renamed class

### Testing

- All 3300+ tests passing
- Array notation backward compatibility validated
- GeometryProtocol compliance verified for all implemented geometries

## [0.9.1] - 2025-11-04

### Added

**PR #242: GFDM Operators with Unified Smoothing Kernels**
- `mfgarchon/utils/numerical/smoothing_kernels.py` (807 lines)
  - Unified kernel implementations: Gaussian, Wendland, Cubic Spline, Quintic Spline, Cubic, Quartic
  - Parameterized Wendland kernels: `WendlandKernel(k=0,1,2,3)` for C^0, C^2, C^4, C^6 smoothness
  - Arbitrary dimension support with proper normalization
  - Factory pattern: `create_kernel(kernel_type, dimension)`
  - Derivative support for gradient-based methods
- `mfgarchon/utils/numerical/gfdm_operators.py` (1050 lines)
  - Weighted least squares gradient/Hessian reconstruction
  - Support for structured and unstructured grids
  - Boundary condition handling (Dirichlet, Neumann)
  - Anisotropic/directional derivative support
- Theory documentation with differential operators (gradient, divergence, Laplacian)
- Comprehensive test suite (502 lines, 54 tests)
- Advanced example demos for nD geometry and implicit geometry

**PR #239: Maze Refactoring**
- Moved maze generation from `alg/reinforcement/environments` to `geometry/mazes`
- Makes maze utilities accessible to all solver types (PDE, particle, neural, RL)
- Backward compatibility through re-exports
- 6 core files relocated: `maze_generator`, `hybrid_maze`, `voronoi_maze`, `maze_config`, `maze_utils`, `maze_postprocessing`

### Changed
- Updated solver integrations to use unified kernel API
- Consolidated 4 separate Wendland classes into single parameterized implementation
- Updated test imports to reference new maze location

### Documentation
- Added `docs/theory/smoothing_kernels_mathematical_formulation.md` with complete mathematical foundations
- Dimension-specific formulas for differential operators (1D, 2D, 3D)
- SPH and GFDM application notes
- Implementation details with code references

### Testing
- All 3300+ tests passing
- New GFDM operator tests validated against analytical solutions
- Kernel tests cover edge cases, normalization, and derivatives

## [0.9.0] - 2025-11-03

### Phase 3 Complete: Unified Architecture

Major architecture refactoring completing Phase 3.1 (MFGProblem), Phase 3.2 (SolverConfig), and Phase 3.3 (Factory Integration).

### Added

**Issue #216: Missing Utilities (Complete - All 4 Parts)**
- **Part 1: Particle Interpolation** (commit 84e6e6d)
  - `interpolate_grid_to_particles()` - Grid → Particles (1D/2D/3D)
  - `interpolate_particles_to_grid()` - Particles → Grid (RBF, KDE, nearest)
  - `estimate_kde_bandwidth()` - Automatic bandwidth selection
  - Saves ~220 lines per research project
- **Part 2: Signed Distance Functions** (commit 83f59f4)
  - Primitives: `sdf_sphere()`, `sdf_box()` for 1D/2D/3D/nD
  - CSG operations: `sdf_union()`, `sdf_intersection()`, `sdf_complement()`, `sdf_difference()`
  - Smooth blending: `sdf_smooth_union()`, `sdf_smooth_intersection()`
  - Gradient: `sdf_gradient()` using finite differences
  - Saves ~150 lines per research project
- **Part 3: QP Solver Caching** (already existed)
  - `QPCache` - Hash-based caching with LRU eviction
  - `QPSolver` - Unified solver with warm-starting
  - Multiple backends: OSQP, scipy SLSQP, scipy L-BFGS-B
  - Saves ~180 lines per project + 2-5× GFDM speedup
- **Part 4: Convergence Monitoring** (already existed)
  - `AdvancedConvergenceMonitor` - Plotting, stagnation detection
  - `AdaptiveConvergenceWrapper` - Adaptive convergence criteria
  - Saves ~60 lines per project
- **Total Impact**: ~610 lines saved per research project + performance improvements

**Phase 3.1: Unified Problem Class (PR #218)**
- Single `MFGProblem` class replacing 5+ specialized problem classes
- Flexible `MFGComponents` system for custom problem definitions
- Auto-detection of problem types (standard, network, variational, stochastic, highdim)
- `MFGProblemBuilder` for programmatic problem construction
- Full backward compatibility with deprecated specialized classes

**Phase 3.2: Unified Configuration System (PR #222)**
- New `SolverConfig` class unifying 3 competing config systems
- Three usage patterns:
  - YAML files for experiments and reproducibility
  - Builder API for programmatic configuration
  - Presets for common use cases
- Modular config components: `PicardConfig`, `HJBConfig`, `FPConfig`, `BackendConfig`, `LoggingConfig`
- Preset configurations: fast, accurate, research, production, domain-specific
- YAML I/O with validation
- Legacy config compatibility layer

**Phase 3.3: Factory Integration (PR #224)**
- Unified problem factories supporting all MFG types:
  - `create_mfg_problem()` - Main factory for any problem type
  - `create_standard_problem()` - Standard HJB-FP MFG
  - `create_network_problem()` - Network/Graph MFG
  - `create_variational_problem()` - Variational/Lagrangian MFG
  - `create_stochastic_problem()` - Stochastic MFG with common noise
  - `create_highdim_problem()` - High-dimensional MFG (d > 3)
  - `create_lq_problem()` - Linear-Quadratic MFG
  - `create_crowd_problem()` - Crowd dynamics MFG
- Updated `solve_mfg()` interface:
  - New `config` parameter accepting `SolverConfig` instances or preset names
  - Deprecated `method` parameter (still works with warning)
  - Automatic config resolution from strings
- Extended `MFGComponents` for all problem types (network, variational, stochastic, highdim)
- Dual-output factory support: unified MFGProblem (default) or legacy classes (deprecated)
- New examples: `factory_demo.py`, updated `solve_mfg_demo.py`
- Comprehensive documentation:
  - Phase 3.3 design documents (2,000+ lines)
  - Problem type taxonomy
  - Migration guides
  - Completion summary

### Changed

**API Improvements**
- Simplified problem creation with unified factories
- Consistent configuration across all solver types
- Three flexible configuration patterns (YAML, Builder, Presets)
- Clearer separation: problem (math) vs solver (algorithm)

**Code Quality**
- Reduced code duplication through unification
- Better type safety with modern Python typing (`@overload`)
- Improved documentation with comprehensive examples
- Cleaner package structure

### Deprecated

**Problem Classes** (to be removed in v2.0.0)
- `LQMFGProblem` → Use `create_lq_problem()` or `MFGProblem`
- `NetworkMFGProblem` → Use `create_network_problem()` or `MFGProblem`
- `VariationalMFGProblem` → Use `create_variational_problem()` or `MFGProblem`
- `StochasticMFGProblem` → Use `create_stochastic_problem()` or `MFGProblem`

**Config Functions** (to be removed in v2.0.0)
- `create_fast_config()` → Use `presets.fast_solver()`
- `create_accurate_config()` → Use `presets.accurate_solver()`
- `create_research_config()` → Use `presets.research_solver()`
- Old `MFGSolverConfig` → Use new `SolverConfig`

**API Parameters** (to be removed in v2.0.0)
- `solve_mfg(method=...)` → Use `solve_mfg(config=...)`

### Migration Guide

**Old API**:
```python
from mfgarchon.problems import LQMFGProblem
from mfgarchon.config import create_accurate_config
from mfgarchon import solve_mfg

problem = LQMFGProblem(...)
result = solve_mfg(problem, method="accurate")
```

**New API** (Recommended):
```python
from mfgarchon.factory import create_lq_problem
from mfgarchon import solve_mfg

problem = create_lq_problem(...)
result = solve_mfg(problem, config="accurate")
```

### Documentation

- Added comprehensive Phase 3 design documents
- Created migration guides for Phase 3.2 and 3.3
- Updated examples with new unified API
- Added problem type taxonomy
- Created Phase 3 completion summary
- **New User Guides**:
  - `docs/user_guides/particle_interpolation.md` - Complete particle interpolation reference
  - `docs/user_guides/sdf_utilities.md` - Complete SDF utilities reference
  - `docs/migration/PHASE_3_MIGRATION_GUIDE.md` - Phase 3 migration guide
  - `docs/tutorials/01_getting_started.md` - Beginner tutorial
  - `docs/tutorials/02_configuration_patterns.md` - Configuration patterns tutorial

### Technical Details

**Total Changes**:
- ~8,000 lines added/modified
- 21 files changed
- 3 major PRs (#218, #222, #224)
- Full backward compatibility maintained

**Key Benefits**:
- Simpler, more consistent API
- Three flexible configuration patterns
- Better documentation and examples
- Easier to maintain and extend
- Better type safety
- Single source of truth

---

## [0.8.1] - 2025-10-08

### Fixed
- Full nD FP Solver implementation
- Semi-Lagrangian 2D solver
- Bug #8 resolution

---

## Historical Versions

Previous versions (< 0.8.1) were tracked in git history but not formally documented in CHANGELOG.

For detailed historical changes, see:
- Git commit history
- Closed issues and PRs
- Development documentation in `docs/development/`

---

**Note**: Starting with v0.9.0, all changes are documented in this CHANGELOG following semantic versioning and Keep a Changelog standards.
