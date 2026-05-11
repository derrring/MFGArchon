# HJB GFDM solver: boundary residual semantics mismatch with Newton Jacobian

- **Status**: identified 2026-05-11, fix in branch `fix/hjb-bc-newton-residual-semantics`
- **Issue**: [#1116](https://github.com/derrring/MFGArchon/issues/1116)
- **Affected file**: `mfgarchon/alg/numerical/hjb_solvers/hjb_gfdm.py`
- **Affected functions**: `_apply_boundary_conditions_to_sparse_system` (lines 2325–2445), `_apply_boundary_conditions_to_system` (legacy dense path, lines 3170+)
- **Surfaced by**: Stage C v3 2D obstacle evacuation (mixed Dirichlet/Neumann BC + non-trivial Picard initial state)

## TL;DR

The HJB Newton iteration assembles a linear system $J\,\delta = -r$ where, at boundary rows, $J[i,:]$ is the Jacobian of the boundary functional $F_{\text{bc}}(u) = w\cdot u - g$ (e.g., $w=e_i$, $g=g_D$ for Dirichlet; $w$ a normal-derivative stencil, $g=g_N$ for Neumann), but $r[i]$ is set to the BC **target** $g$ rather than the current **violation** $F_{\text{bc}}(u_{\text{current}}) - g$. The two halves of the linear system therefore do not describe a Newton step on the same nonlinear function. Newton stalls whenever the resulting inconsistency is non-trivial — mixed BC + non-zero initial state — and was masked elsewhere by a post-step Dirichlet projection that happens to produce the same iterate for the common case `bc=0`.

## 1. Background — what the BC dispatcher does

`_apply_boundary_conditions_to_sparse_system` rewrites the sparse Jacobian and residual at every boundary row before passing them to `spsolve`. Per BC type:

| BC type | `new_row` | `new_rhs` (current code) |
|---|---|---|
| `DIRICHLET` | $e_i$ | `_eval_bc_dirichlet_value(i, …)` = target value $g_D(x_i, t)$ |
| `NEUMANN`, `NO_FLUX` | normal-derivative stencil $w_i$ | target $g_N(x_i, t)$ (zero for no-flux) |
| `PERIODIC`, `ROBIN` | n/a — raise `NotImplementedError` | n/a |

After this, the Newton loop does:

```python
delta_u = spsolve(jacobian_bc, -residual_bc)
u_trial = u_current + alpha * delta_u
u_trial = self._apply_boundary_conditions_to_solution(u_trial, time_idx)  # Dirichlet-only projection
```

The post-step projection (`_apply_boundary_conditions_to_solution`) **sets** $u_{\text{trial}}[i] = g_D$ at Dirichlet boundary points only. Neumann points are left to be enforced by the linear system.

The row-replacement at line 2442 is `jac_lil[i, :] = new_row` only — no column elimination on $J[:,i]$. This matters for the equivalence analysis below.

## 2. Diagnostic protocol — directional FD Jacobian check

At the stuck Newton iterate (`time_idx = Nt-1`, `_newton_iter = 15` in a stalled run), dump $(J_{\text{bc}}, r_{\text{bc}}, u_{\text{current}})$. For each of 5 random unit vectors $d \in \mathbb{R}^n$ and two perturbation magnitudes $\varepsilon \in \{10^{-6}, 10^{-5}\}$:

$$
\tilde{J}\,d \;=\; \frac{r(u_{\text{current}} + \varepsilon d) - r(u_{\text{current}})}{\varepsilon}
\qquad\text{vs}\qquad
J_{\text{bc}} \cdot d
$$

Both are evaluated **through the same residual pipeline**, including BC dispatch. The error is split by row class:

- **Interior** rows: $\partial r/\partial u$ should match $J$ exactly at machine precision if $J$ is correctly assembled.
- **Boundary** rows: $\partial r/\partial u$ should match the BC stencil $w$ if $r_{\text{bc}}$ is the Newton residual; should be **identically zero** if $r_{\text{bc}}$ is the BC target (a constant in $u$).

The instrumentation lived briefly in the Newton loop and dumped to `/tmp/hjb_J_dump/`; preserved in `mfg-research/experiments/gfdm_monotonicity_audit/minors/exp09_obstacle_navigation_full/_staged_buildup/_diagnostics/fd_jac_NT320_2026-05-11/`.

## 3. Evidence

Stage C v3 with `NT=320`, `N_col=300`, `STAGEC_BETA_TERM=100`, `STAGEC_GHOST=0`. After `dt/4` (vs prior `NT=80`), $J$ itself is well-conditioned: $\kappa_2(J_{\text{bc}}) = 112$, $\sigma_{\min} = 0.75$, `spsolve` residual $\|J\delta+r\|/\|r\| = 3\times 10^{-16}$. Newton still stalls at $\|r\| \approx 493$ from iter 1 onwards; $|\delta|_\infty \approx 3\times 10^{-3}$; Armijo bottoms at `MIN_ALPHA = 9.54e-7` every iteration.

Directional FD check (10 measurements):

```
eps=1e-06 dir=0  full |J·d|=6.924e+01 rel-diff=9.19e-03  interior rel-diff=8.15e-08  boundary |J·d|=6.36e-01 |FD|=0.000e+00
eps=1e-06 dir=1  full |J·d|=6.711e+01 rel-diff=1.03e-02  interior rel-diff=9.14e-08  boundary |J·d|=6.94e-01 |FD|=0.000e+00
eps=1e-06 dir=2  full |J·d|=6.971e+01 rel-diff=8.62e-03  interior rel-diff=7.77e-08  boundary |J·d|=6.01e-01 |FD|=0.000e+00
eps=1e-06 dir=3  full |J·d|=7.012e+01 rel-diff=9.92e-03  interior rel-diff=8.45e-08  boundary |J·d|=6.95e-01 |FD|=0.000e+00
eps=1e-06 dir=4  full |J·d|=7.037e+01 rel-diff=8.52e-03  interior rel-diff=8.76e-08  boundary |J·d|=5.99e-01 |FD|=0.000e+00
eps=1e-05 dir=0  full |J·d|=6.662e+01 rel-diff=1.04e-02  interior rel-diff=9.64e-09  boundary |J·d|=6.91e-01 |FD|=0.000e+00
eps=1e-05 dir=1  full |J·d|=6.953e+01 rel-diff=9.26e-03  interior rel-diff=1.11e-08  boundary |J·d|=6.44e-01 |FD|=0.000e+00
eps=1e-05 dir=2  full |J·d|=6.888e+01 rel-diff=9.43e-03  interior rel-diff=1.74e-08  boundary |J·d|=6.49e-01 |FD|=0.000e+00
eps=1e-05 dir=3  full |J·d|=6.858e+01 rel-diff=9.88e-03  interior rel-diff=9.05e-09  boundary |J·d|=6.78e-01 |FD|=0.000e+00
eps=1e-05 dir=4  full |J·d|=7.002e+01 rel-diff=9.36e-03  interior rel-diff=7.87e-09  boundary |J·d|=6.55e-01 |FD|=0.000e+00
```

| Row class | $\|J\!\cdot\!d\|$ | $\|FD\|$ | rel-diff |
|---|---|---|---|
| Interior (~290 rows) | $\sim 70$ | $\sim 70$ | $\sim 10^{-8}$ — machine precision |
| Boundary (~10 rows) | $0.60$ – $0.70$ | **$0.000\mathrm{e}{+}00$** (every direction, every $\varepsilon$) | 100% |

Interior rows confirm $J_{\text{int}}$ is the true Jacobian of $r_{\text{int}}(u)$ to machine precision. Boundary rows give $\partial r_{\text{bc}}/\partial u \equiv 0$ — $r_{\text{bc}}$ is structurally a constant in $u$.

## 4. Mechanism

Reading `_apply_boundary_conditions_to_sparse_system`:

```python
case BCType.DIRICHLET:
    new_row = np.zeros(n); new_row[i] = 1.0
    new_rhs = self._eval_bc_dirichlet_value(...)        # target g_D
case BCType.NEUMANN | BCType.NO_FLUX:
    new_row, new_rhs = self._build_neumann_bc_row(...)  # new_rhs = target g_N
...
jac_lil[i, :] = new_row
residual_bc[i] = new_rhs
```

The Newton loop solves $J\delta = -r$ where:

- Interior: $J_{\text{int}}\delta = -r_{\text{int}}(u_k)$ — consistent Newton step for the PDE residual.
- Boundary: $J_{\text{bc}}\delta = -g$ — does **not** describe a Newton step on $F_{\text{bc}}(u) - g = 0$, which would require $J_{\text{bc}}\delta = -(F_{\text{bc}}(u_k) - g) = g - F_{\text{bc}}(u_k)$.

Effect per BC type (4 cases):

| BC, target | Current code outcome | Newton-correct outcome |
|---|---|---|
| Dirichlet, $g_D = 0$ | $\delta[i] = 0$; projection sets $u_{\text{new}}[i] = 0$ | $\delta[i] = -u_k[i]$; $u_{\text{new}}[i] = 0$ |
| Dirichlet, $g_D \neq 0$ | $\delta[i] = -g_D$; projection sets $u_{\text{new}}[i] = g_D$ | $\delta[i] = g_D - u_k[i]$; $u_{\text{new}}[i] = g_D$ |
| Neumann, $g_N = 0$ | $w\cdot\delta = 0$ (homogeneous Neumann **on $\delta$**); $w\cdot u_k$ never driven to 0 | $w\cdot\delta = -(w\cdot u_k)$; $w\cdot u_{\text{new}} = 0$ |
| Neumann, $g_N \neq 0$ | $w\cdot\delta = -g_N$ persistently; $w\cdot u_k$ trends away from $g_N$ | $w\cdot\delta = g_N - w\cdot u_k$; $w\cdot u_{\text{new}} = g_N$ |

The Dirichlet rows are rescued by the post-step projection at line 2950 — the projection sets $u_{\text{new}}[i]$ to $g_D$ regardless of the Newton output, so the final iterate satisfies Dirichlet exactly. (The projection is only correct for Dirichlet; Neumann is "no direct solution modification" per the same routine, line 3166.)

The Neumann rows have no projection rescue. The linear constraint that survives is *homogeneous-Neumann-on-$\delta$* (when $g_N = 0$), which preserves the BC violation $w\cdot u_k$ over every Newton iteration. Starting from an initial guess $u_0$ that does not satisfy Neumann — generic in Picard with non-zero $u^{n+1}$ initial state in the backward sweep, or with any post-restart resumption — $w\cdot u_k = w\cdot u_0 \neq 0$ persists. The persistent boundary violation contaminates stencil-evaluated $\nabla u$ at adjacent interior points (boundary stencils sample 1–2 layers into the interior — universal in GFDM on irregular 2D clouds), so $r_{\text{int}}(u_k)$ also fails to converge to zero. This is the proximate mechanism for the observed Stage C plateau: interior $r$ stuck at $\sim 493$ because the boundary $u$ values it depends on are stuck.

The pathology is **independent of $\kappa(J)$**, transport-vs-CFL balance, or `qp_optimization_level`. `dt/4` reduces $\kappa(J)$ by $224\times$ but cannot fix the BC mismatch; Newton remains stalled at the same $\|r\|$ level. Howard's policy iteration would have the same issue: the inner linear PDE solver calls the same BC assembly path.

## 5. Why this didn't surface earlier

Three independent conditions must coincide:

1. **At least one Neumann (or non-zero Dirichlet) BC** — otherwise the projection rescue covers everything.
2. **Initial guess violates the BC** — $w\cdot u_0 \neq 0$ for Neumann, or $u_0[i] \neq g_D$ for Dirichlet with $g_D \neq 0$. Constant-zero initial guess (LQ benchmark default) satisfies homogeneous Neumann trivially and zero Dirichlet trivially.
3. **Boundary stencils sample interior** — true in GFDM for any cloud where boundary points have neighbors at depth $\geq 1$, universal in 2D irregular clouds.

Historical coverage:

- 1D LQ corridor benchmarks (`tests/unit/test_alg/test_hjb_gfdm*.py`): pure Dirichlet $g_D = 0$ + zero initial guess. (1) and (2) both violated, projection rescues. Bug dormant.
- Adjoint-consistent BC (Issue #574, #625): ROBIN raises `NotImplementedError`; the coupling-layer `BCValueProvider` resolves intent to concrete values before this stage. Distinct path, not affected.
- Reflecting-BC FP-side studies: FP is independent of the HJB BC assembly. Not affected.
- Stage C v3 (this report): mixed Dirichlet exit + Neumann walls + Neumann obstacle + Picard initial state with $u^{N_t} = g_{\text{terminal}}$ (which does not satisfy Neumann at walls/obstacle). All three conditions hold.

## 6. Fix

Plumb `u_current` into the BC dispatcher and rewrite `new_rhs` to encode the current violation:

```python
case BCType.DIRICHLET:
    new_row = np.zeros(n); new_row[i] = 1.0
    bc_target = self._eval_bc_dirichlet_value(i, segment, legacy_bc_values, current_time)
    new_rhs = u_current[i] - bc_target

case BCType.NEUMANN | BCType.NO_FLUX:
    new_row, bc_target = self._build_neumann_bc_row(
        i, normal, dimension, segment, legacy_bc_values, current_time
    )
    new_rhs = float(new_row @ u_current) - bc_target
```

Two call sites in `solve_hjb_at_step` (one for each Newton path branch — the sparse path at line 2913, none of the legacy dense `_apply_boundary_conditions_to_system` paths are active in the joint_socp regime but the same fix applies for consistency).

The post-step Dirichlet projection (line 2950) becomes redundant after the fix — the Newton step already exactly enforces Dirichlet. Leave the projection for one release cycle as a safety net, then remove with a deprecation note.

## 7. Verification protocol

Five sequential gates. Each gate must pass before the next.

### Gate A — FD Jacobian check on a unit problem

Build a deliberately-uncomfortable 2D HJB sub-problem (mixed Dirichlet/Neumann, non-trivial initial state, ~50 collocation points). At iter 0 of Newton, run the same directional FD check used in §3. Assert:

- Interior `rel-diff < 1e-6`.
- Boundary `rel-diff < 1e-6` (was `≈ 1.0` pre-fix).
- Full `rel-diff < 1e-6` (was `≈ 1e-2` pre-fix).

This becomes a permanent regression test (`tests/unit/test_alg/test_hjb_gfdm_bc_jacobian_consistency.py` or similar).

### Gate B — 1D LQ Dirichlet `bc=0` regression

`pattern a` (row-only replacement, no column elimination on $J[:,i]$, verified by grep) implies:

- If $u_{\text{init}}[\text{bd}] = 0$ exactly (typical of LQ benchmark with explicit zero IC at boundary): **bit-exact** equivalence. Old: $\delta[i] = 0$ ⇒ $J[j,i]\cdot\delta[i] = 0$ for interior $j$. New: $\delta[i] = -(0 - 0) = 0$, same. Interior $\delta$ identical.
- If $u_{\text{init}}[\text{bd}]$ has any roundoff (e.g., from a callable IC evaluated at boundary points with finite precision): **roundoff-level** deviation. Old: $\delta[i] = 0$, projection forces $u[i] = 0$. New: $\delta[i] = -u_{\text{init}}[i] \sim 10^{-15}$, propagates via $J[j,i]\cdot\delta[i]$ to interior $\delta$ at the $10^{-15}\cdot\|J[:,i]\|$ level.

Acceptance: any deviation beyond $10^{-12}$ relative is a red flag and must be investigated before continuing.

### Gate C — 1D Neumann `bc=0` regression (if any exists in the test suite)

Caveat noted in #1116 review: the equivalence claim "Neumann bc=0 with $w\cdot u_{\text{init}}=0$ is mathematically equivalent" only holds at the **first time step**. During time-stepping, $w\cdot u^n$ drifts as the interior PDE solution evolves; old code does not correct this drift (the linear constraint preserves it), new code does. EOC measurements may show small differences depending on drift magnitude. **Not assumed equivalent — measured.**

Acceptance: if EOC table changes by more than $\pm 0.05$ in order, document the difference in the test report. Keep the historical numbers in the test docstring with a `[SUPERSEDED 2026-05]` tag pointing here.

### Gate D — Stage C v3 e2e

`STAGEC_NT=80 STAGEC_NCOL=300 STAGEC_BETA_TERM=100 STAGEC_GHOST=0` (the baseline that previously stalled at $\|r\| \approx 559$):

- Newton residual should drop below `newton_tolerance` within `max_newton_iterations`.
- Picard should converge (previously did not, because inner Newton did not).
- Asymmetry index — previously reported as "85% improvement" — must be re-measured. Prior number is on a non-converged Newton (interior $u^0 \approx \text{const}\cdot g_{\text{terminal}}$ because $\delta$ never updated $u$ meaningfully); the asymmetry was a geometric echo of the terminal cost, not of a converged HJB solution.

### Gate E — Adjacent test suite

Run `pytest tests/unit/test_alg/` and watch for unexpected pass/fail flips. The BC change affects every HJB sub-step in every test; if anything regresses, it indicates an additional code path with hidden dependence on the old (broken) BC semantics.

## 8. Scope of invalidation for prior results

| Result class | Status under new code |
|---|---|
| 1D LQ Dirichlet `bc=0` EOC studies (§main_validation paper) | Bit-exact or roundoff-level, no re-run needed (verify per Gate B) |
| 1D Neumann `bc=0` with constant IC | Re-run if EOC published; may differ at the $\pm 0.05$-order level (verify per Gate C) |
| 1D mixed BC, non-zero target | Re-run all |
| 2D irregular cloud, any non-trivial state (Stage A/B/C series) | **Re-run all** |
| Reported Stage C v3 "85% asymmetry improvement" | Invalid — measured on non-converged Newton. Re-measure. |
| Reported Stage C v3 convergence diagnostics (`|r|` history, Newton iter counts) | Invalid — pre-fix Newton was solving the wrong system |
| Reflecting-BC reference solutions for Adjoint-Consistent Provider studies (Issue #574, #625) | Not affected — distinct code path |

For paper writing: the §main_validation 1D LQ results survive (pending Gate B). The §main_demo 2D obstacle navigation needs full re-run. The Stage C → paper §main_demo handoff timeline shifts by one Picard sweep.

## 9. Reproduction recipe

The full pre-fix stalled-Newton diagnostic, exactly as captured for this report:

```bash
# In mfgarchon, on the pre-fix branch:
git checkout main  # before fix commit, with DIAG instrumentation stashed
# Then apply DIAG[fd-jac] block from this branch's stashed diff if reproducing FD numbers.

# In mfg-research:
cd experiments/gfdm_monotonicity_audit/minors/exp09_obstacle_navigation_full/_staged_buildup
STAGEC_NT=320 STAGEC_NCOL=300 STAGEC_NP=20000 STAGEC_NPICARD=2 \
STAGEC_GHOST=0 STAGEC_BETA_TERM=100 \
python stageC_v3_geodesic_congestion.py 2>&1 | tee /tmp/fd_diag_NT320.log

# Diagnostic artifacts deposited to /tmp/hjb_J_dump/, mirrored under
# experiments/.../  _diagnostics/fd_jac_NT320_2026-05-11/
```

Artifacts preserved for the lifetime of the paper:
- `_diagnostics/fd_jac_NT320_2026-05-11/J_stuck.npz` — sparse Jacobian at stuck iterate
- `_diagnostics/fd_jac_NT320_2026-05-11/r_stuck.npy`, `u_stuck.npy`, `delta_stuck.npy`
- `_diagnostics/fd_jac_NT320_2026-05-11/fd_log.txt` — the 10-line FD measurement table reproduced in §3
- `_diagnostics/fd_jac_NT320_2026-05-11/boundary_indices.npy`, `bc_types.npy` — for post-hoc row-class auditing
