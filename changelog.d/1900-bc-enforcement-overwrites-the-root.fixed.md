- **The 1-D HJB inner Newton no longer overwrites the root it just found** (Issue #1900).
  `solve_hjb_timestep_newton` ran Newton to convergence and then rewrote the boundary values with a
  *different* discretisation of the same boundary condition, so the array it returned was not a root
  — and `converged` was `True`, which means the non-convergence warning, the only instrument that
  existed, was structurally blind to it.

  Measured on `scripts/capability_matrix.py::_smoke_problem`, 5 Picard sweeps, 50 inner solves:

  | | before | after |
  |:--|--:|--:|
  | reported-converged solves returning a **non-root** | **11 of 11** | **0 of 31** |
  | worst residual among reported-converged | 2.417e+03 | **8.433e-07** |
  | array modified by the enforcement | 50 of 50 | — |

  At `t_idx=9` Newton reached `4.424e-07` against a `1e-6` tolerance, reported success, and returned
  an array whose residual was `4.338e-02` — five orders worse, `4×10⁴` times the tolerance it had
  just certified.

- **Three implementations of no-flux were in play**, which makes this #1894/#1896's class one layer
  up:

  | implementation | zero-flux form | order | used by |
  |:--|:--|:--|:--|
  | `pad_array_with_ghosts` | mirror ghost | O(h²) | **the residual** — defines the discrete problem |
  | `enforce_neumann_value_nd` | `u[0] = (4u[1] − u[2])/3` | O(h²) | exists; called by no solver for Neumann |
  | `base_hjb`, hand-rolled 1-D | `u[0] = u[1] − g·dx` | O(h) | **deleted here** |

  With `g = 0` the third collapses to `u[0] = u[1]`, confirmed exactly on the returned arrays
  (`u[0] − u[1] = +0.000e+00` at every timestep). The residual never asked for that: it solves for
  `u[0]` from the PDE with a mirrored ghost, like every other node.

- **Which implementation owns the condition was measured, per BC type, not argued.** Over
  `Nx ∈ {41, 81, 161, 321, 641}` at fixed `t_idx`:

  | BC | pre-enforcement `u[0]` | returned `u[0]` | owner |
  |:--|:--|:--|:--|
  | no_flux | converges | **same limit**, gap falling `8.9e-03 → 4.0e-05` (`O(h²)`) | **the residual** |
  | dirichlet(0.7) | `0.589 → 0.696`, only *approaches* | exactly `0.700000000000` | **the enforcement** |
  | robin | `0.593 → 0.501` | exactly `0.500000000000` | **the enforcement** |
  | periodic | — | — | already a no-op |

  So only the no-flux/Neumann branch is deleted. Dirichlet and Robin keep theirs: the ghost padding
  never pins a prescribed value, and #542's own discussion point 2 named the right fix — row
  replacement in the residual **and** the Jacobian — which is a separate change. Both are recorded
  as `xfail(strict=True)` against the law below, so they redden the day that lands.

- **It also moves #1878 and #1873**, which was a hypothesis in the issue and is now measured:

  | | before | after |
  |:--|--:|--:|
  | inner-Newton non-convergence warnings, 5 sweeps | 39 of 50 | **19 of 50** |
  | same, 20 sweeps | 176 of 200 | **124 of 200** |
  | outer `err_U` after 20 sweeps | 1.177e+02 | **2.181e-01** |

  Each sweep had been starting from the previous sweep's corrupted boundary. Note the 5-sweep
  `err_U` gets *worse* (1.643e+02 → 1.489e+03) before the 20-sweep figure improves by 540×; the
  early iterates are noisier and the endpoint is far better.

- **Second defect in the same block, previously inert.** It computed its own spacing as
  `span / Nx` where `Nx` is the *point* count — `1/21` against a true `1/20`, off by `Nx/(Nx−1)`.
  Inert under no-flux because it multiplied `g = 0`; live for Robin, where it enters the denominator
  `alpha + beta/dx`. It now asks the geometry that owns the spacing instead of re-deriving it, which
  is what stops a fourth instance (#1889, #1896 item 8 are the other two).

- **Oracle**: `tests/unit/test_alg/test_hjb_bc_enforcement_1900.py`. A law, not an agreement test —
  *if the solver reports `converged`, the array it returns must be a root of the residual it
  certified* — and it cannot go tautological, because `F` is defined by the residual while the claim
  is made by the solver. Mutation-verified: restoring the no-flux enforcement reddens 3 (both law
  cases and the vacuity control), restoring the wrong spacing reddens exactly 1. The file carries
  its own controls: that no-flux genuinely converges here (otherwise the law skips its way to green),
  and that the two candidate spacings differ measurably on the fixture.

- **Found while writing the oracle, filed separately**: a uniform `robin_bc()` is silently enforced
  as **Dirichlet**. Its single segment carries no `face`, so `_get_bc_info_1d`'s priority-1 loop
  never matches and priority 2 returns the hardcoded `alpha=1.0, beta=0.0` defaults, discarding the
  segment's real coefficients — measured at `(1.0, 1.0) → (1.0, 0.0)` and `(2.0, −1.0) → (1.0, 0.0)`.
  With `beta = 0` the Robin formula collapses to `u[0] = g`. A *faced* segment works correctly, which
  is why this was invisible: the adjoint-consistent BC example in `CLAUDE.md` uses `boundary="x_min"`.
  Recorded as `xfail(strict=True)` asserting the correct behaviour.
