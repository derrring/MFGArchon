- **The 1-D FDM Jacobian's advection block is now built from the gradient the residual applies**
  (Issue #1896, items 3 and 4). It restated the stencil by hand — central as `[-1/2dx, 0, +1/2dx]`,
  upwind as a one-sided pair — which is right in the interior and wrong at both walls under every
  BC. Measured by column-wise finite difference of `compute_hjb_residual` at `Nx=9`:

  | BC | scheme | true row 0 | hardcoded row 0 |
  |:--|:--|:--|:--|
  | no_flux | central | `{0: -4, 1: +4}` | `{1: +4}` |
  | dirichlet | central | `{0: +4, 1: +4}` | `{1: +4}` |
  | periodic | central | `{1: +4, 8: -4}` | `{1: +4}` |
  | no_flux | upwind | `{}` | `{0: +8}` |
  | dirichlet | upwind | `{0: +16}` | `{0: +8}` |
  | periodic | upwind | `{0: -8, 1: +8}` | `{0: +8}` |

  The missing central entry has the **opposite sign** between no-flux and Dirichlet, so no single
  correction to the hardcoded row could have covered both.

- **Why nothing caught it.** `use_upwind=True` is the default, and the no-flux upwind row is the one
  cell where the defect is invisible: the true row is empty *and* the BC forces `p = 0` at the wall,
  so `dH/dp` multiplies the spurious diagonal away. That is the configuration the `fdm_upwind`
  capability fixture runs. Under Dirichlet or periodic nothing masks it. Same shape as #1894, which
  fixed the diffusion block of the same function and was identically absent at `σ = 0`.

- **The branch was also selected by a second rule** (item 3). `gradient_upwind` selects on
  `sign(central)`; the Jacobian selected on `sign(grad_upwind)`. They coincide only where the two
  share a sign — measured to disagree at **8 of 41 nodes** on a random field, 2 of 41 on
  `sin(4πx)`, 1 of 41 on `|x-0.5|`, and **0 of 41 on the monotone `x²`** that the tests used.

- **Extracted, not restated — and for upwind, not extractable either.** The upwind gradient is not
  linear in `U`, and probing it does not merely lose accuracy: at a node where the branch is
  switching, the directional difference disagrees with itself and the extractor raises (9 of 15
  measured wall cells at `Nx=9`, one-sided differences as well as two-sided). That is not an
  implementation fault — the map is nondifferentiable there and no Jacobian exists, only a Clarke
  generalised Jacobian. So the bands are assembled from the two operators that *are* linear and
  already have owners:

  ```
  forward  = central + (dx/2) * laplacian
  backward = central - (dx/2) * laplacian
  ```

  exact identities of the ghost-padded stencils, walls included — measured at `1.4e-14` over seven
  BCs (no-flux, Dirichlet, Neumann, periodic, three Robin parameter sets) and three states, and
  pinned at `1e-12`. Which
  branch holds at a row is then **observed** rather than restated, which is what closes item 3:
  observation cannot disagree with the residual, because it is a measurement of it.

- **Observing the branch takes two steps, and the second one was found by measurement.** Comparing
  `grad_upwind(U)` against both reconstructions is exact where they differ in value — but they agree
  in value on every locally linear stretch, and the two *rows* still differ there. Reading values
  alone put the whole Jacobian one column across on a piecewise-linear state: `4.000e+01`,
  ε-independent, against a column-wise finite difference at `Nx=21`. Those rows are now decided by
  how `grad_upwind` **moves** under an alternating probe, whose central difference vanishes at every
  interior node and so cannot move the branch it is measuring. Where the probe is inconclusive the
  branch is switching at `U`, and both rows are admissible.

- **Result**, column-wise finite difference against the assembled Jacobian, `Nx=21`, over 3 BCs × 2
  schemes × 4 states: every wall row and every interior row agrees to `< 1e-5`, worst `1.19e-07`
  (Dirichlet, central, monotone), which is the finite difference's own truncation floor. The two residual disagreements are the instrument, not the Jacobian, and each was
  measured rather than assumed: at a switching node the two-sided difference averages two one-sided
  operators and equals neither (the Jacobian row converges to one of them exactly, `4e-1 → 4e-2 →
  4e-3` as the isolating tilt shrinks 10× each time, while the branches stay `2e+01` apart); and at
  the flat state `u ≡ 0` the advection term is quadratic in `p`, so the difference leaves an O(ε)
  tail — `1e-2 → 1e-4 → 1e-6` at ε = `1e-4 / 1e-6 / 1e-8`.

- **Cost**: the analytic path stays O(Nx) and is ~2.0× slower — `3.6 → 7.3` ms at `Nx=1601`,
  `1.9 → 3.9` ms at 801, `0.35 → 0.76` ms at 21 — the ratio flat across `Nx`, so no complexity
  regression. The Laplacian bands are extracted once per Jacobian and shared by both blocks; before
  that they were extracted twice, worth `10.9 → 7.3` ms at `Nx=1601`. The FD fallback path is
  untouched. #1607's reason for the analytic path survives: the alternative is O(Nx²).

- **Oracle**: `tests/unit/test_alg/test_hjb_jacobian_advection_1896.py`, external — a column-wise
  finite difference of the residual is a law the Jacobian must reproduce, computed independently of
  it. Mutation-verified, seven mutations, all killed: neutering the tie-break probe (3 tests),
  restating the branch rule as `sign(grad_upwind)` (15), inverting the mask (20), dropping the
  per-row sign on the wrap entries (6), dropping the wrap entries (7), using central bands for
  upwind (20), and removing the identity guard (1). Reverting the whole change turns 34 of 84 red.
  The file also carries its own positive controls: that the states it calls smooth have no switching
  node (otherwise the exclusion would silently make every assertion vacuous), and that the two
  branch-selection rules actually part on the fixture (otherwise item 3's test proves nothing).

- **The analytic block is opt-in.** It is reached only when `backend is None`, which is
  `HJBFDMSolver(analytic_jacobian=True)` (#1607); the default routes to the per-point FD fallback.
  An earlier measurement of this fix passed a NumPy backend and reproduced the baseline byte for
  byte — it had been measuring the other path.

- **One reference test moved, and was not adjusted to match.**
  `test_jacobian_byte_identical_to_inline_assembly` (#1071) rebuilt the Jacobian from a hand-written
  advection stencil selected by `grad >= 0` on the **upwind** gradient — carrying both defects at
  once, so it pinned them rather than its own subject. That subject is the inline `dp` form; the
  reference now takes its advection bands from the same owner, exactly as #1894 did for its
  diffusion bands. Both stencil halves are tautological there now and are pinned externally instead.
  What the test still discriminates is `dH/dp` — verified, not assumed: perturbing the assembled
  `dH_dp` by one part in 10⁵ turns it red.
