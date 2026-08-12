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
  `sign(central)`; the Jacobian selected on `sign(grad_upwind)`. How far they part on a noisy field
  is a random variable — at `Nx=41`, median **10 of 41**, range 6–16 over 200 seeds. (#1896's
  inventory quoted "8 of 41" from a single unseeded draw; that draw is inside this range, so the
  number was not wrong, but stating it as a constant was.) Deterministic fixtures: 2 of 41 on
  `sin(4πx)`, 1 of 41 on `|x-0.5|`, and **0 of 41 on the monotone `x²`** that the tests used —
  which is the load-bearing half and reproduces exactly.

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
  BCs (no-flux, Dirichlet, Neumann, periodic, three Robin parameter sets) × three states, and pinned
  at `1e-12`, since it is the premise the whole construction rests on.

- **Which branch holds is measured, wherever a measurement exists.** Where forward and backward
  differ in value, `grad_upwind(U)` equals exactly one of them and the branch is read off it. That
  is what closes item 3, and it covers every row where the two rules could have parted: a rule
  disagreement that does not change the value cannot change which stencil produced it either. Where
  they agree in value — every locally linear stretch, so not a rare coincidence — the two *rows*
  still differ and nothing observable separates them, so `sign(central)` decides. Choosing by the
  tie alone was worth `4.000e+01`, ε-independent, at `Nx=21`.

- **Result**, column-wise finite difference against the assembled Jacobian, `Nx=21`, over 4 BCs × 2
  schemes × 4 states: every wall row and every interior row agrees to `< 1e-5`, worst `1.19e-07`
  (Dirichlet, central, monotone), which is the finite difference's own truncation floor. The two
  residual disagreements are the **instrument**, not the Jacobian, and each was measured rather than
  assumed: at a switching node the two-sided difference averages two one-sided operators and equals
  neither (the Jacobian row converges to one of them exactly, `4e-1 → 4e-2 → 4e-3` as the isolating
  tilt shrinks 10× each time, while the branches stay `2e+01` apart); and at the flat state `u ≡ 0`
  the advection term is quadratic in `p`, so the difference leaves an O(ε) tail — `1e-2 → 1e-4 →
  1e-6` at ε = `1e-4 / 1e-6 / 1e-8`.

- **Cost**: the analytic path stays O(Nx) and is ~2.1× slower — `3.6 → 7.4` ms at `Nx=1601`,
  `1.9 → 3.8` ms at 801, `0.35 → 0.73` ms at 21 — the ratio flat across `Nx`, so no complexity
  regression, and the probe count is constant in `Nx` for every BC. The Laplacian bands are
  extracted once per Jacobian and shared by both blocks; before that they were extracted twice,
  worth `10.9 → 7.4` ms at `Nx=1601`. The FD fallback path is untouched. #1607's reason for the
  analytic path survives: the alternative is O(Nx²).

- **Oracle**: `tests/unit/test_alg/test_hjb_jacobian_advection_1896.py`, external — a column-wise
  finite difference of the residual is a law the Jacobian must reproduce, computed independently of
  it. Mutation-verified across three files (122 tests) — twelve mutations, eleven killed and one
  proved equivalent:

  | mutation | tests killed |
  |:--|--:|
  | invert the mask | 30 |
  | use central bands for upwind | 28 |
  | restate the branch rule as `sign(grad_upwind)` (item 3) | 21 |
  | drop the wrap entries entirely | 8 |
  | drop the per-row sign on the wrap entries | 7 |
  | revert the tie-break to bare backward-preference | 5 |
  | invert the tie-break predicate | 5 |
  | let the tie-break decide every row (measurement never fires) | 1 |
  | keep cancelled wrap entries instead of dropping them | 1 |
  | skip the identity guard | 1 |
  | revert the identity-guard scale to the survivor only (F1) | 1 |
  | freeze `current_time` to 0 in the advection call | **equivalent — see below** |

  Reverting the whole change turns 52 of 95 red. The file carries its own positive controls: that
  the states it calls smooth have no switching node (otherwise the exclusion silently makes every
  assertion vacuous), and that the two branch-selection rules actually part on the fixture
  (otherwise item 3's test proves nothing).

- **What review changed, and it was not cosmetic.** The first version resolved tied rows by probing
  how `grad_upwind` *moves* under an alternating direction, on the principle that observing beats
  restating. Review found that probe **structurally blind at a Robin wall with `alpha == beta`**: the
  ghost is `u_ghost = a·u[0] + c` with `a = (2 + α/β)/(2 − α/β)`, so `a = 3` exactly there, and the
  probe's separation at row 0 is `|a − 3|/dx` — identically zero, for every state. On a state whose
  Laplacian also vanishes at the wall the value comparison ties too, and both blind gave the wrong
  branch on a row that is *not* a switching node: `2.0000e+01`, reproduced independently.

  Fixing that made the probe redundant, and the mutation suite said so — neutering it killed zero
  tests. A sweep over **27,345** tied rows (9 grid sizes × 4 spacings × 13 BCs × 36 states) found the
  probe and `sign(central)` never once deciding a row differently, so it was deleted: 14 lines, four
  operator applications, and three magic constants. Deleting it did **not** measurably change
  runtime (`7.31 → 7.44` ms at `Nx=1601`, within noise); it bought simplicity and removed the blind
  spot.

- **The remaining restatement is pinned by a test, not by an argument.** On every configuration this
  repo can build, measuring the branch and restating `sign(central) >= 0` agree — so no ordinary test
  separates them, and "let the tie-break decide everything" killed nothing. The difference only
  appears when the rule *changes*, which is precisely what item 3 was. So
  `test_the_jacobian_follows_a_changed_selection_rule_without_being_told` inverts `gradient_upwind`
  itself and asserts the Jacobian still linearises the residual. A Jacobian that measures follows;
  one that restates cannot.

- **One reference test moved, and was not adjusted to match.**
  `test_jacobian_byte_identical_to_inline_assembly` (#1071) rebuilt the Jacobian from a hand-written
  advection stencil selected by `grad >= 0` on the **upwind** gradient — carrying both defects at
  once, so it pinned them rather than its own subject. That subject is the inline `dp` form; the
  reference now takes its advection bands from the same owner, exactly as #1894 did for its
  diffusion bands. Both stencil halves are tautological there now and are pinned externally instead.
  What the test still discriminates is `dH/dp` — verified, not assumed: perturbing the assembled
  `dH_dp` by one part in 10⁵ turns it red.

- **The analytic block is opt-in.** It is reached only when `backend is None`, which is
  `HJBFDMSolver(analytic_jacobian=True)` (#1607); the default routes to the per-point FD fallback.
  An earlier measurement of this fix passed a NumPy backend and reproduced the baseline byte for
  byte — it had been measuring the other path. The FD fallback remains wrong at periodic row 0
  (`4.145e+01`) and on upwind interior rows (`7.814e+01`) at `Nx=21`; pre-existing, measured during
  review, not addressed by this change.

- **Corrected while here**: `_extract_bands`' docstring claimed its O(Nx²) tier was "reached in
  practice rather than defensively" because obstacle masks, nonlocal terms and Robin BCs defeat the
  comb. Measured false — tier 1 succeeds for every BC constructible in this repo, at a constant 9
  probes for every `Nx`, and nothing in-tree builds a banded-plus-something operator. The claim is
  struck rather than deleted, with the measurement in its place.

- **Second review round — MERGE-OK, three findings, all taken.**

  **F1, introduced.** The identity guard scaled its tolerance by the *survivor* while its error is
  set by the *cancelled* terms. `forward`/`backward` recover a small number by cancelling `g_c`
  against `(dx/2)·laplacian`, so the reconstruction's rounding floor is `eps ×` the cancelled
  magnitude, while `atol` was `1e-9 × max(|g_up|, 1)` — and an inhomogeneous wall points both end
  rows' gradient inward, so a large ghost never reaches `g_up` at all. Reproduced independently: at
  `Nx=51` with a Dirichlet value of `1e6`, mismatch `4.172e-09` against an atol of `1.98e-09` with
  the cancelled terms at `5e+07`. It escapes to the user — `compute_hjb_jacobian` is called outside
  the `try` in `newton_hjb_step`. A false alarm rather than a missed detection: where it tripped,
  `|forward − backward|` is 2× the cancelled magnitude while the error is `eps` times it, so the
  branch measurement is immune by `1/eps` to the very error being flagged. The scale now includes
  the cancelled magnitudes; pinned by `test_a_large_wall_value_does_not_trip_the_identity_guard`.

  **F2 — my own correction was wrong, and wrong in the class it was documenting.** The
  `[CORRECTED 2026-08-12]` note in `_extract_bands` claimed tier 1 succeeds "at a constant 9 probes
  for every Nx" and that the fallback "is reached by no in-tree configuration". Both halves false.
  Measured: 4 / 5 / 6 probes at `Nx = 3 / 4 / 5`, **12 at `Nx = 6`**, a flat **8** for every
  `Nx ≥ 7`. At `Nx=6`, `edges` takes `{0,1,4,5}` leaving `interior=[2,3]`, whose length below 3
  collapses `stride` to 1, so the single comb probes two *adjacent* columns and the control vector
  correctly fails. `tests/conftest.py`'s `tiny_problem` is `Nx=6`, so it fires in-tree on every run.
  The "9" was lifted from the first review, which had measured `Nx ∈ {9,21,101,401}`, and restated
  as a universal without carrying its population — a verdict without its denominator, written into
  the correction that was documenting exactly that. Now pinned by
  `test_nx_6_degenerates_the_comb_and_that_is_recorded_rather_than_assumed_away`, which also asserts
  the bands are still exact there (tier 2 is exact for any structure), so this records a cost of 8
  extra applies on the smallest grid in the tree, not a defect.

  **F3, nit.** `test_the_flat_state_the_repo_defaults_to_is_a_total_branch_degeneracy` cannot see
  the advection block at all: at `u ≡ 0`, `max|dH/dp| = 0`, and review showed that replacing the
  entire output of `_advection_bands` with zero bands changes the assembled Jacobian by exactly
  `0.0` there. The docstring already said the true derivative is zero; only the *name* overclaimed.
  Renamed to `test_the_flat_state_leaves_an_eps_tail_not_a_defect`, and the degeneracy is now a
  **measured premise** (`assert max|dH_dp| == 0`) rather than an implication of the title. The tie
  case it appeared to cover is covered for real by `piecewise_linear` and by
  `test_a_tied_wall_row_...`, both tied in value with `dH/dp ≠ 0`.

- **A coverage gap that turned out to be an equivalent mutation.** Review flagged that freezing
  `current_time` to `0.0` in the `_advection_bands` call survives the whole suite, and named it the
  cheapest gap to close. It is not a gap: a BC value enters the ghost as an affine **offset**, and
  `_advection_bands` subtracts the zero-state, so the offset cancels by construction. Measured over
  time-dependent Dirichlet, Neumann and Robin, both schemes: `max|bands(t=0) − bands(t=0.6)|` is
  `0.0` to `1.4e-14`. Only `alpha`/`beta` could move a coefficient and no constructor makes those
  time-dependent, so the threading **cannot** affect this function's output for any BC constructible
  today. `test_a_time_dependent_boundary_reaches_the_advection_block` is kept anyway, because
  `dH/dp` *does* vary with time and the end-to-end property (the Jacobian linearises the residual at
  the time it was handed) is real. Its fixture carries the sign lesson: with a large **positive**
  wall value, `central < 0` at both walls, upwinding selects the branch that never reads the ghost,
  and the test would have asserted nothing while looking fine — caught by its own control.

- **Five findings were refuted** by the second round's skeptics, including one filed as a blocker:
  that the test advertised as pinning the `sign(central)` restatement pins nothing. Direct mutation
  says otherwise — replacing the whole recovery with `took_backward = g_c >= 0` gives 1 failed, and
  the single failure *is* that test; the proposed "safe" fix turns it red at `4.000e+01`. Also
  refuted: that the probe deletion removed a real discriminator (two independent sweeps, 239,138 and
  160,280 tied rows, zero disagreements — roughly 10× this PR's own 27,345), and that the
  linearisation point is unpinned (behaviourally real, but `main` carries the identical unpinning at
  `e3fdd10c:1005`, so pre-existing).
