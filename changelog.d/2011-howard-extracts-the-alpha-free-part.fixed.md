`inner_solver='howard'` no longer drops a Hamiltonian's alpha-free part when the class does not
expose `_potential` or `_coupling` (#2011). It refused such a Hamiltonian; it now solves it.

The switch that builds Howard's running-cost closure was keyed on those two `SeparableHamiltonian`
internals, so any other `HamiltonianBase` subclass carrying an `F(x, m)` had it discarded — the
issue measured `max|u(g=1) − u(g=0)| = 0.0000e+00` on Howard against `1.469e-01` on Newton for
`H = |p|²/2 + g·x·m`. Measured now: **1.4687e-01 on Howard against
1.4687e-01 on Newton** under this file's `M_MATRIX_QP` scheme — the two agree to the printed
digits, which is stronger than the 5% the test asserts and stronger than the 0.7% recorded when
this file still used `joint_socp`/`precompute`. The scheme, not the extraction, is what moves
Howard: under that older scheme it gives 1.4586e-01 and Newton is unchanged at 1.4687e-01.

**No new extraction machinery was needed.** `howard_running_cost` already evaluated the Hamiltonian
at `p = 0` through the same `eval_H_batch` the Newton residual uses, which requires no private
attribute. Only the *switch* deciding whether to build that closure was keyed on the internals. It
is now a **union**: the declared route is kept as a fast path and the guard's own measurement of
`|H(x, m, 0, t)|` is added, so the change can only admit more than before, never less. Replacing
the declared route rather than joining it would have dropped a wired potential that happens to
vanish at the probe's sample points.

The refusal for an unwired alpha-free part is gone, since it is now computed. **The `_ke` refusal
stays, and it is the real limit**: Howard substitutes a quadratic Lagrangian, and no probe recovers
a control cost that is not quadratic.

The test file's `_REFUSE` list is split on that line, and the split landed on the physics rather
than being drawn. `congestion_c1_is_one` (`½|p|²/m`), `quartic_kinetic` (`½|p|⁴`) and
`anisotropic_kinetic` all have a non-unit-quadratic control cost and stay refusals; the three
`½|p|² + F(x,m)` entries move to a new `_EXTRACT` list, where each must be **accepted and change
the answer** — accepting without acting is the pre-fix behaviour wearing a green test.

On soundness: `H(x, m, 0, t)` need not *be* the alpha-free part, and the guarantee does not rest on
`H_control(0) = 0`. Howard assembles `ref(∇u) + H(x, m, 0, t)` and `_ke` certifies
`ref(p) = H(p) − H(0)`, so the sum telescopes back to `H(∇u)` whatever the extracted value is.

The check that carries this is pointwise and algebraic, not a solve comparison:
`−α*·p − L(α*) + H(x, m, 0, t) == H(x, m, p, t)` at `α* = −∂H/∂p`. It is exact on every
algebraically-exact class — including alpha-free parts depending on `m` at amplitude `1e4` — and
broken on every refused one. What acceptance guarantees is not exactness but `_ke ≤ tol`: a
barely-accepted class sits at that bound. A solve comparison is the weaker instrument and was nearly vacuous as first stated: a
perturbation constant in `x` and `m` shifts both solvers identically whatever the guard does, so it
separates nothing. The identity is pointwise in `(x, m, t)`, which is why the probe must cover every
time slice — see the companion fragment. One hypothesis is not checked by the guard and is pinned
elsewhere (#1645, `test_hl_convention.py`) for the **declared** path: that `control_cost.evaluate`
and its Lagrangian are a Legendre pair with no additive constant. That test's `CONTROL_COSTS` covers
quadratic, bounded and l1, and does not reach the fallback path — which does not need it, since
`hjb_howard.py` and `_kinetic_ref` both hardcode the unit quadratic and are conjugate by
construction.

`H = sqrt(1+|p|²)` is refused on the separate, real ground that its control cost is not
unit-quadratic — `_ke = 6.499e+01` against a tolerance of `1.2e-09` — and it now has a fixture in
`_REFUSE` asserting exactly that, so the pair can fail instead of going stale as it did once already.
