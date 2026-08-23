`inner_solver='howard'` no longer drops a Hamiltonian's alpha-free part when the class does not
expose `_potential` or `_coupling` (#2011). It refused such a Hamiltonian; it now solves it.

The switch that builds Howard's running-cost closure was keyed on those two `SeparableHamiltonian`
internals, so any other `HamiltonianBase` subclass carrying an `F(x, m)` had it discarded — the
issue measured `max|u(g=1) − u(g=0)| = 0.0000e+00` on Howard against `1.469e-01` on Newton for
`H = |p|²/2 + g·x·m`. Measured now: **1.4586e-01 on Howard against 1.4687e-01 on Newton**, a 0.7%
gap that is the two inner solvers' own discretisation difference.

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

On soundness: reading the alpha-free part as `H(x, m, 0, t)` is exact only when `H_control(0) = 0`,
and the file's own docstring named `H = sqrt(1+|p|²)` as the counterexample. Measured through the
guard, that Hamiltonian is refused with `_ke = 3.121e+03` against a tolerance of `8.0e-09` — six
orders of magnitude. What `_ke` cannot separate is `H_control(p) = ½|p|² + C` from a constant
potential `V = C`, and no probe can: they are the same function. Treating `H(0)` as the alpha-free
part is the standard normalisation.

Not addressed, and filed while measuring this: `inner_solver='howard'` hard-requires SOCP-precomputed
stencils for operators it could obtain from the Wendland-Taylor path the Newton solver already uses,
and that gate checks for the stencil *object* rather than for monotonicity — a `joint_socp` run logs
that SOCP-infeasible interior nodes "fall through to bare Wendland-Taylor LSQ (NON-MONOTONE)" and the
gate passes anyway. #2066.
