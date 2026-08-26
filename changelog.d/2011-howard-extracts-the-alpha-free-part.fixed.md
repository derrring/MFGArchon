- **`inner_solver='howard'` no longer drops a Hamiltonian's alpha-free part when the class does not
  expose `_potential` or `_coupling`** (Issue #2011). It refused such a Hamiltonian; it now solves
  it. Before the fix, `max|u(g=1) − u(g=0)|` was `0.0000e+00` on Howard against `1.469e-01` on
  Newton for `H = |p|²/2 + g·x·m` — the alpha-free part was dropped bitwise.

  **No new extraction machinery was needed.** `howard_running_cost` already evaluated the
  Hamiltonian at `p = 0` through the same `eval_H_batch` the Newton residual uses; only the *switch*
  deciding whether to build that closure was keyed on `SeparableHamiltonian` internals. It is now a
  union with the guard's own measurement, so the change can only admit more than before.

  **What licenses it is an algebraic identity, not `H_control(0) = 0`.** Howard assembles
  `ref(∇u) + H(x, m, 0, t)` and the `_ke` gate certifies `ref(p) = H(p) − H(0)`, so the sum
  telescopes back to `H(∇u)` for whatever the extracted value is. `H_control(p) = ½|p|² + C` and a
  constant potential `V = C` are the same function and no probe separates them; the identity is why
  nothing has to.

  **`_ke` stays a refusal and is the real limit**: Howard substitutes a quadratic Lagrangian, and no
  probe recovers a control cost that is not quadratic. Its probe now covers every `M_collocation`
  time slice rather than three, because #2011 makes it the only gate — three sample points leave a
  zero-width hole that a defect vanishing at exactly those times sits in.
