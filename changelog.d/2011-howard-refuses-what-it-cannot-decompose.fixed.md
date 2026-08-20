`inner_solver='howard'` no longer solves a different problem when handed a Hamiltonian it cannot
decompose (#2011).

`_solve_backward_howard` reconstructs what Howard's policy evaluation needs by reading attributes
only `SeparableHamiltonian` has. Absent them, `control_lagrangian` stays `None` and `hjb_howard`
substitutes the **unit** quadratic whatever the actual control cost, while `has_H_extra` — keyed on
`_potential` / `_coupling` — drops the alpha-free part **bitwise**. The gate that would have caught
it sat behind `if control_cost is not None:`, so the class it was written to stop was the class that
skipped it.

**Three versions of this guard, and the first two failed the same way one level apart.** Recorded
because the shape recurs:

1. Keying on `getattr(H_class, "control_cost", None) is None` refused `H = |p|²/2` written as a bare
   subclass — a Hamiltonian the substitution is *exact* for. It broke seven existing tests whose
   fixture is that shape. An attribute standing in for a question about behaviour.
2. Probing behaviour, but at `m = ones`, `t = 0`, `p = e₀`, **accepted six wrong Hamiltonians**: any
   `f(m)` with `f(1) = 0`, `|p|²/(2m)` congestion with `c(1) = 1` (which the `_congestion_factor`
   gate twenty lines below refuses by name for the real class), `(1/2)|p|⁴` (agrees with the unit
   quadratic at both sampled points), and any anisotropy. A stand-in for the data standing in for
   the data.

The shipped guard probes `H(x, m, 0, t) == 0` and `H(x, m, p, t) − H(x, m, 0, t) == (1/2)|p|²` **on
the problem's own data**, at three `M_collocation` time slices, at the matching physical times, and
over momentum vectors whose MAGNITUDES are derived from the terminal datum rather than hard-coded.
It skips the alpha-free half when `_potential` / `_coupling` are present, because that route wires it
— an earlier version refused a `HamiltonianBase` subclass that sets `_potential` and agrees with
Newton to 2.30%.

Three corrections to earlier versions of this guard, each measured:

- **It ran only in the `else:` of the `control_cost` branch**, so any Hamiltonian exposing a
  `QuadraticControlCost` — the ordinary way to write one — skipped the whole gate. That is the third
  version repeating the same structural mistake one branch over: v1 keyed the *refusal* on the
  attribute, v3 keyed the *probe's scope* on it. The probe now runs for every Hamiltonian, which is
  only possible because its kinetic reference is the **declared** control cost when there is one:
  against a hard-coded `(1/2)|p|²` it would false-refuse `λ ≠ 1`, since
  `QuadraticControlCost(2.0)` gives `H(p=1) − H(0) = 0.25`. Verified: `λ` = 1.0, 2.0 and 0.5 are all
  accepted, and a `SeparableHamiltonian` subclass hiding an alpha-free term in `__call__` is now
  refused (Newton moves 7.3015e-01 on the same pair).
- **The momenta were hard-coded** at `|p| ∈ {0.5, 1, 2}` while the solve reaches `max|∇u| = 6.18`,
  so `H = (1/2)|p|² + C·max(0, |p|²−4)²` — convex, C¹, and wrong for Howard exactly where the solve
  lives — sat in the probe's null space at every `C`. Now refused at `C` = 0.05 and 0.001 (Newton
  moves 6.3180e-02 and 1.5349e-03). The scale comes from a spacing bound on the terminal datum, not
  from `_compute_gradient_at_point`: that accessor raises `KeyError('weights')` on this path, and a
  first attempt at this fix wrapped it in a bare `except` and silently fell back to 1.0 — so the
  widening never happened and the super-quadratic stayed accepted through a green run.
- **The accumulators used the builtin `max`**, and `max(0.0, nan)` is `0.0` because `nan > 0.0` is
  False. One non-finite value anywhere in the probe therefore zeroed the entire measurement and
  dropped the tolerance to its floor — a fail-silent inside the fail-loud guard, demonstrated on this
  file's own refuse-case fixture, where one NaN turned a correctly refused Hamiltonian into an
  accepted one and `_af` fell from 5.98 to exactly 0.0. Now `np.maximum`, which propagates. The tolerance is
**relative** to the probed `|H|`. The stated reason for that was wrong and is withdrawn: it claimed
relativity rescues an exact unit quadratic whose alpha-free part cancels from terms of magnitude
≳ 5e5. It does not. `_scale = max|H|` never sees a cancelling term — by definition the cancellation
does not reach the output — so `_scale` stays pinned at the kinetic value while the residue grows,
and relativity buys a factor of 2, not orders of magnitude. Measured on
`H = |p|²/2 + K·sin(3x)(1+m) − K·(sin(3x) + sin(3x)·m)`, algebraically the unit quadratic for every
`K`: `_scale` is 2.000000 at `K` = 0, 1e4, 1e5, 3e5, 5e5, 1e6 and 1e7, and the guard still refuses
from `K = 5e5` — the exact magnitude the withdrawn sentence gave as already handled.

Relativity is kept because scale-awareness is right in principle and harmless here, but the false
refusal at large cancelling `K` is **not fixed** by it and is recorded as open rather than claimed.
A tolerance that tracked the cancellation would need a term measured against the magnitudes *inside*
`H`, not against its output.

Verified, all eight: `BareUnitQuadratic` and `WiredPotential` accepted; hidden `f(1)=0` coupling,
`|p|²/(2m)`, `2 log m`, `2(m−1)`, `(1/2)|p|⁴` and `k=4` anisotropy all refused. Six of those were
accepted by version 2.

It **probes**; it does not extract. Reading the alpha-free part *as* `H(x, m, 0, t)` is unsound —
`H = sqrt(1+|p|²)` injects a spurious 1.0 — but a non-zero value there violates Howard's assumption
whichever term produced it, which is all a refusal needs. Correct admission still needs the
extraction and stays open on #2011.

The accept controls pass a non-trivial terminal condition and assert the field varies. With `u_T = 0`
the exact solution is identically zero, so a solver returning `np.zeros(...)` passed the earlier
`isfinite` assertion — verified by replacing the whole Howard sweep with zeros. And the first attempt
at *that* fix changed only the `U_terminal` argument while `MFGComponents` still hard-coded
`u_terminal = 0`, so the solve stayed zero and the control stayed vacuous; the positive control that
caught it is `ptp(u) = 2.0` with the terminal against `0.0` without.
