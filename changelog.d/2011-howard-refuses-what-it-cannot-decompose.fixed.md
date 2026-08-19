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
the problem's own `M_collocation`**, at three time slices, at the matching physical times, over five
momentum vectors spanning two magnitudes and random directions. It skips the alpha-free half when
`_potential` / `_coupling` are present, because that route wires it — an earlier version refused a
`HamiltonianBase` subclass that sets `_potential` and agrees with Newton to 2.30%. The tolerance is
**relative** to the probed `|H|`, because an absolute `1e-10` false-refuses an exact unit quadratic
whose alpha-free part cancels from terms of magnitude ≳ 5e5.

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
