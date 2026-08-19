`inner_solver='howard'` no longer solves a different problem when handed a Hamiltonian it cannot
decompose (#2011).

`_solve_backward_howard` reconstructs what Howard's policy evaluation needs by reading attributes
only `SeparableHamiltonian` has. Absent them, two things happened silently:

- `control_lagrangian` stayed `None`, so `hjb_howard` substituted the **unit** quadratic
  `L(alpha) = (1/2)|alpha|^2`. Measured against Newton on `u_T = cos(2*pi*x)`: a `lambda = 2`
  subclass is **31.4%** wrong relative, against a **5.5% control** from a unit quadratic on the same
  problem — the control being the two inner solvers' own discretisation difference, so the signal is
  the gap, not the 31.4% alone.
- `has_H_extra` is keyed on `_potential` / `_coupling`, so the alpha-free part was dropped
  **bitwise**: `|u(g=1) - u(g=0)| = 0.000e+00` on Howard for `H = |p|^2/2 + g*x*m`, against
  `1.469e-01` for the same Hamiltonian on Newton.

The gate that would have caught it sat behind `if control_cost is not None:` — so the class it was
written to stop was the class that skipped it.

**The fix gates on behaviour, not on the attribute, and the first attempt did not.** Keying the
refusal on "exposes no `control_cost`" broke seven existing tests whose fixture is `H = |p|^2/2`
written as a bare subclass — a Hamiltonian Howard's substitution is *exact* for. That is the same
mistake one level down: a predicate on an attribute standing in for a question about behaviour. The
shipped guard probes the two assumptions directly:

```
H(x, m, 0, t) == 0                              no alpha-free part to drop, and H_control(0) = 0
H(x, m, p, t) - H(x, m, 0, t) == (1/2)|p|^2     the Lagrangian Howard will substitute
```

It **probes**; it does not extract. Reading the alpha-free part *as* `H(x, m, 0, t)` is unsound —
`H = sqrt(1+|p|^2)` injects a spurious 1.0 — but for a refusal that is irrelevant: a non-zero value
there violates Howard's assumption whichever term produced it. Admitting such a Hamiltonian
correctly still needs the extraction, and stays open on #2011.

Both halves are load-bearing, verified by mutation: dropping the alpha-free check lets
`H = |p|^2/2 + g*x*m` through (2 of 6 fail), dropping the kinetic check lets a `lambda = 2` cost
through whose alpha-free part is exactly zero (1 of 6 fail), dropping both fails 3 of 6. The
accept-side control — a bare unit-quadratic subclass — is the case the attribute-keyed version got
wrong, so it is pinned explicitly.
