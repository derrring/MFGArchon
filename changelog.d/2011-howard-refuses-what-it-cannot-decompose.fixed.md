`inner_solver='howard'` no longer solves a different problem when handed a Hamiltonian it cannot
decompose (#2011).

`_solve_backward_howard` reconstructs what Howard's policy evaluation needs by reading attributes
only `SeparableHamiltonian` has. Absent them, `control_lagrangian` stays `None` and `hjb_howard`
substitutes the **unit** quadratic whatever the actual control cost is — a `lambda = 2` subclass
came out 31.4% from Newton, against a 5.5% control from a genuine unit quadratic on the same
problem. The gate that would have caught it sat behind `if control_cost is not None:`, so the class
it was written to stop was the class that skipped it.

The shipped guard measures `H(x, m, 0, t)` and probes `H(x, m, p, t) − H(x, m, 0, t) == (1/2)|p|²`
**on the problem's own data** — every `M_collocation` time slice, at the matching physical times,
over momentum vectors whose magnitudes are derived from the terminal datum rather than hard-coded.
Only the second is a refusal; the first decides whether to build Howard's running-cost closure.

A probe that could not be **evaluated** is not a probe that passed: without an explicit finiteness
check, a single non-finite value anywhere in the sweep leaves the comparison as `nan > tol`, which
is `False`, so the guard accepts — the outcome it exists to prevent.
