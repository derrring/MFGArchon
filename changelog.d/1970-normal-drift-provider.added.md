`NormalDriftProvider` supplies the `alpha` of a Fokker-Planck no-flux wall.

An impermeable wall is `J . n = 0` with `J = m*v - D*grad(m)`, i.e. `v_n*m - D*d_n m = 0` — Robin
in `m` with `alpha = v_n`, `beta = -D`, `g = 0`. Imposing `d_n m = 0` instead is the same
condition only when the drift is tangential at the wall; otherwise mass leaves at a rate
proportional to `m_wall * v_n`. Measured on this package at a wall-normal drift of 3.2, the
non-conservative assembly loses **5.4% of the mass**.

`v_n = -c*d_n U` is a functional of the *coupled* solution, so it is known only per Picard
iterate — which is what a provider is for, and why this one sits on `alpha` rather than on
`value`: `value` is the homogeneous right-hand side, zero for an impermeable wall. It requires
the widening of 2a, since `alpha` was not a field a provider could reach.

Verified against the flux itself rather than another code path: over four value functions and
both walls, the ghost the shipped padding path returns leaves `|J . n| <= 2.2e-16`. A control
asserts that the pointwise `d_n m = 0` wall leaks on the same configuration, so the comparison is
not vacuous.

Sign convention is the **outward normal**, matching `BCSegment.beta` and every ghost formula
since #1907: at the low wall the outward normal is `-x`, so `v_n = -v_x` there. `c` has no
default — it is the Hamiltonian's control law (#1420), and a defaulted one silently rescales the
wall. An nD wall raises rather than guessing an axis, the same limit `AdjointConsistentProvider`
states (#624).
