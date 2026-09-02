The `regime_switching/non_negativity` capability cell no longer gates a proposition that is false of
a correct solver. Its verdict required each regime's mass to stay **constant** to 1e-6; in a
Markov-switching MFG the generator `Q` transfers mass between regimes by construction, so per-regime
mass follows `M(t) = M(0) expm(Q t)` and only the **sum** is conserved. Integrating the FP system
over a no-flux domain gives `dM/dt = M Q` directly. The cell was red on something no implementation
can satisfy — worse than no check, because it teaches a reader to ignore the report.

And the clause was not merely inert: it was **anti-correlated with correctness** on the axis it
named. The only way to hold per-regime mass constant is to not transfer mass at all, so a transposed
or sign-flipped transfer implementation would have **passed** the old gate with zero drift, and fails
the new one at 8.9e-02.

The same oracle is already asserted on this same fixture by
`test_regime_masses_track_the_markov_chain_closed_form` (#1802/#1906), at a tighter 5e-3 against a
different denominator. It predates this change and independently fixes the convention. This cell is
not a duplicate: the capability matrix records what a configuration does, and its verdict must also
read `_solved`, which a unit assertion does not.

**Replaced, not relaxed.** #1767 named tolerance-fitting as the route the matrix exists to catch, and
the previous `intended` note explicitly refused it. The quantity changed instead. The 8.88e-02 that
note called drift is almost entirely the physical transfer: at `t = 1` the measured masses are
0.594648 / 0.499156 against the oracle's 0.593323 / 0.498957.

**What is gated now:** `max_rel_vs_expm_oracle` — the external oracle, computed without reference to
the scheme — at 1e-2. The tolerance is bracketed on both sides by measurement rather than fitted to
one: it admits this fixture's 2.43e-03 and refuses the harness's own 10% density mutation
(1.11e-01).

That 2.43e-03 is a **Picard-lag floor, not a `dt` term**, and a draft of this fragment called it
first order in `dt`. At the shipped 3-sweep budget an 8× refinement removes 14% of it — 2.43e-03 at
`Nt = 10` against 2.10e-03 at `Nt = 80` — where first order predicts 87%.
`tests/unit/test_alg/test_regime_switching_iterator.py` already recorded exactly that beside its own
expm assertion: the residual is the lagged inflow plus the piecewise-constant-in-time source, and it
does not clean up under refinement at fixed iteration count.

A mass term had to remain in the gate. This cell is a member of `MASS_ORACLE_CELLS`, and that listing
is a claim its verdict reads mass — measured, non-negativity alone does **not** discriminate, the
same mutation moving `min_density` only from 1.087e-04 to 1.141e-04. Total-mass drift
(1.40e-03 → 1.02e-01 under mutation) is recorded rather than gated, because a leak in the sum shows
up as a departure from the oracle too.

**The cell stays FAIL, and no status moved** (FAIL=7, PASS=2, UNSUPPORTED=2, unchanged). It is now red
on `picard_converged: false` at its 3-sweep budget — the same axis as its FDM and SL siblings — rather
than on the per-regime mass question. `min_density` is 1.087e-04, strictly positive, so #1681 remains
fixed. Whether it converges at a real budget is unmeasured.

Regenerating the baseline dropped this cell's `intended` note, which is the carry-forward mechanism
working: a changed artifact is meant to make the cell read as unexplained so someone looks at it. The
baseline carries **nine** notes across eleven cells, so the **eight** others are what had to
survive; they are byte-identical, verified by hash, no status moved, and the unexplained count is
back to 0. (An earlier draft said "the other ten notes", which is the count of other *cells*.)
(#1798)
